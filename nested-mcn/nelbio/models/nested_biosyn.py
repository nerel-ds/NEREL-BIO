import logging
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_url, cached_download
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    default_data_collator
)

from nelbio.biosyn.sparse_encoder import SparseEncoder

LOGGER = logging.getLogger()


class NamesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class NestedBioSyn(object):
    """
    Wrapper class for dense encoder and sparse encoder
    """

    def __init__(self, max_length, use_cuda, initial_sparse_weight=None):
        self.max_length = max_length
        self.use_cuda = use_cuda

        self.tokenizer = None
        self.encoder = None
        self.sparse_encoder = None
        self.sparse_weight = None

        if initial_sparse_weight != None:
            self.sparse_weight = self.init_sparse_weight(initial_sparse_weight)

    def init_sparse_weight(self, initial_sparse_weight):
        """
        Parameters
        ----------
        initial_sparse_weight : float
            initial sparse weight
        """
        if self.use_cuda:
            self.sparse_weight = nn.Parameter(torch.empty(1).cuda())
        else:
            self.sparse_weight = nn.Parameter(torch.empty(1))
        self.sparse_weight.data.fill_(initial_sparse_weight)  # init sparse_weight

        return self.sparse_weight

    def init_sparse_encoder(self, corpus):
        self.sparse_encoder = SparseEncoder().fit(corpus)

        return self.sparse_encoder

    def get_dense_encoder(self):
        assert (self.encoder is not None)

        return self.encoder

    def get_dense_tokenizer(self):
        assert (self.tokenizer is not None)

        return self.tokenizer

    def get_sparse_encoder(self):
        assert (self.sparse_encoder is not None)

        return self.sparse_encoder

    def get_sparse_weight(self):
        assert (self.sparse_weight is not None)

        return self.sparse_weight

    def save_model(self, path):
        # save dense encoder
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # save sparse encoder
        sparse_encoder_path = os.path.join(path, 'sparse_encoder.pk')
        self.sparse_encoder.save_encoder(path=sparse_encoder_path)

        sparse_weight_file = os.path.join(path, 'sparse_weight.pt')
        torch.save(self.sparse_weight, sparse_weight_file)
        logging.info("Sparse weight saved in {}".format(sparse_weight_file))

    def load_model(self, model_name_or_path):
        self.load_dense_encoder(model_name_or_path)
        self.load_sparse_encoder(model_name_or_path)
        self.load_sparse_weight(model_name_or_path)

        return self

    def load_dense_encoder(self, model_name_or_path):
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.use_cuda:
            self.encoder = self.encoder.to("cuda")

        return self.encoder, self.tokenizer

    def load_sparse_encoder(self, model_name_or_path):
        sparse_encoder_path = os.path.join(model_name_or_path, 'sparse_encoder.pk')
        # check file exists
        if not os.path.isfile(sparse_encoder_path):
            # download from huggingface hub and cache it
            sparse_encoder_url = hf_hub_url(model_name_or_path, filename="sparse_encoder.pk")
            sparse_encoder_path = cached_download(sparse_encoder_url)

        self.sparse_encoder = SparseEncoder().load_encoder(path=sparse_encoder_path)

        return self.sparse_encoder

    def load_sparse_weight(self, model_name_or_path):
        sparse_weight_path = os.path.join(model_name_or_path, 'sparse_weight.pt')
        # check file exists
        if not os.path.isfile(sparse_weight_path):
            # download from huggingface hub and cache it
            sparse_weight_url = hf_hub_url(model_name_or_path, filename="sparse_weight.pt")
            sparse_weight_path = cached_download(sparse_weight_url)

        self.sparse_weight = torch.load(sparse_weight_path)

        return self.sparse_weight

    def get_score_matrix_torch(self, query_embeds, dict_embeds, device):
        """
        Return score matrix
        Parameters
        ----------
        query_embeds : np.array
            2d numpy array of query embeddings
        dict_embeds : np.array
            2d numpy array of query embeddings
        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        with torch.no_grad():
            score_matrix = torch.matmul(query_embeds.to(device), dict_embeds.T.to(device)).detach().cpu().numpy()

        return score_matrix

    def get_torch_query_dict_score_matrix(self, query_names, vocab_names, device, vocab_batch_size=256,
                                          show_progress=False, dense_encoder_score_type="matmul"):
        """
        Return score matrix
        Parameters
        ----------
        query_embeds : np.array
            2d numpy array of query embeddings
        dict_embeds : np.array
            2d numpy array of query embeddings
        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        num_queries = len(query_names)
        vocab_length = len(vocab_names)

        sparse_score_matrix = np.zeros(shape=(num_queries, vocab_length), dtype=float)

        # (query_batch_size +  * 4k
        # query_batch_size = 1024

        if show_progress:
            vocab_iterations = tqdm(range(0, vocab_length, vocab_batch_size))
        else:
            vocab_iterations = range(0, vocab_length, vocab_batch_size)
        query_embeddings = torch.from_numpy(self.sparse_encoder(query_names).numpy()).to(device)

        for vocab_start_pos in vocab_iterations:
            vocab_end_pos = min(vocab_start_pos + vocab_batch_size, vocab_length)
            batch_vocab_names = vocab_names[vocab_start_pos:vocab_end_pos]
            batch_vocab_embeddings = torch.from_numpy(self.sparse_encoder(batch_vocab_names).numpy()).to(device)
            with torch.no_grad():
                # query_embeddings -- <num_queries, emb_size>
                # batch_vocab_embeddings -- <batch_size, emb_size>
                if dense_encoder_score_type == "matmul":
                    batch_score_matrix = torch.matmul(query_embeddings, batch_vocab_embeddings.t()).detach().cpu().numpy()
                elif dense_encoder_score_type == "cosine":
                    batch_score_matrix = F.cosine_similarity(query_embeddings.unsqueeze(1),
                                                             batch_vocab_embeddings.unsqueeze(0), dim=-1).detach().cpu().numpy()
                assert batch_score_matrix.shape == (num_queries, vocab_end_pos - vocab_start_pos)
                sparse_score_matrix[:, vocab_start_pos:vocab_end_pos] = batch_score_matrix

        del query_embeddings
        torch.cuda.empty_cache()

        return sparse_score_matrix

    def get_score_matrix(self, query_embeds, dict_embeds, dense_encoder_score_type="matmul"):
        """
        Return score matrix
        Parameters
        ----------
        query_embeds : np.array <num_queries, emb_size>
            2d numpy array of query embeddings
        dict_embeds : np.array <dict_size, emb_size>
            2d numpy array of query embeddings
        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """

        if dense_encoder_score_type == "matmul":
            score_matrix = np.matmul(query_embeds, dict_embeds.T)
        elif dense_encoder_score_type == "cosine":
            score_matrix = cosine_similarity(query_embeds, dict_embeds)
        else:
            raise Exception(f"Invalid dense_encoder_score_type: {dense_encoder_score_type}")

        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):
        """
        Return sorted topk idxes (descending order)
        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates
        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """

        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0, cols.shape[0])[:, np.newaxis], cols.shape[1], axis=1)
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix, -topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix)
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def embed_sparse(self, names, show_progress=False):
        """
        Embedding data into sparse representations
        Parameters
        ----------
        names : np.array
            An array of names
        Returns
        -------
        sparse_embeds : np.array
            A list of sparse embeddings
        """
        batch_size = 1024
        sparse_embeds = []

        if show_progress:
            iterations = tqdm(range(0, len(names), batch_size))
        else:
            iterations = range(0, len(names), batch_size)

        for start in iterations:
            end = min(start + batch_size, len(names))
            batch = names[start:end]
            batch_sparse_embeds = self.sparse_encoder(batch)
            batch_sparse_embeds = batch_sparse_embeds.numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = torch.from_numpy(np.concatenate(sparse_embeds, axis=0), )

        return sparse_embeds

    def embed_dense(self, names, show_progress=False):
        """
        Embedding data into dense representations

        Parameters
        ----------
        names : np.array or list
            An array of names

        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """
        self.encoder.eval()  # prevent dropout

        batch_size = 1024
        dense_embeds = []

        if isinstance(names, np.ndarray):
            names = names.tolist()
        name_encodings = self.tokenizer(names, padding="max_length", max_length=self.max_length,
                                        return_token_type_ids=True, truncation=True, return_tensors="pt")
        if self.use_cuda:
            name_encodings = name_encodings.to('cuda')
        name_dataset = NamesDataset(name_encodings)
        name_dataloader = DataLoader(name_dataset, shuffle=False, collate_fn=default_data_collator,
                                     batch_size=batch_size)

        with torch.no_grad():
            for batch in tqdm(name_dataloader, disable=not show_progress, desc='embedding dictionary'):
                outputs = self.encoder(**batch)
                batch_dense_embeds = outputs[0][:, 0].cpu().detach().numpy()  # [CLS] representations
                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)

        return dense_embeds
