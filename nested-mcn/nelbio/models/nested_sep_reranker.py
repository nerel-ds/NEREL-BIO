import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from nelbio.models.abstract_rerank_net import AbstractRerankNet
from nelbio.utils.utils import marginal_nll_no_mask

LOGGER = logging.getLogger(__name__)


class NestedScoreRerankNet(nn.Module, AbstractRerankNet):
    def __init__(self, bert_encoder, tokenizer, sparse_encoder, learning_rate, weight_decay,
                 flat_dense_weight, sep_dense_weight, sparse_weight, dense_encoder_score_type,
                 sep_pooling="cls", fixed_flat_dense_weight=False, fixed_flat_sparse_weight=False,
                 fixed_sep_dense_weight=False):
        LOGGER.info(f"RerankNet! learning_rate={learning_rate} weight_decay={weight_decay} sparse_weight="
                    f"{sparse_weight} flat_dense_weight={flat_dense_weight} sep_dense_weight={sep_dense_weight}")

        super(NestedScoreRerankNet, self).__init__()
        self.bert_encoder = bert_encoder
        self.tokenizer = tokenizer
        self.sparse_encoder = sparse_encoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.flat_dense_weight = self.init_weight(flat_dense_weight)
        if fixed_flat_dense_weight:
            self.flat_dense_weight.requires_grad = False
        self.sparse_weight = self.init_weight(sparse_weight)
        if fixed_flat_sparse_weight:
            self.sparse_weight.requires_grad = False
        self.sep_dense_weight = self.init_weight(sep_dense_weight)
        if fixed_sep_dense_weight:
            self.sep_dense_weight.requires_grad = False
        self.dense_encoder_score_type = dense_encoder_score_type
        self.sep_pooling = sep_pooling
        assert self.sep_pooling in ("cls", "mean")
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

        self.optimizer = optim.Adam([
            {'params': self.bert_encoder.parameters()},
            {'params': self.sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': self.flat_dense_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': self.sep_dense_weight, 'lr': 0.01, 'weight_decay': 0}],
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.criterion = marginal_nll_no_mask

    def init_weight(self, weight_value):
        weight = nn.Parameter(torch.empty(1))
        weight.data.fill_(weight_value)  # init sparse_weight

        return weight

    def mean_pooling(self, embs, att_mask):
        """
        :param embs: <batch, seq, h>
        :param att_mask: <batch, seq>
        :return:
        """
        # <batch, seq, h>
        input_mask_expanded = att_mask.unsqueeze(-1).expand(embs.size()).float()
        # <batch, h>
        sum_embeddings = torch.sum(embs * input_mask_expanded, 1)
        # <batch, h>
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def bert_encode(self, input_ids, att_mask, pooling="cls"):

        k = None
        if input_ids.dim() == 3:

            (batch_size, k, max_length) = input_ids.size()
        elif input_ids.dim() == 2:
            (batch_size, max_length) = input_ids.size()
        else:
            raise Exception(f"Invalid number of dims: {input_ids.dim()}")

        if k is not None:
            input_ids = input_ids.view((batch_size * k, max_length))
            att_mask = att_mask.view((batch_size * k, max_length))
        query_bert_embed = self.bert_encoder(
            input_ids=input_ids,
            attention_mask=att_mask,
        )

        # <batch_size, hidden_size>
        if pooling == "cls":
            query_bert_embed = query_bert_embed[0][:, 0]
        elif pooling == "mean":
            query_bert_embed = self.mean_pooling(query_bert_embed[0], att_mask)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        if k is not None:
            query_bert_embed = query_bert_embed.view((batch_size, k, -1))

        return query_bert_embed

    def calculate_dense_scores(self, query_emb, candidate_emb):
        if self.dense_encoder_score_type == "matmul":
            # <b, k> = <b, 1, h> x <b, k, h>
            candidate_d_score = torch.bmm(query_emb.unsqueeze(1), candidate_emb.permute(0, 2, 1)).squeeze(1)
        elif self.dense_encoder_score_type == "cosine":
            # <b, k> = cos(<b, 1, h>, <b, k, h>)
            candidate_d_score = self.cosine_sim(query_emb.unsqueeze(1), candidate_emb)
        else:
            raise Exception(f"Invalid dense_score_type: {self.dense_encoder_score_type}")
        return candidate_d_score

    def forward(self, query_token_input, candidate_input, sep_context_input, nested_sep_candidate_input,
                candidate_s_scores, nested_mask):
        """
        query : (N_entities, nested_depth, seq_length), candidates : (N_entities, nested_depth, topk, seq_length)

        output : (N, topk)
        """
        # <b, seq>
        query_inp_ids, query_att_mask = query_token_input
        query_inp_ids, query_att_mask = query_inp_ids.squeeze(1), query_att_mask.squeeze(1)
        # <b, k, seq>
        candi_inp_ids, candi_att_mask = candidate_input
        # <b, seq>
        sep_context_inp_ids, sep_context_att_mask = sep_context_input
        sep_context_inp_ids, sep_context_att_mask = sep_context_inp_ids.squeeze(1), sep_context_att_mask.squeeze(1)
        # <b, k, seq>
        nested_sep_candidate_inp_ids, nested_sep_candidate_att_mask = nested_sep_candidate_input

        batch_size, k, seq_length = nested_sep_candidate_inp_ids.size()

        query_emb = self.bert_encode(query_inp_ids, query_att_mask)
        candidate_emb = self.bert_encode(candi_inp_ids, candi_att_mask)
        sep_context_emb = self.bert_encode(sep_context_inp_ids, sep_context_att_mask, pooling=self.sep_pooling)
        sep_candidate_emb = self.bert_encode(nested_sep_candidate_inp_ids, nested_sep_candidate_att_mask,
                                             pooling=self.sep_pooling)
        assert query_emb.size() == sep_context_emb.size()
        assert candidate_emb.size() == sep_candidate_emb.size()

        # <b, k>
        flat_dense_score = self.calculate_dense_scores(query_emb=query_emb, candidate_emb=candidate_emb)

        sep_dense_score = self.calculate_dense_scores(query_emb=sep_context_emb, candidate_emb=sep_candidate_emb, )
        sep_dense_score = sep_dense_score * nested_mask.unsqueeze(-1)

        score = self.sparse_weight * candidate_s_scores + flat_dense_score * self.flat_dense_weight \
                + sep_dense_score * self.sep_dense_weight

        return score

    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates

    def get_loss(self, outputs, targets, device):
        targets = targets.to(device)
        loss = self.criterion(outputs, targets)
        return loss

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table

    def save_model(self, path):
        sparse_encoder_path = os.path.join(path, 'sparse_encoder.pk')
        sparse_weight_path = os.path.join(path, 'sparse_weight.pt')
        flat_dense_weight_path = os.path.join(path, 'flat_dense_weight.pt')
        sep_dense_weight_path = os.path.join(path, 'sep_dense_weight.pt')
        # transformer_encoder_path = os.path.join(path, "nested_encoder.pt")

        # save dense encoder
        self.bert_encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # save sparse encoder
        self.sparse_encoder.save_encoder(path=sparse_encoder_path)

        torch.save(self.sparse_weight, sparse_weight_path)
        torch.save(self.flat_dense_weight, flat_dense_weight_path)
        torch.save(self.sep_dense_weight, sep_dense_weight_path)
        logging.info("Sparse weight saved in {}".format(sparse_weight_path))
        logging.info("Flat dense weight saved in {}".format(flat_dense_weight_path))
        logging.info("Sep dense weight saved in {}".format(sep_dense_weight_path))

    def load_model(self, model_name_or_path):
        self.load_dense_encoder(model_name_or_path)
        self.load_sparse_encoder(model_name_or_path)
        print("model_name_or_path", model_name_or_path)
        self.sparse_weight = self.load_weight_by_name(model_name_or_path, "sparse")
        self.flat_dense_weight = self.load_weight_by_name(model_name_or_path, "flat_dense")
        self.sep_dense_weight = self.load_weight_by_name(model_name_or_path, "sep_dense")
        LOGGER.info(f"Loaded flat dense weight: {self.flat_dense_weight.data.item()}")
        LOGGER.info(f"Loaded sparse weight: {self.sparse_weight.data.item()}")
        LOGGER.info(f"Loaded sep dense weight: {self.sep_dense_weight.data.item()}")

        return self

    def calculate_context_query_vocab_scores(self, flat_bert_nested_query_embeddings, nested_entity_flat_index,
                                             vocab_embeddings, device, max_vocab_batch_size=512):
        # num_queries = len(query_embeddings)

        vocab_size = len(vocab_embeddings)
        num_nested_queries = len(nested_entity_flat_index)
        num_flattened_queries = len(flat_bert_nested_query_embeddings)
        max_nested_depth = max(t[1] - t[0] for t in nested_entity_flat_index)

        vocab_iterations = range(0, vocab_size, max_vocab_batch_size)
        self.nested_entity_encoder.eval()
        self.nested_entity_encoder.nested_entity_encoder.eval()
        # (start_ent_pos, end_ent_pos) = self.nested_entity_flat_index[query_idx],
        emb_size = flat_bert_nested_query_embeddings.size(-1)
        query_vocab_dense_score_matrix = np.zeros(shape=(num_flattened_queries, vocab_size), dtype=np.float32)

        for (start_ent_pos, end_ent_pos) in nested_entity_flat_index:
            n_nested = end_ent_pos - start_ent_pos
            nested_q_batch = torch.zeros(size=(max_vocab_batch_size, n_nested + 1, emb_size)).to(device)
            att_mask = torch.ones(size=(vocab_size, n_nested + 1), dtype=torch.float32).to(device)

            bert_nested_query_embeds = flat_bert_nested_query_embeddings[start_ent_pos:end_ent_pos]

            nested_q_batch[:, :n_nested, :] = bert_nested_query_embeds
            # print(f"nested_q_batch[:, :n_nested, :] {nested_q_batch[:, :n_nested, :].size()}")
            for vocab_start_pos in vocab_iterations:
                vocab_batch_size = min(max_vocab_batch_size, vocab_size - vocab_start_pos)
                vocab_end_pos = min(vocab_start_pos + vocab_batch_size, vocab_size)
                vocab_batch = vocab_embeddings[vocab_start_pos:vocab_end_pos]
                nested_q_batch[:vocab_batch_size, n_nested] = vocab_batch
                with torch.no_grad():
                    encoded_embs = self.nested_entity_encoder \
                        .nested_entity_encoder(nested_q_batch[:vocab_batch_size],
                                               src_key_padding_mask=att_mask[:vocab_batch_size])
                    context_vocab_embeds = encoded_embs[:, n_nested, :]
                    # <batch_size, nested_depth, emb_size>
                    context_query_embeds = encoded_embs[:, :n_nested, :]
                    assert context_vocab_embeds.size() == (vocab_batch_size, 1, emb_size) or \
                           context_vocab_embeds.size() == (vocab_batch_size, emb_size)
                    if context_vocab_embeds.dim() == 2:
                        # <batch_size, 1, emb_size>
                        context_vocab_embeds = context_vocab_embeds.unsqueeze(1)
                    # <batch_size, emb_size, 1>
                    context_vocab_embeds = context_vocab_embeds.permute(0, 2, 1)
                    # <batch_size, nested_depth>
                    scores = torch.bmm(context_query_embeds, context_vocab_embeds).detach().cpu().squeeze(-1)
                    # <nested_depth, batch_size>
                    scores = scores.t().numpy()
                    query_vocab_dense_score_matrix[start_ent_pos:end_ent_pos, vocab_start_pos:vocab_end_pos] = scores[
                                                                                                               :vocab_batch_size]

        return query_vocab_dense_score_matrix


def marginal_nll(score, target, mask):
    """
    sum all scores among positive samples
    """
    (batch_mul_depth, topk) = score.size()
    assert mask.size() == (batch_mul_depth, 1)
    # print(f"score: {score}")
    predict = F.softmax(score, dim=-1)
    with torch.no_grad():
        num_ones = int(torch.sum(mask).detach().cpu().item())

    loss = (predict[mask.repeat(1, topk) > 0] * target[mask.repeat(1, topk) > 0]).view(-1, topk)
    assert loss.size(0) == num_ones

    loss = loss.sum(dim=-1)  # sum all positive scores
    loss = loss[loss > 0]  # filter sets with at least one positives

    loss = torch.clamp(loss, min=1e-9, max=1)  # for numerical stability
    loss = -torch.log(loss)  # for negative log likelihood
    if len(loss) == 0:
        loss = loss.sum()  # will return zero loss
    else:
        loss = loss.mean()

    return loss
