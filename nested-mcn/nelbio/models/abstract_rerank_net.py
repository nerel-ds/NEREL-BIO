import os
from abc import ABC

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_url, cached_download
from transformers import AutoModel, AutoTokenizer, BertModel

from nelbio.biosyn.sparse_encoder import SparseEncoder


class AbstractRerankNet(ABC):
    nested_entity_encoder: nn.TransformerEncoder
    # use_cuda: bool
    bert_encoder: BertModel

    def bert_encode_query(self, query_input, device):
        batch_size, nested_depth, max_length = query_input['input_ids'].shape

        # candidate_s_scores = candidate_s_scores.cuda()
        query_input['input_ids'] = query_input['input_ids'].to(device)
        query_input['token_type_ids'] = query_input['token_type_ids'].to(device)
        query_input['attention_mask'] = query_input['attention_mask'].to(device)

        query_bert_embed = self.bert_encoder(
            input_ids=query_input['input_ids'].view(batch_size * nested_depth, max_length),
            token_type_ids=query_input['token_type_ids'].view(batch_size * nested_depth, max_length),
            attention_mask=query_input['attention_mask'].view(batch_size * nested_depth, max_length)
        )

        # Query embeddings. <batch_size, depth, hidden_size>
        query_bert_embed = query_bert_embed[0][:, 0].view((batch_size, nested_depth, -1))
        return query_bert_embed

    def bert_encode_flat_query(self, query_input, device):

        # candidate_s_scores = candidate_s_scores.cuda()
        query_input['input_ids'] = query_input['input_ids'].to(device)
        query_input['token_type_ids'] = query_input['token_type_ids'].to(device)
        query_input['attention_mask'] = query_input['attention_mask'].to(device)

        query_bert_embed = self.bert_encoder(
            input_ids=query_input['input_ids'],
            token_type_ids=query_input['token_type_ids'],
            attention_mask=query_input['attention_mask']
        )
        # <batch_size, hidden_size>
        query_bert_embed = query_bert_embed[0][:, 0]
        assert query_bert_embed.dim() == 2

        return query_bert_embed

    def bert_encode_candidates(self, candidate_input, device):
        batch_size, nested_depth, topk, max_length = candidate_input['input_ids'].shape

        candidate_input['input_ids'] = candidate_input['input_ids'].to(device)
        candidate_input['token_type_ids'] = candidate_input['token_type_ids'].to(device)
        candidate_input['attention_mask'] = candidate_input['attention_mask'].to(device)

        candidate_bert_embeds = self.bert_encoder(
            input_ids=candidate_input['input_ids'].view(batch_size * nested_depth * topk, max_length),
            token_type_ids=candidate_input['token_type_ids'].view(batch_size * nested_depth * topk, max_length),
            attention_mask=candidate_input['attention_mask'].view(batch_size * nested_depth * topk, max_length)

        )
        candidate_bert_embeds = candidate_bert_embeds[0][:, 0].view(batch_size, nested_depth, topk, -1)

        return candidate_bert_embeds

    def load_dense_encoder(self, model_name_or_path):
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

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

    def load_weight_by_name(self, model_name_or_path, weight_name):
        weight_path = os.path.join(model_name_or_path, f'{weight_name}_weight.pt')
        # check file exists
        if not os.path.isfile(weight_path):
            # download from huggingface hub and cache it
            weight_url = hf_hub_url(model_name_or_path, filename=f"{weight_name}_weight.pt")
            weight_path = cached_download(weight_url)
        weight = torch.load(weight_path)

        return weight

    def load_nested_encoder(self, model_path):
        transformer_encoder_path = os.path.join(model_path, "nested_encoder.pt")
        # save nested encoder
        self.nested_entity_encoder.load_state_dict(torch.load(transformer_encoder_path))
