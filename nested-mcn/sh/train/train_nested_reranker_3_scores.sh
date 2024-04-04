#!/bin/bash


OUTPUT_DIR=results/train_nested_reranker_3_scores/random_split/
mkdir -p ${OUTPUT_DIR}
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python nelbio/training/train_nested_reranker_3_scores.py \
--model_name_or_path "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--train_dictionary_path "data/dictionary/vocab_umls_rus_biosyn.txt" \
--train_dir "data/biosyn_format/random_split/train/" \
--output_dir ${OUTPUT_DIR} \
--query_max_length 32 \
--context_max_length 64 \
--dense_encoder_score_type "matmul" \
--sep_pooling "cls" \
--initial_sparse_weight 1. \
--initial_flat_dense_weight 1. \
--initial_sep_dense_weight 1. \
--fixed_flat_dense_weight \
--fixed_flat_sparse_weight \
--use_cuda \
--topk 20 \
--keep_longest_entity_only \
--save_checkpoint_all \
--epoch 15 \
--train_batch_size 16 \
--learning_rate 1e-5


OUTPUT_DIR=results/train_nested_reranker_3_scores/zeroshot_split/
mkdir -p ${OUTPUT_DIR}
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python nelbio/training/train_nested_reranker_3_scores.py \
--model_name_or_path "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--train_dictionary_path "data/dictionary/vocab_umls_rus_biosyn.txt" \
--train_dir "data/biosyn_format/zeroshot_split/train/" \
--output_dir ${OUTPUT_DIR} \
--query_max_length 32 \
--context_max_length 64 \
--dense_encoder_score_type "matmul" \
--sep_pooling "cls" \
--initial_sparse_weight 1. \
--initial_flat_dense_weight 1. \
--initial_sep_dense_weight 1. \
--fixed_flat_dense_weight \
--fixed_flat_sparse_weight \
--use_cuda \
--topk 20 \
--keep_longest_entity_only \
--save_checkpoint_all \
--epoch 15 \
--train_batch_size 16 \
--learning_rate 1e-5

