#!/bin/bash


OUTPUT_DIR=results/train_mlp_reranker_frozen_biosyn/random_split/
mkdir -p ${OUTPUT_DIR}
export CUDA_VISIBLE_DEVICES=0
nvidia-smi

echo "Training on random split..."
python nelbio/training/train_nested_mlp_reranker_frozen_biosyn.py \
--pretrained_biosyn_path "results/train_biosyn/random_split/checkpoint_9/" \
--train_dictionary_path "data/dictionary/vocab_umls_rus_biosyn.txt" \
--train_dir "data/biosyn_format/random_split/train/" \
--output_dir ${OUTPUT_DIR} \
--query_max_length 32 \
--context_max_length 128 \
--dense_encoder_score_type "matmul" \
--sep_pooling "cls" \
--drop_cuiless \
--use_cuda \
--topk 20 \
--keep_longest_entity_only \
--save_checkpoint_all \
--epoch 10 \
--train_batch_size 16 \
--learning_rate 1e-5

echo "Finished training on zeroshot split..."

OUTPUT_DIR=results/train_mlp_reranker_frozen_biosyn/zeroshot_split/
mkdir -p ${OUTPUT_DIR}
echo "Training on zero-shot split..."
python nelbio/training/train_nested_mlp_reranker_frozen_biosyn.py \
--pretrained_biosyn_path "results/train_biosyn/zeroshot_split/checkpoint_9/" \
--train_dictionary_path "data/dictionary/vocab_umls_rus_biosyn.txt" \
--train_dir "data/biosyn_format/zeroshot_split/train/" \
--output_dir ${OUTPUT_DIR} \
--query_max_length 32 \
--context_max_length 128 \
--dense_encoder_score_type "matmul" \
--sep_pooling "cls" \
--drop_cuiless \
--use_cuda \
--topk 20 \
--keep_longest_entity_only \
--save_checkpoint_all \
--epoch 10 \
--train_batch_size 16 \
--learning_rate 1e-5

