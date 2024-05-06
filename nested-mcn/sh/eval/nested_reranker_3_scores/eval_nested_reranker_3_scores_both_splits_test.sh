#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ep=10
echo "Processing checkpoint: ${ep} epochs, random split"
mkdir -p results/eval_nested_reranker_3_scores/random_split_test/e_${ep}/
python nelbio/evaluation/eval_nested_reranker_3_scores.py \
--model_name_or_path results/train_nested_reranker_3_scores/random_split/checkpoint_${ep}/ \
--model_config_path results/train_nested_reranker_3_scores/random_split/experiment_config.txt \
--dictionary_path "data/dictionary/vocab_umls_rus_biosyn.txt" \
--data_dir "data/biosyn_format/random_split/test/" \
--use_cuda \
--topk 20 \
--score_mode "hybrid" \
--query_max_length 32 \
--context_max_length 96 \
--output_dir results/eval_nested_reranker_3_scores/random_split_test/e_${ep} \
--filter_composite \
--filter_duplicate \
--save_predictions

ep=10
echo "Processing checkpoint: ${ep} epochs, zeroshot_split"
mkdir -p results/eval_nested_reranker_3_scores/zeroshot_split_test/e_${ep}/
python nelbio/evaluation/eval_nested_reranker_3_scores.py \
--model_name_or_path results/train_nested_reranker_3_scores/zeroshot_split/checkpoint_${ep}/ \
--model_config_path results/train_nested_reranker_3_scores/zeroshot_split/experiment_config.txt \
--dictionary_path "data/dictionary/vocab_umls_rus_biosyn.txt" \
--data_dir "data/biosyn_format/zeroshot_split/test/" \
--use_cuda \
--topk 20 \
--score_mode "hybrid" \
--query_max_length 32 \
--context_max_length 96 \
--output_dir results/eval_nested_reranker_3_scores/zeroshot_split_test/e_${ep} \
--filter_composite \
--filter_duplicate \
--save_predictions







