#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

for ep in {1..15}
do
echo "Processing checkpoint: ${ep} epochs, dev"
mkdir -p results/eval_nested_reranker_3_scores/zeroshot_split_dev/e_${ep}
python nelbio/evaluation/eval_nested_reranker_3_scores.py \
--model_name_or_path results/train_nested_reranker_3_scores/zeroshot_split/checkpoint_${ep}/ \
--model_config_path results/train_nested_reranker_3_scores/zeroshot_split/experiment_config.txt \
--dictionary_path "data/dictionary/vocab_umls_rus_biosyn.txt" \
--data_dir "data/biosyn_format/zeroshot_split/dev" \
--use_cuda \
--topk 20 \
--score_mode "hybrid" \
--query_max_length 32 \
--context_max_length 96 \
--output_dir results/eval_nested_reranker_3_scores/zeroshot_split_dev/e_${ep} \
--filter_composite \
--filter_duplicate \
--save_predictions
done

