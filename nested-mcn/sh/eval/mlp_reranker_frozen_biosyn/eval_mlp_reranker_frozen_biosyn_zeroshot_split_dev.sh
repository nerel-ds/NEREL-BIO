#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

for ep in {1..10}
do
echo "Processing checkpoint: ${ep} epochs, dev"
mkdir -p results/eval_mlp_reranker_frozen_biosyn/zeroshot_split_dev/e_${ep}
python nelbio/evaluation/eval_mlp_reranker_frozen_biosyn.py \
--model_name_or_path results/train_mlp_reranker_frozen_biosyn/zeroshot_split/checkpoint_${ep}/ \
--model_config_path results/train_mlp_reranker_frozen_biosyn/zeroshot_split/experiment_config.txt \
--dictionary_path "data/dictionary/vocab_umls_rus_biosyn.txt" \
--data_dir "data/biosyn_format/zeroshot_split/dev" \
--use_cuda \
--topk 20 \
--score_mode "hybrid" \
--query_max_length 32 \
--context_max_length 96 \
--output_dir results/eval_mlp_reranker_frozen_biosyn/zeroshot_split_dev/e_${ep} \
--filter_composite \
--filter_duplicate \
--drop_cuiless \
--save_predictions

done