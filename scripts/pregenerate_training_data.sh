#!/bin/bash
source ~/env37/bin/activate

python $HOME/Pretrained-Language-Model/TinyBERT/pregenerate_training_data.py \
    --train_corpus $HOME/Pretrained-Language-Model/TinyBERT/pretrain_mrpc_retrieved_raw_36M.txt \
    --output_dir $HOME/Pretrained-Language-Model/TinyBERT/pretrain_mrpc_retrieved_raw_36M_processed \
    --bert_model $SCRATCH/huggingface/bert-base-uncased \
    --do_lower_case \
    --epochs_to_generate 50 \
    --max_seq_len 128;