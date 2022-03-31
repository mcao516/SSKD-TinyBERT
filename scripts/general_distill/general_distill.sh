#!/bin/bash
source ~/env37/bin/activate

# --continue_train \
python $HOME/Pretrained-Language-Model/TinyBERT/general_distill.py \
    --pregenerated_data $HOME/Pretrained-Language-Model/TinyBERT/pretrain_mrpc_retrieved_raw_36M_processed \
    --teacher_model $SCRATCH/huggingface/bert-base-uncased \
    --student_model $SCRATCH/General_TinyBERT_4L_312D \
    --output_dir $SCRATCH/TinyBERT_TEST/TinyBERTPretrainMRPC_36M_4L_Epoch50 \
    --do_lower_case \
    --max_seq_len 128 \
    --num_train_epochs 50;