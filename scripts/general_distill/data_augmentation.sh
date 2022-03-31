#!/bin/bash
source ~/env37/bin/activate

BERT_BASE_DIR=$SCRATCH/huggingface/bert-base-uncased
GLOVE_EMB=$SCRATCH/GloVe/glove.6B.300d.txt
GLUE_DIR=$SCRATCH/glue_data/
TASK_NAME=SST-2

python $HOME/Pretrained-Language-Model/TinyBERT/data_augmentation.py \
    --pretrained_bert_model $BERT_BASE_DIR \
    --glove_embs $GLOVE_EMB \
    --glue_dir $GLUE_DIR \
    --task_name $TASK_NAME;