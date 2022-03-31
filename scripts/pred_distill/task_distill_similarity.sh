#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=MNLI
FT_BERT_BASE_DIR=$SCRATCH/huggingface/MNLI/uncased/
TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/intermediate 
TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/similarity-256  # output directory

mkdir $TINYBERT_DIR
python $HOME/Pretrained-Language-Model/TinyBERT/task_distill.py \
    --similarity_distill \
    --teacher_model ${FT_BERT_BASE_DIR} \
    --student_model ${TMP_TINYBERT_DIR} \
    --data_dir ${TASK_DIR} \
    --task_name ${TASK_NAME} \
    --output_dir ${TINYBERT_DIR} \
    --aug_train \
    --do_lower_case \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --eval_step 1000 \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --sample_n_example 256;
