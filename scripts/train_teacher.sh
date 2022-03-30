#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=SST-2
CLUSTER_NUM=2
BERT_BASE_DIR=$SCRATCH/huggingface/bert-base-uncased
TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
OUTPUT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/teacher-${CLUSTER_NUM}

mkdir $OUTPUT_DIR
    # --aug_train \
python $HOME/Pretrained-Language-Model/TinyBERT/train_teacher.py \
    --teacher_model ${BERT_BASE_DIR} \
    --data_dir ${TASK_DIR} \
    --task_name ${TASK_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --do_lower_case \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --eval_step 30 \
    --max_seq_length 128 \
    --train_batch_size 128 \
    --k ${CLUSTER_NUM} \
    --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_sst2_k${CLUSTER_NUM}.json;