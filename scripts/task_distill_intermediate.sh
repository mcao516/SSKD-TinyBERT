#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=SST-2
CLUSTER_NUM=2
FT_BERT_BASE_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/teacher-${CLUSTER_NUM}
GENERAL_TINYBERT_DIR=$SCRATCH/General_TinyBERT_6L_768D
TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/intermediate-${CLUSTER_NUM}

mkdir $TMP_TINYBERT_DIR
    # --aug_train \
    # --init_student_from_scratch \
python $HOME/Pretrained-Language-Model/TinyBERT/task_distill.py \
    --teacher_model ${FT_BERT_BASE_DIR} \
    --student_model ${GENERAL_TINYBERT_DIR} \
    --data_dir ${TASK_DIR} \
    --task_name ${TASK_NAME} \
    --output_dir ${TMP_TINYBERT_DIR} \
    --max_seq_length 128 \
    --train_batch_size 64 \
    --num_train_epochs 10 \
    --eval_step 50 \
    --do_lower_case \
    --k ${CLUSTER_NUM} \
    --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_sst2_k${CLUSTER_NUM}.json;