#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=MRPC
CLUSTER_NUM=2
FT_BERT_BASE_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/teacher-${CLUSTER_NUM}
TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/intermediate-4L-TinyBERTPretrainMRPC_36M_4L_Epoch50-${CLUSTER_NUM}
TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/final-4L-TinyBERTPretrainMRPC_36M_4L_Epoch50-${CLUSTER_NUM}

mkdir $TINYBERT_DIR
    # --aug_train \
python $HOME/Pretrained-Language-Model/TinyBERT/task_distill.py \
    --pred_distill \
    --teacher_model ${FT_BERT_BASE_DIR} \
    --student_model ${TMP_TINYBERT_DIR} \
    --data_dir ${TASK_DIR} \
    --task_name ${TASK_NAME} \
    --output_dir ${TINYBERT_DIR} \
    --do_lower_case \
    --learning_rate 1e-6 \
    --num_train_epochs 5 \
    --eval_step 30 \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --k ${CLUSTER_NUM} \
    --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_mrpc_k${CLUSTER_NUM}.json;