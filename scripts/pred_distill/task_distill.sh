#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=SST-2

for SEQ in 3
do
    # 1. train teacher
    echo "- CLUSTER: ${SEQ}"
    CLUSTER_NUM=2
    # BERT_BASE_DIR=$SCRATCH/huggingface/bert-base-uncased
    # TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
    # OUTPUT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/teacher-${CLUSTER_NUM}

    # mkdir $OUTPUT_DIR
    #     # --aug_train \
    # python $HOME/Pretrained-Language-Model/TinyBERT/train_teacher.py \
    #     --teacher_model ${BERT_BASE_DIR} \
    #     --data_dir ${TASK_DIR} \
    #     --task_name ${TASK_NAME} \
    #     --output_dir ${OUTPUT_DIR} \
    #     --do_lower_case \
    #     --learning_rate 3e-5 \
    #     --num_train_epochs 10 \
    #     --eval_step 30 \
    #     --max_seq_length 128 \
    #     --train_batch_size 128 \
    #     --k ${CLUSTER_NUM} \
    #     --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_sst2_k${CLUSTER_NUM}.json;

    # 2. intermediate distillation
    FT_BERT_BASE_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/teacher-${CLUSTER_NUM}
    GENERAL_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/intermediate-4L-${SEQ}
    TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
    TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/intermediate-4L-${CLUSTER_NUM}-seq${SEQ}
    
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
        --eval_step 500 \
        --do_lower_case \
        --k ${CLUSTER_NUM} \
        --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_sst2_k${CLUSTER_NUM}.json;
    
    # 3. final layer distillation
    FT_BERT_BASE_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/teacher-${CLUSTER_NUM}
    TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/intermediate-4L-${CLUSTER_NUM}-seq${SEQ}
    TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
    TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/final-4L-${CLUSTER_NUM}-seq${SEQ}
    
    mkdir $TINYBERT_DIR
        #   --aug_train \
    python $HOME/Pretrained-Language-Model/TinyBERT/task_distill.py \
        --pred_distill \
        --teacher_model ${FT_BERT_BASE_DIR} \
        --student_model ${TMP_TINYBERT_DIR} \
        --data_dir ${TASK_DIR} \
        --task_name ${TASK_NAME} \
        --output_dir ${TINYBERT_DIR} \
        --do_lower_case \
        --learning_rate 3e-5 \
        --num_train_epochs 3 \
        --eval_step 100 \
        --max_seq_length 128 \
        --train_batch_size 128 \
        --k ${CLUSTER_NUM} \
        --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_sst2_k${CLUSTER_NUM}.json;
done
