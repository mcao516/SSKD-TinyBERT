#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=SST-2
PREDICTION_EVAL_STEP=200
INTERMEDIATE_EVAL_STEP=300
STUDENT_SIZE=4L
STUDENT_MODEL=General_TinyBERT_4L_312D  # General_TinyBERT_4L_312D
# TEACHER_TYPE=unsup-simcse-bert-base-uncased

CLASS_NUM=2
for SEED in 2 3 4
    do
    STUDENT_SIZE=4L-seed${SEED}

    for CLUSTER_NUM in 2 3 4 8 16 32
    do
        echo "- CLUSTER: ${CLUSTER_NUM}; SEED: ${SEED}"
        
        # 1. train teacher
        # BERT_BASE_DIR=$SCRATCH/huggingface/bert-base-uncased
        # TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
        # OUTPUT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/${TEACHER_TYPE}/teacher-${CLUSTER_NUM}

        # mkdir $OUTPUT_DIR
        #     # --aug_train \
        # python $HOME/Pretrained-Language-Model/TinyBERT/train_teacher.py \
        #     --teacher_model ${BERT_BASE_DIR} \
        #     --data_dir ${TASK_DIR} \
        #     --task_name ${TASK_NAME} \
        #     --output_dir ${OUTPUT_DIR} \
        #     --do_lower_case \
        #     --learning_rate 2e-5 \
        #     --num_train_epochs 10 \
        #     --eval_step ${PREDICTION_EVAL_STEP} \
        #     --max_seq_length 128 \
        #     --train_batch_size 128 \
        #     --k ${CLUSTER_NUM} \
        #     --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_sst2_k${CLUSTER_NUM}.json;
    

        # 2. intermediate distillation
        echo "[=== Intermediate Distillation (cluster: ${CLUSTER_NUM}) ===]"
        FT_BERT_BASE_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/${TEACHER_TYPE}/teacher-${CLUSTER_NUM}
        GENERAL_TINYBERT_DIR=$SCRATCH/${STUDENT_MODEL}
        TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
        TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/${STUDENT_SIZE}/intermediate-${CLUSTER_NUM}
        
        mkdir $TMP_TINYBERT_DIR
            # --aug_train \
            # init_student_from_scratch
        python $HOME/Pretrained-Language-Model/TinyBERT/task_distill.py \
            --teacher_model ${FT_BERT_BASE_DIR} \
            --student_model ${GENERAL_TINYBERT_DIR} \
            --data_dir ${TASK_DIR} \
            --task_name ${TASK_NAME} \
            --output_dir ${TMP_TINYBERT_DIR} \
            --max_seq_length 128 \
            --train_batch_size 256 \
            --num_train_epochs 10 \
            --eval_step ${INTERMEDIATE_EVAL_STEP} \
            --do_lower_case \
            --k ${CLUSTER_NUM} \
            --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_sst2_k${CLUSTER_NUM}.json;


        # 3. final layer distillation
        echo "[=== Prediction Layer Distillation (cluster: ${CLUSTER_NUM}) ===]"
        FT_BERT_BASE_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/${TEACHER_TYPE}/teacher-${CLUSTER_NUM}
        TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/${STUDENT_SIZE}/intermediate-${CLUSTER_NUM}
        TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
        TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/${STUDENT_SIZE}/final-${CLUSTER_NUM}
        
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
            --eval_step ${PREDICTION_EVAL_STEP} \
            --max_seq_length 128 \
            --train_batch_size 128 \
            --k ${CLUSTER_NUM} \
            --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_sst2_k${CLUSTER_NUM}.json;


        # 4. intermediate layer distillation (pretrained on cluster data)
        echo "[=== Intermediate Layer Distillation (pretrained on cluster: ${CLUSTER_NUM}) ===]"
        FT_BERT_BASE_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/${TEACHER_TYPE}/teacher-${CLASS_NUM}
        GENERAL_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/${STUDENT_SIZE}/final-${CLUSTER_NUM}
        TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
        TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/${STUDENT_SIZE}/intermediate-${CLASS_NUM}-seq${CLUSTER_NUM}

        mkdir $TMP_TINYBERT_DIR
            # --aug_train \
        python $HOME/Pretrained-Language-Model/TinyBERT/task_distill.py \
            --teacher_model ${FT_BERT_BASE_DIR} \
            --student_model ${GENERAL_TINYBERT_DIR} \
            --data_dir ${TASK_DIR} \
            --task_name ${TASK_NAME} \
            --output_dir ${TMP_TINYBERT_DIR} \
            --max_seq_length 128 \
            --train_batch_size 256 \
            --num_train_epochs 10 \
            --eval_step ${INTERMEDIATE_EVAL_STEP} \
            --do_lower_case \
            --k ${CLASS_NUM} \
            --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_sst2_k${CLASS_NUM}.json;


        # 5. final layer distillation (pretrained on cluster data)
        echo "[=== Final Layer Distillation (pretrained on cluster: ${CLUSTER_NUM}) ===]"
        FT_BERT_BASE_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/${TEACHER_TYPE}/teacher-${CLASS_NUM}
        TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/${STUDENT_SIZE}/intermediate-${CLASS_NUM}-seq${CLUSTER_NUM}
        TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
        TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/${STUDENT_SIZE}/final-${CLASS_NUM}-seq${CLUSTER_NUM}
        
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
            --eval_step ${PREDICTION_EVAL_STEP} \
            --max_seq_length 128 \
            --train_batch_size 128 \
            --k ${CLASS_NUM} \
            --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_sst2_k${CLASS_NUM}.json;

    done
done