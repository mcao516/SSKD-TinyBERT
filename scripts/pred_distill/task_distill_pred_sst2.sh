#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=SST-2
CLUSTER_NUM=2
FT_BERT_BASE_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/teacher-${CLUSTER_NUM}
TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/final-${CLUSTER_NUM}-test-scratch-sim512
TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/final-${CLUSTER_NUM}-test-scratch-sim512-pred

mkdir $TINYBERT_DIR
    # --aug_train \
    # --pred_distill \
    # --similarity_distill \
    # --sample_n_example 512 \
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
