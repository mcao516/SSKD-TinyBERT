#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=SST-2
CLUSTER_NUM=2
FT_BERT_BASE_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/teacher-${CLUSTER_NUM}
TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/intermediate-${CLUSTER_NUM}
TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/hinge-${CLUSTER_NUM}-margin_00

mkdir $TINYBERT_DIR
    # --aug_train \
    # --pred_distill \
    # --sample_n_example 256 \
    # --similarity_distill \
python $HOME/Pretrained-Language-Model/TinyBERT/task_distill.py \
    --margin 0.0 \
    --presentation_distill \
    --teacher_model ${FT_BERT_BASE_DIR} \
    --student_model ${TMP_TINYBERT_DIR} \
    --data_dir ${TASK_DIR} \
    --task_name ${TASK_NAME} \
    --output_dir ${TINYBERT_DIR} \
    --do_lower_case \
    --learning_rate 1e-6 \
    --num_train_epochs 3 \
    --eval_step 100 \
    --max_seq_length 128 \
    --train_batch_size 64 \
    --k ${CLUSTER_NUM} \
    --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_sst2_k${CLUSTER_NUM}.json;