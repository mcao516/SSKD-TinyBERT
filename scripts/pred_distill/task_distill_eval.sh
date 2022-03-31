#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=SST-2
TINYBERT_DIR=$SCRATCH/6L_768D_FinalModel/SST-2
TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
OUTPUT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/eval  # output directory

mkdir $OUTPUT_DIR
python $HOME/Pretrained-Language-Model/TinyBERT/task_distill.py \
    --do_eval \
    --student_model ${TINYBERT_DIR} \
    --data_dir ${TASK_DIR} \
    --task_name ${TASK_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --do_lower_case \
    --eval_batch_size 64 \
    --max_seq_length 128;
