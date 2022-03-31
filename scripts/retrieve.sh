#!/bin/bash
module load cuda/11.0
module load python/3.8
source $HOME/envFS/bin/activate

python $HOME/SSKD-TinyBERT/retrieve.py \
    --task_name mrpc \
    --data_dir $SCRATCH/glue_data/MRPC \
    --output_file $HOME/SSKD-TinyBERT/general_distill_corpus/mrpc_retrived.txt \
    --k 20;