## Install example requirements
#pip install -r ../requirements.txt

## Download glue data
#python3 ../../utils/download_glue_data.py
export GPU
export TASK=mrpc
export TASK_NAME=MNLI
#export DATA_DIR=./glue_data/MRPC/
export MAX_LENGTH=128
export LEARNING_RATE=2e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR_NAME=mnli-for-bert
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
## bash
# shellcheck disable=SC1068
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

## Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
## Add parent directory to python path to access lightning_base.py
#export PYTHONPATH="../":"${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 python -u $CURRENT_DIR/scripts/run_glue.py \
--task $TASK \
--task_name $TASK_NAME \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--learning_rate $LEARNING_RATE \
--num_train_epochs $NUM_EPOCHS \
--seed $SEED \
--do_train \
--do_predict
