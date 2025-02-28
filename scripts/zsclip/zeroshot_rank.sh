#!/bin/bash

#cd ../..

# custom config
DATA=data
TRAINER=ZeroshotCLIP_rank
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
ITER=$4
SPLIT=$5
BLOCKSIZE=$6
SEED=1
PLOC=[$7]
PLOC_VALUE=${PLOC#[}
PLOC_VALUE=${PLOC_VALUE%]}
ERANK=True
RANKMODE=$8
NUMTOKEN=[$3] 
VALUE=${NUMTOKEN#[}
VALUE=${VALUE%]}

if [[ "$RANKMODE" != "label" && "$RANKMODE" != "max" && "$RANKMODE" != "similarity" ]]; then
  echo "Error: RANKMODE must be 'label', 'max', or 'similarity'."
  exit 1
fi

TSCORE="/home/cwu/Workspace/GoldenPruningCLIP/output/ZeroshotCLIP_rank/vit_b16/${SPLIT}_${RANKMODE}_${BLOCKSIZE}x${BLOCKSIZE}_${PLOC_VALUE}/${DATASET}/tokens_rank/${DATASET}_tokens_rank_${VALUE}token_$((ITER-1))iter.json"

DIR="/home/cwu/Workspace/GoldenPruningCLIP/output/ZeroshotCLIP_rank/vit_b16/${SPLIT}_${RANKMODE}_${BLOCKSIZE}x${BLOCKSIZE}_${PLOC_VALUE}/${DATASET}/tokens_rank/"

if [ ! -d "$DIR" ]; then
  mkdir -p "$DIR"
fi

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${SPLIT}_${RANKMODE}_${BLOCKSIZE}x${BLOCKSIZE}_${PLOC_VALUE}/${DATASET} \
--split ${SPLIT} \
--eval-rank \
TRAINER.COOP.PRUNING_LOC ${PLOC} \
TRAINER.COOP.NUM_TOKEN ${NUMTOKEN} \
TRAINER.COOP.EVAL_RANK ${ERANK} \
TRAINER.COOP.TOKENS_SCORE ${TSCORE} \
TRAINER.COOP.BLOCK_SIZE ${BLOCKSIZE} \
TRAINER.COOP.RANK_MODE ${RANKMODE} \
DATALOADER.TEST.BATCH_SIZE 1 \
DATALOADER.TRAIN_X.BATCH_SIZE 1

