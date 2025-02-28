#!/bin/bash

#cd ../..

# custom config
DATA=data
TRAINER=ZeroshotCLIP_rank
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
ITER=$4
SEED=1
PLOC=[0]
ERANK=True
NUMTOKEN=[0] 
VALUE=${NUMTOKEN#[}
VALUE=${VALUE%]}
TSCORE="/home/cwu/Workspace/GoldenPruningCLIP/output/ZeroshotCLIP_rank/vit_b16/${DATASET}/tokens_rank/${DATASET}_tokens_rank_${VALUE}token_$((ITER-1))iter.json"

DIR="/home/cwu/Workspace/GoldenPruningCLIP/output/ZeroshotCLIP_rank/vit_b16/${DATASET}/tokens_rank/"

if [ ! -d "$DIR" ]; then
  mkdir -p "$DIR"
fi

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only \
TRAINER.COOP.PRUNING_LOC ${PLOC} \
TRAINER.COOP.NUM_TOKEN ${NUMTOKEN} \
TRAINER.COOP.EVAL_RANK ${ERANK} \
TRAINER.COOP.TOKENS_SCORE ${TSCORE} \
DATALOADER.TEST.BATCH_SIZE 1

