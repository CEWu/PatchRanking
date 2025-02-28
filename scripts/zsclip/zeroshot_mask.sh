#!/bin/bash

#cd ../..

# custom config
DATA=data
TRAINER=ZeroshotCLIP_rank
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
SEED=1
PLOC=[3]
NUMTOKEN=[$3]
ERANK=False
EPRUNE=False
EMASK=True
VALUE=${NUMTOKEN#[}
VALUE=${VALUE%]}
TSCORE=/home/cwu/Workspace/GoldenPruningCLIP/output/ZeroshotCLIP_rank/vit_b16/${DATASET}/tokens_rank/${DATASET}_tokens_rank_${VALUE}token.json


python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-mask \
TRAINER.COOP.PRUNING_LOC ${PLOC} \
TRAINER.COOP.NUM_TOKEN ${NUMTOKEN} \
TRAINER.COOP.EVAL_MASK ${EMASK} \
TRAINER.COOP.TOKENS_SCORE ${TSCORE} \
DATALOADER.TEST.BATCH_SIZE 100

# Copy the source to destination
# cp "$TSCORE" "$TSCOREBACK"


# for TRATIO in 0.9; do
#     TRATIO_LIST="[$TRATIO]"
#     python train.py \
#     --root ${DATA} \
#     --seed ${SEED} \
#     --trainer ${TRAINER} \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/CoOp/${CFG}.yaml \
#     --output-dir output/${TRAINER}/${CFG}/${DATASET} \
#     --eval-prune \
#     TRAINER.COOP.PRUNING_LOC ${PLOC} \
#     TRAINER.COOP.TOKEN_RATIO ${TRATIO_LIST} \
#     TRAINER.COOP.EVAL_PRUNE ${EPRUNE} \
#     TRAINER.COOP.TOKENS_SCORE ${TSCORE} \
#     TRAINER.COOP.PRUNED_TOKENS_ID ${TOKENID}
# done
