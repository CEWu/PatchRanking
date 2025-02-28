#!/bin/bash

#cd ../..

# custom config
DATA=data
TRAINER=ZeroshotCLIP_rank
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
SEED=1
PLOC=[$7]
PLOC_VALUE=${PLOC#[}
PLOC_VALUE=${PLOC_VALUE%]}
NUMTOKEN=[$3]
ITER=$4
SPLIT=$5
BLOCKSIZE=$6
RANKMODE=$8
ERANK=False
EPRUNE=True
ISPRUNE=True
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
--eval-prune \
TRAINER.COOP.PRUNING_LOC ${PLOC} \
TRAINER.COOP.NUM_TOKEN ${NUMTOKEN} \
TRAINER.COOP.EVAL_PRUNE ${EPRUNE} \
TRAINER.COOP.TOKENS_SCORE ${TSCORE} \
TRAINER.COOP.IS_PRUNE ${ISPRUNE} \
TRAINER.COOP.RANK_MODE ${RANKMODE} \
DATALOADER.TEST.BATCH_SIZE 64 \
DATALOADER.TRAIN_X.BATCH_SIZE 64

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
