# #!/bin/bash

declare -a datasets=("imagenet")
declare -a splits=("test")
# declare -a splits=("train")
declare -a plocs=(3)  # Array for ploc values
declare -a blocksizes=(2)  # Array for blocksize values
declare -a rankmodes=("similarity")  # Array for rank modes

# Other fixed parameters
cfg="vit_b16"
numtoken="100"

# Number of iterations to run
num_iterations=1

# Loop over the datasets
for dataset in "${datasets[@]}"; do
  # Loop over the split types
  for split in "${splits[@]}"; do
    # Loop over the ploc values
    for ploc in "${plocs[@]}"; do
      # Loop over the blocksize values
      for blocksize in "${blocksizes[@]}"; do
        # Loop over the rank modes
        for rankmode in "${rankmodes[@]}"; do
          # Run zeroshot ranking script only once per dataset, split, ploc, blocksize, and rankmode
          CUDA_VISIBLE_DEVICES=0 bash scripts/zsclip/zeroshot_rank.sh "${dataset}" "${cfg}" "${numtoken}" 1 "${split}" "${blocksize}" "${ploc}" "${rankmode}"

          # Loop over the iterations for the pruning script
          for ((i=1; i<=num_iterations; i++)); do
            echo "Iteration $i for Dataset: ${dataset}, Split: ${split}, Blocksize: ${blocksize}, Ploc: ${ploc}, Rankmode: ${rankmode}"

            # Run zeroshot pruning script
            CUDA_VISIBLE_DEVICES=0 bash scripts/zsclip/zeroshot_prune.sh "${dataset}" "${cfg}" "${numtoken}" "${i}" "${split}" "${blocksize}" "${ploc}" "${rankmode}"
          done
        done
      done
    done
  done
done