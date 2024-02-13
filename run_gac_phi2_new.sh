export CUDA_VISIBLE_DEVICES=1

for seed in 0
do
for TASK in 2
do
  python ./main.py \
    --model-name "phi2" \
    --algorithm 'gac' \
    --data-dir "./natural-instructions/tasks/" \
    --mode "Instruction Only" \
    --task-idx ${TASK} \
    --train-seed ${seed} \
    --data-seed 42 \
    --num-compose 1 \
    --num-candidates 2 \
    --num-offspring 5 \
    --num-iter 50 \
    --patience 14 \
    --write-preds \
    --meta-dir "./logs/" \
    --meta-name "PHI2-gac-${TASK}-seed-${seed}" \
    --print-orig \
    --agnostic \
    --batch-size 1 \
    --tournament-selection 2 \
    --project-name 'PHI2-gac-nolimt' \
    --backbone 'phi2' \
    --api-idx 33 \
    --level "phrase" \
    --checkpoint-freq 100000 \
    --budget 10000000000 \
    --output-dir './output'   # dir to save cheskpoints
    
    # --level "word" \
    # add the following argument to resume the searching from the chechpoint
    # --resume /home/szdiao/bbt/ours/grips_heuristicalgs/output/checkpoints/task0_step19.pickle" 

    # add the following arguments to test the performance of the loaded model
    # --model-dir /home/szdiao/bbt/ours/grips_heuristicalgs/output/checkpoints/task0_step19.pickle 
    # --eval-only
        # --simulated-anneal \
done
done