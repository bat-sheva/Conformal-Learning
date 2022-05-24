#!/bin/bash
#for a in 0.4 0.6 0.7 0.9 1
for delta_2 in 0.1 0.2 0.3 0.4 0.5
#for n_train in 1000 2000 2500 3000 4000
#for K in 4 6 8 10 12

do
for seed in {1..50}
do

sbatch -c2 --gres=gpu:0 -o slurm-test.out -J exp --export=delta_2=$delta_2,seed=$seed submit_synthetic.sh

done
done