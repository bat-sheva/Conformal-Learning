#!/bin/bash

for seed in {1..25}
do

sbatch -N1 -c1 --gres=gpu:A4000:1 -w plato2 -o slurm-test.out -J exp --export=seed=$seed submit_credit.sh

done
