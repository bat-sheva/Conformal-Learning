#!/bin/bash
for seed in 1; do export seed; sbatch submit.slurm_cifar10; done 

##### To start training models or doing evaluations, specify the seeds above and submit this file

# For the results in the paper, we used the following seeds

# 800 87658 877 998888 76542 765183 2333 22142 1325 133215 (3k, 10k training data)

# 2 531 999 2333 6771 22145 46233 76542 98908 100001 (16.5k training data)

# 15 3891 21 130502 771 8 256 819 1955 97667 (27.5k training data)

# 22142 133215 998888 800 2 3 76542 1 877 4 (45k training data)