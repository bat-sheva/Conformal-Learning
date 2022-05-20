#!/bin/bash


source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate exp

python3 main_synthetic.py $delta_2 $seed
