#!/bin/bash


source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate conf

python3 main_credit.py $seed
