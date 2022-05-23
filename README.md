## Training Uncertainty-Aware Classifiers with Conformalized Deep Learning

This repository contains code accompanying the following paper: "Training Uncertainty-Aware Classifiers with Conformalized Deep Learning".
The contents of this repository include a Python package implementing the conformalized training method described in the paper, an implementation of the benchmark methods described therein, and code to reproduce the experiments with synthetic and real data.

### Abstract

Deep neural networks are powerful tools to detect hidden patterns in data and leverage them to make predictions, 
but they are not designed to understand uncertainty and estimate reliable probabilities. 
In particular, they tend to be overconfident. We address this problem by developing novel 
training algorithms that can lead to more dependable uncertainty estimates, without 
sacrificing predictive power. The idea is to mitigate overconfidence by minimizing a 
loss function inspired by advances in conformal inference that quantifies model uncertainty 
by carefully leveraging hold-out data. Experiments with synthetic and real data 
demonstrate this method leads to smaller prediction sets with higher conditional coverage, 
after exact conformal calibration with hold-out data, compared to state-of-the-art alternatives.


### Contents
•	experiments_cifar10/ : Code for the experiments with CIFAR-10 dataset.\
•	experiments_credit_synthetic_data/ : Code for the experiments with the credit card data and the synthetic data.

### Prerequisites
Python package dependencies:

•	numpy\
•	torch\
•	tqdm\
•	panda\
•	matplotlib\
•	sys\
•	os\
•	sklearn\
•	torchsort\
•	scipy

The code for the numerical experiments was written to be run on a computing cluster using the SLURM scheduler.
