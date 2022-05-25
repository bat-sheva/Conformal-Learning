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
•	`conformal_learning/` : Python package implementing our methods and some alternative benchmarks.\
•	`examples/` : Jupyter notebooks to carry out a single instance of the numerical experiments and visualize the results.\
•	`experiments/` : Three sub-directories containing the code for conducting the experiments with CIFAR-10, credit card and the synthetic data.\
For credit and synthetic data run 'main' file in order to run all experiments in parallel on a computing cluster and reproduce results from the paper. 'Run_all' and 'submit' files run the main file with different seeds (and also varying parameters for the synthetic data) on the clusters. 'Show_results' notebook visualizes the results achieved in these experiments and create the figures presented in the paper.

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
•	random\
•	seaborn\
•	scipy

The code for the numerical experiments was written to be run on a computing cluster using the SLURM scheduler.
