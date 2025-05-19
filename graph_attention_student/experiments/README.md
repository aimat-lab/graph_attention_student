# Computational Experiments

This directory contains the modules defining the computational experiments for the project. Each module is a self-contained experiment based on the ``pycomex`` library. These experiments can be initiated by executing the corresponding module. Upon execution, the experiments will create a new archive folder within the ``results`` directory containing all the artifacts, logs, metrics etc. of the experiment run.

- ``vgd_torch.py`` The base implementation of training a torch graph neural network model on a visual graph dataset (VGD) and subsequently analyszing the models performance as well as it's attributional explanations.

# Experiment Sweeps

In addition to the individual experiment modules, we also define experiment sweeps which systematically investigate different configurations.

- ``ex_01``: An experiment which sweeps over different learning rates and different LR 
  scheduler configurations to analyze the impact on both the explanation approximation accuracy and 
  the primary prediction performance.
    - **a**: variation of learning rates 0.0001 and 0.00001 and either no scheduler or cyclic learning 
      rate scheduler. On the ``rb_dual_motifs`` dataset.
    - **b**: variation of learning rates between 1e-6, 1e-5, 1e-4 and schedulers None, cyclic for the 
      ``mutagenicity`` dataset.

- ``ex_02``: An experiment which sweeps over different batch sizes to analyze the impact on both the 
  explanation approximation accuracy and the primary prediction performance.
    - **a**: variation of batch sized between 16 and 128 for the ``mutagenicity`` dataset.
    - **b**: variation of batch sizes between 16 and 128 for ``aqsodlb`` dataset.