"""
This module illustrates an example of how to train a MEGAN model for a custom dataset.
In principle it is possible to use the MEGAN

**WHERE IS ALL THE CODE?**

"""
import os
import pathlib

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from graph_attention_student.utils import EXPERIMENTS_PATH


experiment = Experiment.extend(
    os.path.join(EXPERIMENTS_PATH, 'vgd_torch__megan.py'),
)

experiment.run_if_main()
