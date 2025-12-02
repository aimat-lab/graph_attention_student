import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path


experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('generate_data')
def generate_data(e: Experiment) -> Tuple[np.ndarray, np.ndarray]:
    pass


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
    
experiment.run_if_main()