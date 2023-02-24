"""
Inherits ""

**CHANGLELOG**

0.1.0 - xx.xx.2023 - Initial version
"""
import os
import pathlib
import typing as t

from pycomex.util import Skippable
from pycomex.experiment import SubExperiment


# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, '')
BASE_PATH = PATH
NAMESPACE = 'results/'
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):
    pass