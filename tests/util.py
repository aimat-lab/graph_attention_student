import os
import sys
import pathlib
import logging

import matplotlib.pyplot as plt

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')
ARTIFACTS_PATH = os.path.join(PATH, 'artifacts')

LOG_TESTING = True
LOG = logging.Logger('Test')
LOG.addHandler(logging.NullHandler())
if LOG_TESTING:
    LOG.addHandler(logging.StreamHandler(sys.stdout))


def save_fig(fig: plt.Figure) -> str:
    path = os.path.join(ARTIFACTS_PATH, 'fig.pdf')
    fig.savefig(path)

    return path
