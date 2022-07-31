"""
Utility methods
"""
import os
import pathlib

PATH = pathlib.Path(__file__).parent.absolute()
VERSION_PATH = os.path.join(PATH, 'VERSION')

DATASETS_FOLDER = os.path.join(PATH, 'datasets')
EXAMPLES_FOLDER = os.path.join(PATH, 'examples')


def get_version():
    with open(VERSION_PATH) as file:
        return file.read().replace(' ', '').replace('\n', '')
