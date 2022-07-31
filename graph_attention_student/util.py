"""
Utility methods
"""
import os
import pathlib

PATH = pathlib.Path(__file__).parent.absolute()
VERSION_PATH = os.path.join(PATH, 'VERSION')


def get_version():
    with open(VERSION_PATH) as file:
        return file.read().replace(' ', '').replace('\n', '')
