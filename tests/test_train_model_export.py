"""
The CSV training experiment should produce a browser-ready model by default: trained on
the portable feature set, archived next to the matching process.py, and exported to ONNX.

The experiment runs for real, through "pycomex run" in a subprocess, so that its global
state - the matplotlib backend, the seeds, the hooks registered on import - cannot leak
into the other tests.
"""
import filecmp
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from graph_attention_student.util import EXPERIMENTS_PATH
from graph_attention_student.torch.megan import Megan
from graph_attention_student.torch.export import portable_processing
from graph_attention_student.torch.export.wrapper import (
    MeganOnnxWrapper, INPUT_NAMES, OUTPUT_NAMES)

from .util import ASSETS_PATH

ort = pytest.importorskip('onnxruntime')
pytest.importorskip('onnxscript')


def _run_training(base_path, **parameters) -> str:
    """
    Trains train_model__megan.py for one epoch on the small test CSV.

    :returns: the archive folder of the run.
    """
    config = {
        'extend': os.path.join(EXPERIMENTS_PATH, 'train_model__megan.py'),
        'name': 'test_train_model_export',
        'base_path': str(base_path),
        'namespace': 'run',
        'parameters': {
            'CSV_FILE_PATH': os.path.join(ASSETS_PATH, 'test.csv'),
            'VALUE_COLUMN_NAME': 'smiles',
            'TARGET_COLUMN_NAMES': ['value'],
            'EPOCHS': 1,
            'NUM_EXAMPLES': 2,
            'DO_CLUSTERING': False,
            'TRACK_EMBEDDING_HEALTH': False,
            'TRACK_EMBEDDING_UNIFORMITY': False,
            'TRACK_EMBEDDING_TARGET_ALIGNMENT': False,
            'TRACK_EMBEDDING_PROJECTIONS': False,
            **parameters,
        },
    }
    config_path = os.path.join(base_path, 'config.yml')
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    pycomex = shutil.which('pycomex', path=os.path.dirname(sys.executable)) or 'pycomex'
    # UTF-8 mode on both ends: on Windows a piped child would otherwise write cp1252 and
    # fail on the first character outside it that the experiment logs.
    result = subprocess.run(
        [pycomex, 'run', config_path],
        capture_output=True, encoding='utf-8', errors='replace',
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONUTF8': '1'},
    )
    # The experiment is in debug mode, so its archive is always the "debug" folder.
    archive = os.path.join(base_path, 'run', 'debug')
    assert result.returncode == 0 and os.path.exists(os.path.join(archive, 'model.ckpt')), (
        f'the training run failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}')

    return archive


def test_default_training_run_is_browser_ready(tmp_path):
    """
    A run with default settings should leave everything a browser needs in its
    "onnx_export" folder, and the archived process.py should match the model - the mismatch
    that used to make "megan export" fail on portable models.
    """
    archive = _run_training(tmp_path)

    assert filecmp.cmp(os.path.join(archive, 'process.py'), portable_processing.__file__,
                       shallow=False), 'the archived process.py is not the portable one'

    onnx_path = os.path.join(archive, 'onnx_export')
    for name in ('model.onnx', 'manifest.json', 'featurization.json', 'fixtures.json',
                 'megan.js'):
        assert os.path.exists(os.path.join(onnx_path, name)), f'{name} was not exported'

    manifest = json.load(open(os.path.join(onnx_path, 'manifest.json')))
    processing = portable_processing.processing
    assert manifest['architecture']['config']['node_dim'] == processing.get_num_node_attributes()

    # The export has to be of the model that was saved, not of some intermediate state.
    fixtures = json.load(open(os.path.join(onnx_path, 'fixtures.json')))
    tensors = [
        torch.tensor(fixtures['inputs'][name]['data'],
                     dtype=torch.long if fixtures['inputs'][name]['type'] == 'int64'
                     else torch.float32).reshape(fixtures['inputs'][name]['dims'])
        for name in INPUT_NAMES
    ]
    model = Megan.load(os.path.join(archive, 'model.ckpt')).eval()
    with torch.no_grad():
        reference = MeganOnnxWrapper(model).eval()(*tensors)

    session = ort.InferenceSession(os.path.join(onnx_path, 'model.onnx'),
                                   providers=['CPUExecutionProvider'])
    got = session.run(None, {name: tensor.numpy() for name, tensor in zip(INPUT_NAMES, tensors)})
    for name, expected, actual in zip(OUTPUT_NAMES, reference, got):
        difference = np.abs(expected.numpy() - actual).max()
        assert difference < 1e-4, f'{name} differs from the saved checkpoint by {difference}'
