# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Setup and Installation
```bash
# Install torch first due to dependency requirements
uv pip install torch==2.1.2

# Install package in development mode
uv pip install -e .

# Alternative: Install from PyPI
uv pip install graph_attention_student
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_util.py

# Using nox (recommended)
nox -s test
```

### Building
```bash
# Build package using nox
nox -s build

# Build with poetry directly
uv build
```

### Running Experiments
```bash
# Use the CLI to run experiments
python -m graph_attention_student.cli run <experiment_name>

# Run specific examples
python -m graph_attention_student.cli example -n quickstart
```

## High-Level Architecture

### Core Components

**MEGAN Model (`graph_attention_student.torch.megan`)**
- Multi-Explanation Graph Attention Network implementation
- Provides self-explaining graph neural networks with multiple explanation channels
- Main model class: `Megan` - extends PyTorch Lightning module
- Supports both regression and classification tasks

**Experiment Framework (`graph_attention_student.experiments/`)**
- Built on PyComex microframework for reproducible experiments
- Base experiment: `vgd_torch__megan.py` - core MEGAN training pipeline
- Dataset-specific experiments inherit from base (e.g., `vgd_torch__megan__aqsoldb.py`)
- Experiments are parameterized through global variables at the top of each file

**Visual Graph Datasets Integration**
- Uses VGD format for graph datasets (JSON + PNG per element)
- Located in `graph_attention_student.torch.data`
- Supports various molecular and graph datasets (AqSolDB, Mutagenicity, etc.)

**PyTorch Implementation (`graph_attention_student.torch/`)**
- `layers.py` - Graph attention layers (GAT variants)
- `model.py` - Base model abstractions and mixins
- `megan.py` - MEGAN model implementation with explanation capabilities
- `data.py` - Data loading and preprocessing utilities

### Key Patterns

**Experiment Inheritance**
- Create sub-experiments by extending base experiments
- Override parameters at module level (e.g., `VISUAL_GRAPH_DATASET = 'aqsoldb'`)
- Use `Experiment.extend()` pattern for code reuse

**Model Configuration**
- `UNITS` - Layer structure for graph encoder
- `NUM_CHANNELS` - Number of explanation channels (typically 2 for regression)
- `IMPORTANCE_FACTOR` - Weight for explanation training loss
- `FINAL_UNITS` - Final MLP structure (last value = number of targets)

**Dataset Types**
- Regression: Use `DATASET_TYPE = 'regression'` and `FINAL_ACTIVATION = 'linear'`
- Classification: Use `DATASET_TYPE = 'classification'` and `FINAL_ACTIVATION = 'softmax'`

**Testing Structure**
- Tests in `tests/` directory follow pytest conventions
- Test utilities in `tests/util.py`
- Visual outputs saved to `ARTIFACTS_PATH` for validation

### CLI Usage

The package provides a CLI through `graph_attention_student.cli`:
- Main command: `megan` (defined via ExperimentCLI)
- Examples: Use `example` subcommand to run predefined examples
- Version info available via `--version`