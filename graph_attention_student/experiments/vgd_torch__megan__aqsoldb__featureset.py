"""
MEGAN on AqSolDB with the 1.6.0 redesigned explanation loss, parameterized by which node
feature columns the model is allowed to see.

This exists to answer one question: how much does the model lose if the featurizer is
restricted to what can be reproduced in a browser with RDKit.js?

RDKit.js exposes no per-atom API at all. Its ``get_json()`` gives atomic number, implicit
hydrogen count, formal charge, isotope, radical count and an aromaticity/ring extension
block, and its ``get_descriptors()`` returns 43 *molecule-level* scalars. The per-atom
decompositions - Crippen, TPSA, Labute ASA, Gasteiger, EState - are simply not reachable.
Hybridization is a separate case: it is not in the JSON either, and RDKit computes it with
a perception algorithm that accounts for conjugation (an amide nitrogen comes out SP2, not
SP3), so deriving it in JavaScript from degree and bond orders would be wrong.

The AqSolDB visual graph dataset already carries all 45 node features precomputed, so the
arms here differ *only* in which columns are sliced out. Same molecules, same splits, same
everything else - which is the cleanest way to isolate the effect of the features.

Column layout of the precomputed node attributes:

    [ 0:17]  symbol            one-hot over 16 elements + unknown
    [17:23]  hybridization     one-hot, PERCEPTION-DEPENDENT
    [23:29]  total_degree      one-hot
    [29:34]  num_hydrogens     one-hot
    [34:35]  mass
    [35:36]  formal charge
    [36:37]  is_aromatic
    [37:38]  is_in_ring
    [38:40]  crippen           NOT REACHABLE from RDKit.js
    [40:41]  tpsa              NOT REACHABLE
    [41:42]  lasa              NOT REACHABLE
    [42:44]  gasteiger         NOT REACHABLE
    [44:45]  estate            NOT REACHABLE

The explanation settings are ported from vgd_torch__megan__rb_dual_motifs_v2__redesign.py.
Note that DIFFERENCE_MARGIN_SCALE does NOT transfer between datasets and is calibrated
separately for AqSolDB - see the parameter comment below.
"""
import os
import random
import typing as t

import numpy as np
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == FEATURE SET ==

# :param NODE_FEATURE_COLUMNS:
#       The column indices of the precomputed node attributes that the model is allowed to
#       see. None means all 45 of them. The slicing happens in the "after_dataset" hook, so
#       every arm trains on identical molecules and identical splits and differs only here.
NODE_FEATURE_COLUMNS: t.Optional[t.List[int]] = None
# :param FEATURE_SET_NAME:
#       A human readable label for the arm, recorded in the experiment data so the results
#       can be collated afterwards without having to infer which arm produced them.
FEATURE_SET_NAME: str = 'full'
# :param KEKULIZE:
#       When True the molecules are re-featurized from their SMILES after kekulization with
#       the aromatic flags cleared, so aromatic rings arrive as alternating single/double
#       bonds and nothing is marked aromatic. That is exactly what RDKit.js reports in its
#       CommonChem output if the aromaticity extension block is ignored.
#
#       What this removes is only the explicit aromatic designation: the node and edge
#       is_aromatic columns go to all-zero and the bond-type one-hot stops using its
#       aromatic slot. Hybridization and is_conjugated were measured to be bit-identical
#       either way, so the pi system is still represented - the question is whether naming
#       it "aromatic" adds anything on top.
#
#       The columns are deliberately kept rather than dropped, so the architecture is
#       identical across arms and the comparison isolates information rather than width.
KEKULIZE: bool = False

# == EXPLANATION OBJECTIVE (1.6.0 redesign) ==
# Ported from vgd_torch__megan__rb_dual_motifs_v2__redesign.py. The stock aqsoldb
# experiment predates 1.6.0 entirely and still runs the old pooled_bce objective.

# :param EXPLANATION_OBJECTIVE:
EXPLANATION_OBJECTIVE: str = 'difference_margin'
# :param DIFFERENCE_MARGIN_SCALE:
#       CALIBRATED PER DATASET - this constant does not transfer. The reference value of 4.0
#       was tuned on rb_dual_motifs_v2, where it was satisfied on 6% of graphs; the same
#       value is satisfied on 83% of aqsoldb graphs, which leaves the term nearly inert.
#       The logged 'margin_met' metric reports the satisfied fraction; the target is roughly
#       half. This value is set from the calibration pilot.
#       Overridable from the environment so the calibration pilot can sweep it without
#       needing a file per value.
DIFFERENCE_MARGIN_SCALE: float = float(os.environ.get('MEGAN_MARGIN_SCALE', '4.0'))
# :param REGRESSION_MARGIN:
#       Must be positive to take effect at all - the implementation guards the margin block
#       with 'if regression_margin > 0', so the negative default silently disables it.
REGRESSION_MARGIN: t.Optional[float] = 0.1

# == RDT FIDELITY ==

# :param RDT_FACTOR:
RDT_FACTOR: float = 1.0
# :param RDT_MASK_ONLY:
RDT_MASK_ONLY: bool = True
# :param RDT_USE_RAW:
RDT_USE_RAW: bool = True
# :param RDT_WARMUP:
#       Zero on purpose: ramping the term in lets the explanation collapse before it can
#       defend the nodes it is meant to protect.
RDT_WARMUP: float = 0.0
# :param RDT_RAMP:
RDT_RAMP: float = 0.0
# :param RDT_SAMPLES:
RDT_SAMPLES: int = 1

# == MASK SHAPE ==

# :param POLAR_FACTOR:
POLAR_FACTOR: float = 0.3
# :param SPREAD_FACTOR:
SPREAD_FACTOR: float = 0.5
# :param SPARSITY_FACTOR:
SPARSITY_FACTOR: float = 1.0

# == EXPERIMENT ==

# :param SPLIT_SEED:
#       Seeds the train/val/test split and the model initialization. The base experiment
#       draws a fresh random split for every run, which would mean the arms never see the
#       same molecules and any difference between them would be part split luck. Pinning it
#       makes the same seed produce the same split in every arm, so the arms are directly
#       comparable and the seed becomes a genuine replicate.
#
#       Read from the environment so that one arm file can be launched several times as
#       independent parallel processes rather than needing one file per (arm, seed) pair.
#       The value actually used is recorded in experiment_data.json either way.
SPLIT_SEED: int = int(os.environ.get('MEGAN_SPLIT_SEED', '0'))
# :param REPETITIONS:
#       One per process: replicates are run as separate processes so they can go in
#       parallel, each with its own SPLIT_SEED.
REPETITIONS: int = 1

# :param EPOCHS:
#       Overridable so the calibration pilot can run short without touching the parent.
EPOCHS: int = int(os.environ.get('MEGAN_EPOCHS', '150'))

# Not debug: debug mode writes every run into a single fixed "debug" folder, so two runs of
# the same arm in parallel would overwrite one another's results.
__DEBUG__ = False
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_torch__megan__aqsoldb.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


def _kekulize_dataset(e: Experiment, index_data_map: dict) -> None:
    """
    Re-featurizes every molecule from its SMILES with the aromatic system kekulized.

    Uses the dataset's own processing module, so the only difference from the stored
    features is the kekulization itself - re-featurizing without it was verified to
    reproduce the stored arrays exactly, which is what makes this comparison clean.
    """
    import importlib.util
    from rdkit import Chem

    module_path = os.path.join(e['dataset_path'], 'process.py')
    spec = importlib.util.spec_from_file_location('vgd_processing', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processing = module.processing

    converted, failed = 0, 0
    for data in index_data_map.values():
        metadata = data['metadata']
        smiles = metadata.get('smiles') or metadata.get('repr')
        try:
            mol = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            graph = processing.process(mol)
        except Exception:
            # A molecule that will not kekulize keeps its original featurization; the
            # count is logged so a silent partial conversion is impossible.
            failed += 1
            continue

        original = metadata['graph']
        # Only the attribute arrays change - connectivity, positions and targets are left
        # exactly as they were so nothing else about the element shifts.
        original['node_attributes'] = np.array(graph['node_attributes'])
        original['edge_attributes'] = np.array(graph['edge_attributes'])
        converted += 1

    e.log(f'kekulized {converted} molecules ({failed} could not be kekulized and kept '
          f'their original aromatic featurization)')
    e['kekulize/converted'] = converted
    e['kekulize/failed'] = failed


@experiment.hook('train_test_split', default=False, replace=True)
def train_test_split(e: Experiment, indices: list, index_data_map: dict, **kwargs):
    """
    Draws a deterministic train/val/test split from SPLIT_SEED.

    The same seed yields the same split in every arm, so a difference between arms cannot
    be an artifact of one of them having drawn an easier test set.
    """
    rng = random.Random(e.SPLIT_SEED)
    shuffled = list(indices)
    rng.shuffle(shuffled)

    test_indices = shuffled[:e.NUM_TEST]
    remaining = shuffled[e.NUM_TEST:]
    # A validation set of the same size as the test set: model selection has to happen on
    # something other than the test set for the reported number to mean anything.
    val_indices = remaining[:e.NUM_TEST]
    train_indices = remaining[e.NUM_TEST:]

    e.log(f'split seed {e.SPLIT_SEED}'
          f' - {len(train_indices)} train / {len(val_indices)} val / {len(test_indices)} test')
    return train_indices, val_indices, test_indices


@experiment.hook('after_dataset', default=False, replace=True)
def after_dataset(e: Experiment, index_data_map: dict, **kwargs) -> None:
    """
    Restricts the node attributes to NODE_FEATURE_COLUMNS, in place on the loaded dataset.

    Doing it here rather than by re-featurizing the molecules means the arms cannot differ
    in any way other than the columns: same RDKit parse, same graph, same split, same
    target. Anything that changed between arms would otherwise be a confound.
    """
    # Seed everything here too, so that weight initialization and any other sampling is
    # also pinned to SPLIT_SEED rather than left to chance.
    import torch
    import pytorch_lightning as pl
    pl.seed_everything(e.SPLIT_SEED, workers=True)
    torch.manual_seed(e.SPLIT_SEED)

    if e.KEKULIZE:
        _kekulize_dataset(e, index_data_map)

    graphs = [data['metadata']['graph'] for data in index_data_map.values()]

    values = [graph['graph_labels'][0] for graph in graphs]
    e.log(f'dataset - {len(graphs)} elements'
          f' - target mean: {np.mean(values):.3f}'
          f' - min: {np.min(values):.3f}'
          f' - max: {np.max(values):.3f}')

    if e.NODE_FEATURE_COLUMNS is None:
        num_features = np.array(graphs[0]['node_attributes']).shape[1]
        e.log(f'feature set "{e.FEATURE_SET_NAME}" - using all {num_features} node features')
        e['feature_set/name'] = e.FEATURE_SET_NAME
        e['feature_set/num_features'] = int(num_features)
        return

    columns = np.array(e.NODE_FEATURE_COLUMNS, dtype=int)
    for graph in graphs:
        graph['node_attributes'] = np.array(graph['node_attributes'])[:, columns]

    # The base experiment derives node_dim from the dataset *before* this hook runs, so it
    # still holds 45 and the first layer would be built for the wrong input width.
    e['node_dim'] = int(len(columns))

    e.log(f'feature set "{e.FEATURE_SET_NAME}" - restricted node features to '
          f'{len(columns)} of 45 columns (node_dim now {len(columns)})')
    e['feature_set/name'] = e.FEATURE_SET_NAME
    e['feature_set/num_features'] = int(len(columns))
    e['feature_set/columns'] = [int(c) for c in columns]


@experiment.testing
def testing(e: Experiment):
    e.log('TESTING MODE')
    e.NUM_EXAMPLES = 5
    e.EPOCHS = 3


experiment.run_if_main()
