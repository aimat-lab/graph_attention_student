"""
Generates the export fixtures that the JavaScript test suite in ``js/`` checks against.

The reference featurizer in ``js/featurizer.js`` has to reproduce the Python
featurization column for column - a consumer that gets this wrong feeds the model
plausible but incorrect features and gets plausible but incorrect predictions. The only
way to know it is right is to compare against output that Python actually produced,
which is what this script writes.

Run it before ``npm test``::

    python scripts/generate_js_fixtures.py

The fixtures are build products and are not checked in; ``js/test/featurizer.test.js``
skips itself when they are absent.
"""
import os
import sys

import torch

from graph_attention_student.torch.megan import Megan
from graph_attention_student.torch.export import export_megan_onnx
from graph_attention_student.torch.export.portable_processing import (
    PortableMoleculeProcessing)

#: Chosen to cover the cases where a JavaScript port is most likely to diverge:
#: aromatic rings (RDKit.js reports kekulized bond orders), fused and hetero aromatics,
#: a formal charge, a halogen, and an element outside the default vocabulary.
SMILES = [
    'CCO',
    'c1ccccc1O',
    'CC(=O)Oc1ccccc1C(=O)O',
    'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',
    'FC(F)(F)c1ccccc1',
    'CC(=O)[O-]',
    'c1ccsc1',
    'C1CC1N',
    'CC[NH3+]',
    'BrCCBr',
    'CC(=O)N(C)C',
    'O=S(=O)(O)c1ccccc1',
    'CP(=O)(O)O',
    'C/C=C/C',
    'C/C=C\\\\C',
    'O=[N+]([O-])c1ccccc1',
    'c1ccc2c(c1)[nH]c1ccccc12',
    'COc1ccc2NC(=S)Nc2c1',
    'N#Cc1ccccc1',
    'c1cccn1C',
]

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'js', 'test', 'fixtures')


def main() -> int:
    # The portable processing, not the production one: production includes five per-atom
    # descriptors RDKit.js cannot compute at all, so a browser targets the 38-column
    # portable variant.
    processing = PortableMoleculeProcessing()
    graphs = [processing.process(value) for value in SMILES]

    # A small, deterministically initialized model: the JavaScript side is being checked
    # for wiring and featurization, not for predictive quality, so training would only
    # make the fixtures slower to produce and no more informative.
    torch.manual_seed(3)
    model = Megan(
        node_dim=processing.get_num_node_attributes(),
        edge_dim=processing.get_num_edge_attributes(),
        units=[16, 16],
        num_channels=2,
        projection_units=[16],
        final_units=[8, 1],
    )
    model.eval()

    paths = export_megan_onnx(model, OUTPUT_PATH, graphs, processing=processing)

    print(f'Wrote fixtures for {len(SMILES)} molecules to {OUTPUT_PATH}')
    for role, path in paths.items():
        print(f'  {role:<14} {os.path.getsize(path) / 1024:7.0f} KB')

    return 0


if __name__ == '__main__':
    sys.exit(main())
