"""
Derivation of a machine-readable featurization spec from a processing object.

The exported ONNX model consumes graph tensors, so a non-Python consumer has to
build ``x`` and ``edge_attr`` itself, column for column, in exactly the order the
model was trained on. That order lives in the ``node_attribute_map`` /
``edge_attribute_map`` of a :class:`MoleculeProcessing` instance and is *not*
stored in the checkpoint, so it is derived here from a live processing object and
written out beside the model.

Not every encoder can be reproduced outside Python. Anything reading Gasteiger
charges, Crippen or TPSA contributions or EState indices needs RDKit internals that
RDKit.js does not expose. Such a spec is not emitted at all rather than emitted with
columns that would silently disagree - :func:`derive_featurization_spec` reports it
as unportable, with the offending encoder named, and the model and manifest still
export normally.
"""
import typing as t

from graph_attention_student.utils import get_version

__all__ = [
    'PORTABLE_ATOM_PROPERTIES',
    'PORTABLE_BOND_PROPERTIES',
    'FeaturizationSpec',
    'derive_featurization_spec',
]

#: RDKit atom properties with a verified RDKit.js equivalent in ``featurizer.js``.
#: Every entry here is covered by the Python-vs-JS fixture test; nothing is listed
#: on the assumption that it "should work".
PORTABLE_ATOM_PROPERTIES: t.Set[str] = {
    'GetSymbol',
    'GetFormalCharge',
    'GetIsAromatic',
    'GetDegree',
    'GetTotalDegree',
    'GetTotalNumHs',
    'GetMass',
    'IsInRing',
    # Not exposed by RDKit.js and not naively derivable - RDKit's perception accounts for
    # conjugation, so an amide nitrogen comes out SP2 rather than SP3. featurizer.js
    # reproduces it for 99.93% of atoms; the residue is bracketed silicon in siloxanes,
    # organometallics and nitroxide radicals.
    'GetHybridization',
}

#: RDKit bond properties with a verified RDKit.js equivalent in ``featurizer.js``.
PORTABLE_BOND_PROPERTIES: t.Set[str] = {
    'GetBondType',
    'GetIsAromatic',
    'IsInRing',
    'GetStereo',
    # A transcription of MolOps::setConjugation rather than an approximation of it -
    # verified against RDKit over ~429,000 bonds with zero disagreements. Inferring the
    # rule from its behaviour only ever reached 79-96%, because the real algorithm marks
    # bonds pairwise and depends on RDKit's aromatic valence clamp.
    'GetIsConjugated',
}


class FeaturizationSpec(t.NamedTuple):
    """
    The result of inspecting a processing object.

    :param portable: whether every encoder has a verified JavaScript equivalent. When
        False, ``spec`` is still populated for documentation but must not be written
        out as a contract for a JS featurizer.
    :param reason: None when portable, else a human-readable explanation naming the
        encoder that cannot be reproduced.
    :param spec: the JSON-serializable spec.
    """
    portable: bool
    reason: t.Optional[str]
    spec: dict


def _property_name(callback: t.Callable) -> t.Optional[str]:
    """
    Recovers the RDKit method name a ``chem_prop`` callback reads.

    ``chem_prop`` captures the name in a closure and exposes only the encoder as
    ``.callback``, so the name is recovered from the closure cells - it is the only
    string among them.
    """
    cells = [c.cell_contents for c in (callback.__closure__ or ())]
    names = [c for c in cells if isinstance(c, str)]
    return names[0] if len(names) == 1 else None


def _measure_width(processing, callback, mock) -> int:
    """
    :returns: how many columns ``callback`` contributes to the feature vector.

    Goes through the processing object's own ``apply_callback``, which knows about the
    two different callback signatures in use: most take just the atom or bond, but ones
    flagged ``requires_molecule`` - the Crippen, TPSA, Labute ASA, Gasteiger, EState and
    random-walk encoders - take ``(mol, element)`` instead. Calling the callback
    directly works for the first kind and raises a TypeError for the second.
    """
    return len(processing.apply_callback(callback, processing.MOCK_MOLECULE, mock))


def _describe_encoder(processing, name: str, entry: dict, mock, offset: int) -> dict:
    """
    Describes one encoder as a JSON-serializable dict, including the column range it
    occupies in the assembled feature vector.

    :param mock: the mock atom or bond used to measure the encoder's output width,
        the same technique ``MoleculeProcessing.get_description_map`` uses.
    :param offset: the first column index this encoder writes to.
    """
    callback = entry['callback']
    encoder = getattr(callback, 'callback', None)
    width = _measure_width(processing, callback, mock)

    described = {
        'name': name,
        'property': _property_name(callback),
        'slice': None if offset is None else [offset, offset + width],
        'width': width,
        'description': entry.get('description'),
    }

    # A OneHotEncoder exposes its whole vocabulary publicly, which is what a JS port
    # needs; anything else is treated as passing the raw value straight through.
    if hasattr(encoder, 'values'):
        described.update({
            'kind': 'one_hot',
            'values': list(encoder.values),
            'string_values': list(encoder.string_values) if encoder.string_values else None,
            'add_unknown': bool(encoder.add_unknown),
            'unknown': encoder.unknown,
            'dtype': getattr(encoder.dtype, '__name__', str(encoder.dtype)),
        })
    else:
        described['kind'] = 'identity'

    return described


def _describe_map(processing, attribute_map: dict, mock, portable: t.Set[str]
                  ) -> t.Tuple[t.List[dict], t.List[str]]:
    """
    :returns: the encoder descriptions, and the reasons any of them are not portable.
    """
    encoders: t.List[dict] = []
    problems: t.List[str] = []
    offset = 0

    for name, entry in attribute_map.items():
        try:
            described = _describe_encoder(processing, name, entry, mock, offset)
        except Exception as exc:
            # The width is unknown, so every following column index would be wrong.
            # Stop numbering rather than silently mis-numbering the rest: this spec is
            # never written out (an uninspectable encoder always makes it unportable),
            # but it is read by whoever is working out why their export was refused.
            problems.append(f'encoder "{name}" could not be inspected: {exc}')
            encoders.append({'name': name, 'property': None, 'slice': None,
                             'kind': 'unknown',
                             'description': entry.get('description')})
            offset = None
            continue

        offset = None if described['slice'] is None else described['slice'][1]
        encoders.append(described)

        if described['property'] is None:
            problems.append(
                f'encoder "{name}" does not read a plain RDKit property (it is probably a '
                f'descriptor or a computed quantity), so it has no JavaScript equivalent'
            )
        elif described['property'] not in portable:
            problems.append(
                f'encoder "{name}" reads "{described["property"]}", which has no verified '
                f'RDKit.js equivalent in featurizer.js'
            )

    return encoders, problems


def derive_featurization_spec(processing) -> FeaturizationSpec:
    """
    Inspects ``processing`` and derives the featurization contract from it.

    :param processing: a :class:`MoleculeProcessing` instance - the *same one* used to
        build the training data, since the attribute maps are what define the column
        order and are not recorded in the model checkpoint.
    """
    node_encoders, node_problems = _describe_map(
        processing, processing.node_attribute_map, processing.MOCK_ATOM,
        PORTABLE_ATOM_PROPERTIES)
    edge_encoders, edge_problems = _describe_map(
        processing, processing.edge_attribute_map, processing.MOCK_BOND,
        PORTABLE_BOND_PROPERTIES)

    try:
        from importlib.metadata import version as _pkg_version
        # NOT visual_graph_datasets.__version__, which is stale (it reports 0.15.7 for
        # an 0.17.1 install) and would put a wrong contract version in the manifest.
        vgd_version = _pkg_version('visual_graph_datasets')
    except Exception:
        vgd_version = None

    def _size(getter) -> t.Optional[int]:
        # These call every encoder, so a broken one throws here rather than in the
        # inspection above; the spec is unusable either way, but it must be reported
        # rather than raised.
        try:
            return getter()
        except Exception as exc:
            problems.append(f'the attribute size could not be determined: {exc}')
            return None

    problems = node_problems + edge_problems
    node_size = _size(processing.get_num_node_attributes)
    edge_size = _size(processing.get_num_edge_attributes)

    spec = {
        'specVersion': 1,
        'generatedBy': f'graph_attention_student {get_version()}',
        'processingClass': type(processing).__name__,
        'visualGraphDatasetsVersion': vgd_version,
        # process() appends both directions of every bond, back to back, and node order
        # is the RDKit atom index order. A JS port must reproduce both.
        'doubleEdgesUndirected': True,
        'nodeOrder': 'rdkit_atom_index',
        'edgeOrder': 'per_bond_forward_then_reverse',
        'nodeAttributes': {
            'size': node_size,
            'encoders': node_encoders,
        },
        'edgeAttributes': {
            'size': edge_size,
            'encoders': edge_encoders,
        },
    }

    return FeaturizationSpec(
        portable=not problems,
        reason=None if not problems else '; '.join(problems),
        spec=spec,
    )
