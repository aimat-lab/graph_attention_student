"""
Extraction of a portable, JSON-serializable description of a model's architecture.

This is deliberately independent of any one export target. The ONNX exporter writes
it into ``manifest.json`` so a consumer can see what it is running without loading
the graph, and the Rust/WebAssembly compilation path described in ``RUST_COMPILE.md``
and ``WASM_COMPILE.md`` needs the same information to generate code. Keeping one
extractor means the two paths cannot disagree about what a model *is*.
"""
import typing as t

import torch.nn as nn

__all__ = ['extract_architecture', 'extract_config']

#: Constructor arguments that determine the shape or the semantics of what ``forward``
#: returns. Read off the live model rather than off ``ckpt['hyper_parameters']``, which
#: is a hand-maintained subset (``save_hyperparameters`` is never called) and silently
#: omits several of these.
CONFIG_ATTRIBUTES: t.Tuple[str, ...] = (
    'node_dim',
    'edge_dim',
    'units',
    'hidden_units',
    'layer_version',
    'projection_units',
    'num_channels',
    'final_units',
    # final_units is a constructor argument that Megan does not keep, so out_dim /
    # target_dim (its last element) are what actually survive onto the instance.
    'out_dim',
    'target_dim',
    'prediction_mode',
    'attention_aggregation',
    'normalize_embedding',
    'output_norm',
    'importance_mode',
    'importance_factor',
    'importance_offset',
    'leave_one_out_fix',
)


def _describe_module(name: str, module: nn.Module) -> t.Optional[dict]:
    """
    :returns: a descriptor for the parameterized leaf modules, or None for modules that
        carry no weights and no configuration worth recording.
    """
    if isinstance(module, nn.Linear):
        return {
            'op': 'linear',
            'name': name,
            'in_dim': module.in_features,
            'out_dim': module.out_features,
            'has_bias': module.bias is not None,
        }
    if isinstance(module, nn.BatchNorm1d):
        return {
            'op': 'batch_norm',
            'name': name,
            'features': module.num_features,
            'eps': module.eps,
        }
    if isinstance(module, nn.LayerNorm):
        return {
            'op': 'layer_norm',
            'name': name,
            'features': list(module.normalized_shape),
            'eps': module.eps,
        }
    if isinstance(module, nn.LeakyReLU):
        return {'op': 'leaky_relu', 'name': name, 'negative_slope': module.negative_slope}
    if isinstance(module, nn.Dropout):
        # Recorded because a non-zero rate is a signal that the model was exported
        # from a train-mode configuration; it is inert under eval().
        return {'op': 'dropout', 'name': name, 'p': module.p}
    if isinstance(module, (nn.SiLU, nn.ReLU, nn.Tanh, nn.Softplus, nn.Identity)):
        return {'op': type(module).__name__.lower(), 'name': name}

    return None


def extract_config(model: nn.Module) -> dict:
    """
    :returns: the model-level configuration, as a JSON-serializable dict.
    """
    config = {}
    for attribute in CONFIG_ATTRIBUTES:
        if not hasattr(model, attribute):
            continue
        value = getattr(model, attribute)
        # Tensors and other non-JSON values are stringified rather than dropped, so the
        # manifest never silently loses a configuration entry.
        if isinstance(value, (str, int, float, bool, type(None))):
            config[attribute] = value
        elif isinstance(value, (list, tuple)):
            config[attribute] = [v if isinstance(v, (str, int, float, bool)) else str(v)
                                 for v in value]
        else:
            config[attribute] = str(value)

    return config


def extract_architecture(model: nn.Module) -> dict:
    """
    Walks the module tree and describes the model layer by layer.

    :returns: ``{'config': {...}, 'layers': [...], 'num_parameters_total': int}``

    Note that ``num_parameters_total`` counts every parameter of the PyTorch module,
    including the contrastive, uncertainty and importance heads that ``forward`` never
    calls. The exported ONNX graph contains only the reachable subset, so it is
    normally a good deal smaller than this figure implies.
    """
    layers = []
    for name, module in model.named_modules():
        described = _describe_module(name, module)
        if described is not None:
            layers.append(described)

    return {
        'config': extract_config(model),
        'layers': layers,
        'num_parameters_total': sum(p.numel() for p in model.parameters()),
    }
