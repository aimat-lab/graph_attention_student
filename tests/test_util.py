import pytest

from graph_attention_student.util import update_nested_dict


def test_update_nested_dict():
    # The whole point of this function is that it performs a dict update which respects nested dicts.
    # So whenever a nested dict is found which exists in the original and the new dict then instead of
    # replacing the entire dict of the original with the version of the new one (as is the standard
    # behavior of dict.update()) it performs a dict.update() on those two dicts recursively again.
    original = {
        'nesting1': {
            'value_original': 10
        },
        'nesting2': {
            'value_original': 10
        }
    }
    extension = {
        'value': 20,
        'nesting1': {
            'value_extension': 20
        },
        'nesting2': {
            'value_original': 20
        }
    }

    merged = update_nested_dict(original, extension)
    assert isinstance(merged, dict)
    assert 'value' in merged and merged['value'] == 20
    assert len(merged['nesting1']) == 2
    assert len(merged['nesting2']) == 1
    assert merged['nesting2']['value_original'] == 20
