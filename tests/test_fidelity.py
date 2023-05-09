from graph_attention_student.testing import get_mock_graphs
from graph_attention_student.models.megan import MockMegan

from graph_attention_student.fidelity import leave_one_out_analysis


def test_leave_one_out_analysis_basically_works():
    """

    """
    num_channels = 2
    num_targets = 3
    num_graphs = 20
    model = MockMegan(
        importance_channels=num_channels,
        final_units=[num_targets]
    )
    graphs = get_mock_graphs(num_graphs)

    results = leave_one_out_analysis(
        model=model,
        graphs=graphs,
        num_targets=num_targets,
        num_channels=num_channels
    )

    assert len(results) == num_graphs
    assert len(results[0]) == num_targets
    assert len(results[0][0]) == num_channels
    assert isinstance(results[0][0][0], float)

