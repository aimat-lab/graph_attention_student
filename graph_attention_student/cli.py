import os
import sys
import click

from pycomex.experiment import run_experiment
from pycomex.cli import ExperimentCLI

from graph_attention_student.util import get_version
from graph_attention_student.util import DATASETS_FOLDER
from graph_attention_student.util import EXAMPLES_FOLDER
from graph_attention_student.util import EXPERIMENTS_PATH

EXAMPLES = [name.split('.')[0] for name in os.listdir(EXAMPLES_FOLDER)]


cli = ExperimentCLI(name='megan', experiments_path=EXPERIMENTS_PATH, version=get_version())


@click.command(short_help='executes the example code')
@click.option('-n', '--name', type=click.Choice(EXAMPLES), default='solubility_regression',
              help='the name of the example file to be executed')
@click.pass_context
def example(ctx, name: str):
    """
    Executes the example code
    """
    click.secho(f'Running example code: {name}!')
    example_path = os.path.join(EXAMPLES_FOLDER, name + '.py')
    experiment_path, proc = run_experiment(example_path)

    click.secho(f'Experiment completed and saved to: {experiment_path}')


cli.add_command(example)

if __name__ == '__main__':
    cli()
