import os
import sys
import click

from pycomex.experiment import run_experiment

from graph_attention_student.util import get_version
from graph_attention_student.util import DATASETS_FOLDER
from graph_attention_student.util import EXAMPLES_FOLDER

EXAMPLES = [name.split('.')[0] for name in os.listdir(EXAMPLES_FOLDER)]


@click.group(invoke_without_command=True)
@click.option("-v", "--version", is_flag=True)
@click.pass_context
def cli(ctx: click.Context, version: bool):
    """
    Console scripts for graph_attention_student
    """
    if version:
        version = get_version()
        click.secho(version)
        sys.exit(0)


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


@click.command(short_help='print a list of all dataset folders')
@click.pass_context
def list_datasets(ctx):
    names = os.listdir(DATASETS_FOLDER)
    click.secho('datasets found in the dataset folder:')
    for name in names:
        click.secho(f'- {name}')


cli.add_command(example)
cli.add_command(list_datasets)

if __name__ == '__main__':
    cli()
