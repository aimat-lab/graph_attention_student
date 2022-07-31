import sys
import click

from graph_attention_student.util import get_version


@click.group(invoke_without_command=True)
@click.option("-v", "--version", is_flag=True)
@click.pass_context
def cli(ctx: click.Context, version: bool):
    """Console script for pycomex."""
    if version:
        version = get_version()
        click.secho(version)
        sys.exit(0)
