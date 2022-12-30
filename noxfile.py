import os
import shutil

import nox


def get_wheel_path(path: str = 'dist') -> str:
    for name in os.listdir(path):
        path = os.path.join(path, name)
        if path.endswith('.whl'):
            return path


@nox.session
def test(session: nox.Session) -> None:
    session.run('poetry', 'install')
    session.install('pytest')
    session.run('pytest')


@nox.session
def build(session: nox.Session) -> None:
    if os.path.exists('dist'):
        shutil.rmtree('dist')
        session.log('purged dist folder')

    session.run('poetry', 'build')

    wheel_path = get_wheel_path()
    session.install(wheel_path)
    session.run('python', '-m', 'pip', 'show', 'graph_attention_student')
    session.run('python', '-m', 'graph_attention_student.cli', '--version')
