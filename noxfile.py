import os
import shutil
from pathlib import Path

import nox

# Configure nox to use uv by default
nox.options.default_venv_backend = "uv"

# Python versions to test against
PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]
LATEST_PYTHON = "3.12"



def get_wheel_path(path: str = 'dist') -> str:
    """Find the wheel file in the dist directory."""
    for name in os.listdir(path):
        wheel_path = os.path.join(path, name)
        if wheel_path.endswith('.whl'):
            return wheel_path
    raise FileNotFoundError("No wheel file found in dist directory")


@nox.session(python=PYTHON_VERSIONS)
def test(session: nox.Session) -> None:
    """Run tests with pytest and coverage."""
    session.install("pytest", "pytest-cov", "pytest-xdist")
    session.install("-e", ".")
    session.run(
        "pytest",
        *session.posargs
    )


@nox.session(python=LATEST_PYTHON)
def lint(session: nox.Session) -> None:
    """Run linting with ruff."""
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python=LATEST_PYTHON)
def format(session: nox.Session) -> None:
    """Format code with ruff."""
    session.install("ruff")
    session.run("ruff", "format", ".")
    session.run("ruff", "check", "--fix", ".")


@nox.session(python=LATEST_PYTHON)
def typecheck(session: nox.Session) -> None:
    """Run type checking with mypy."""
    session.install("mypy", "types-setuptools")
    session.install("-e", ".")
    session.run("mypy", "graph_attention_student")


@nox.session(python=LATEST_PYTHON)
def security(session: nox.Session) -> None:
    """Run security scanning with bandit and safety."""
    session.install("bandit[toml]", "safety")
    session.run("bandit", "-r", "graph_attention_student", "-f", "json", "-o", "bandit-report.json")
    session.run("bandit", "-r", "graph_attention_student")
    session.run("safety", "check", "--json", "--output", "safety-report.json")
    session.run("safety", "check")


@nox.session(python=LATEST_PYTHON)
def docs(session: nox.Session) -> None:
    """Build documentation with sphinx."""
    session.install("sphinx", "sphinx-rtd-theme", "sphinx-autodoc-typehints")
    session.install("-e", ".")

    docs_dir = Path("docs")
    if not docs_dir.exists():
        session.log("No docs directory found, skipping documentation build")
        return

    session.run("sphinx-build", "-b", "html", "docs", "docs/_build/html")


@nox.session(python=LATEST_PYTHON)
def build(session: nox.Session) -> None:
    """Build the package."""
    if os.path.exists('dist'):
        shutil.rmtree('dist')
        session.log('purged dist folder')

    session.install("build")
    session.run("python", "-m", "build")

    wheel_path = get_wheel_path()
    session.install(wheel_path)
    session.run("python", "-m", "pip", "show", "graph_attention_student")
    session.run("python", "-m", "graph_attention_student.cli", "--version")


@nox.session(python=LATEST_PYTHON)
def dev(session: nox.Session) -> None:
    """Set up development environment."""
    session.install("-e", ".")
    session.install("pytest", "pytest-cov", "ruff", "mypy", "pre-commit")
    session.run("pre-commit", "install")


@nox.session(python=LATEST_PYTHON)
def clean(session: nox.Session) -> None:
    """Clean up build artifacts and cache files."""
    patterns = [
        "build/",
        "dist/",
        "*.egg-info/",
        ".pytest_cache/",
        ".coverage",
        "htmlcov/",
        ".mypy_cache/",
        ".ruff_cache/",
        "**/__pycache__/",
        "**/*.pyc",
        "**/*.pyo",
        "bandit-report.json",
        "safety-report.json",
    ]

    for pattern in patterns:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                session.log(f"Removed directory: {path}")
            elif path.is_file():
                path.unlink()
                session.log(f"Removed file: {path}")
