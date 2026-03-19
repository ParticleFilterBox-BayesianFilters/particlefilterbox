"""Main CLI application for particlefilterbox.

Entry point for the `pfbox` command-line tool. Uses Typer for
argument parsing, help generation, and command routing.

References
----------
Typer documentation: https://typer.tiangolo.com/
"""

from __future__ import annotations

import typer

from particlefilterbox.cli.compare_cmd import compare_command
from particlefilterbox.cli.estimate_cmd import estimate_command
from particlefilterbox.cli.filter_cmd import filter_command
from particlefilterbox.cli.simulate_cmd import simulate_command

app = typer.Typer(
    name="pfbox",
    help="particlefilterbox - Particle filtering and Sequential Monte Carlo CLI",
    add_completion=False,
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo("particlefilterbox 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """particlefilterbox CLI - Particle filtering and SMC methods."""


app.command(name="filter")(filter_command)
app.command(name="estimate")(estimate_command)
app.command(name="compare")(compare_command)
app.command(name="simulate")(simulate_command)


if __name__ == "__main__":
    app()
