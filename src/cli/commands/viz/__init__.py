import matplotlib

matplotlib.use("Agg")  # the CLI writes files; a notebook importing src.viz keeps its own backend

import typer

from ...common import CONTEXT_SETTINGS, HelpfulCommand, HelpfulGroup
from . import activations, circuits, comparison, dataset, model, probing, runs, steering
from . import dashboard as dashboard_command

"""
Render the charts in src/viz/ and write them to disk. Each command mirrors one
viz function: the command loads what that function needs and saves the figure
it returns, and does no plotting of its own.

One module per group, and the groups follow the shape of the questions rather
than the shape of the code -- what is in the data, what is in the model, what
is in the activations, what the probe found, what steering did, which heads do
the task, what the runs say. That is the same split src/viz/ uses, so a chart
and the command that draws it sit at the same path in two trees.

The matplotlib backend is selected here, at the top, before any group module
loads -- and therefore before any src.viz module does. Selecting a backend is
process-wide, so it belongs to the CLI and nowhere else: src/viz/ must never
call use(), or it takes the backend away from a notebook that imported it.

Every command takes --output and --show, and --show renders the chart inline
in the terminal for anyone working over ssh.

Run with: python -m src.cli viz <group> <command> [options]
"""

app = typer.Typer(
    help="Draw the dataset, the model, the activations, the probes and the runs.",
    cls=HelpfulGroup,
    context_settings=CONTEXT_SETTINGS,
)

app.add_typer(dataset.app, name="dataset")
app.add_typer(model.app, name="model")
app.add_typer(activations.app, name="act")
app.add_typer(probing.app, name="probe")
app.add_typer(steering.app, name="steer")
app.add_typer(circuits.app, name="circuit")
app.add_typer(comparison.app, name="compare")
app.add_typer(runs.app, name="runs")

# Registered rather than decorated: the dashboard spans every group, so it imports
# from them, and it cannot also import the app it hangs off without a cycle.
app.command("dashboard", cls=HelpfulCommand)(dashboard_command.dashboard)
