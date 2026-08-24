import typer
from typer._click.exceptions import UsageError
from typer.core import TyperCommand, TyperGroup

"""
Shared Click/Typer customization used by every app and command in the CLI:
- CONTEXT_SETTINGS adds -h as a short alias for --help.
- HelpfulCommand/HelpfulGroup print the full --help output (not just the
  one-line usage) whenever a command is invoked with bad or missing
  arguments, so the user sees the available options right away instead of
  having to re-run with --help.

Typer 0.27 vendors its own click fork under typer._click, so the UsageError
raised while parsing arguments is typer._click.exceptions.UsageError, not
click.UsageError from the top-level click package.
"""

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

class _ShowsHelpOnError:
    """Mixin that prints ctx.get_help() before failing a bad invocation

    Bad options/arguments fail inside parse_args, but an unknown subcommand
    on a group only fails once resolve_command runs inside invoke, so both
    are wrapped.
    """

    def parse_args(self, ctx, args):
        try:
            return super().parse_args(ctx, args)
        except UsageError as error:
            self._show_help_and_exit(ctx, error)

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except UsageError as error:
            self._show_help_and_exit(ctx, error)

    @staticmethod
    def _show_help_and_exit(ctx, error):
        typer.echo(ctx.get_help(), err=True)
        typer.echo(err=True)
        typer.echo(f"Error: {error.format_message()}", err=True)
        ctx.exit(2)

class HelpfulCommand(_ShowsHelpOnError, TyperCommand):
    """A Typer command that shows its own help when it fails to parse"""

class HelpfulGroup(_ShowsHelpOnError, TyperGroup):
    """A Typer group that shows its own help when it fails to parse or dispatch"""
