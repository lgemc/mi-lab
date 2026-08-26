import hydra
from omegaconf import DictConfig, OmegaConf

from .experiment.runner import run_directory, run_experiment
from .experiment.spec import _register_schema

"""
The Hydra entry point, and the only place @hydra.main is used.

It exists for one thing the Typer CLI cannot do: sweeps. `--multirun` with a
comma-separated override runs the whole cross product, one Run per
combination, which is how a question like "does the best depth move with model
size" gets answered in a single command.

    python -m src.app --multirun model=gpt2-small,pythia-70m
    python -m src.app --multirun method.lr=0.01,0.05,0.2 seed=0,1,2

It is kept separate from the CLI on purpose: @hydra.main takes over argv, so a
command group cannot live inside it. Everything else -- composing the same
specs, running one of them, reading runs back -- is `python -m src.cli run`.

Returning the headline metric lets a sweeper optimize against it; the basic
sweeper ignores the return value.
"""

_register_schema()

@hydra.main(version_base="1.3", config_path="../specs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Run one composed spec, and report its headline number"""
    spec = OmegaConf.to_object(cfg).validate()
    print(f"running '{spec.experiment}' ({spec.kind}) hash {spec.spec_hash}")
    run = run_experiment(spec)

    headline = run.metrics.get("best_auc", run.metrics.get("auc", float("nan")))
    print(f"{run.status} in {run.duration_seconds:.1f}s -> {run_directory(spec, run)}  AUC {headline:.3f}")
    return headline

if __name__ == "__main__":
    main()
