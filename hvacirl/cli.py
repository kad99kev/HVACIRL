import click

from hvacirl.config import run_gym_options, run_mode_options
from hvacirl.utils import run_util


@click.group()
def cli():
    """Command line interface for IL RL."""


@cli.command()
@run_mode_options
def controller(config, seed, model_path=None, expert=False):
    run_util(config, seed, controller=True)


@cli.command()
@run_mode_options
def imitate(config, seed, model_path=None, expert=False):
    run_util(config, seed, imitate=True)


@cli.command()
@run_mode_options
def finetune(config, seed, model_path, expert=False):
    run_util(config, seed, model_path)


@cli.command()
@run_mode_options
def scratch(config, seed, model_path=None, expert=False):
    run_util(config, seed, scratch=True)


@cli.command()
@run_mode_options
def test(config, seed, model_path, expert):
    run_util(config, seed, model_path, test=True, expert=expert)


@cli.command()
@run_gym_options
def run(
    config,
    imitation,
    scratch,
    test,
    controller,
    seed,
    model_path=None,
    expert=None,
):
    from hvacirl.training.imitate import bc as imitate
    from hvacirl.training import reinforce

    from .controller import run as controller_run

    if scratch:
        reinforce.run(config, seed, scratch=True)
    elif controller:
        controller_run(config, seed)
    elif test:
        reinforce.test_run(config, seed, model_path, expert)
    elif imitation:
        imitate.run(config, seed)
    else:
        reinforce.run(config, seed, model_path)
