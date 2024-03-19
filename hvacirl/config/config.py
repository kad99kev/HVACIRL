import functools

import click
import yaml


def parse_config(filename):
    """
    Reads yaml configuration file.
    Arguments:
        filename: The name of the yaml file.
    """
    with open(filename, "r") as yfile:
        cfg = yaml.load(yfile, Loader=yaml.FullLoader)
    return cfg


# Reference: https://stackoverflow.com/a/52147284/13082658
def run_mode_options(f):
    options = [
        click.option(
            "--config",
            "-c",
            default=None,
            required=True,
            help="Path to configuration file.",
            type=str,
        ),
        click.option(
            "--seed", "-s", required=True, help="Seed of experiment.", type=int
        ),
        click.option(
            "--model_path",
            "-m",
            help="Path of model if evaluating or finetuning.",
            type=str,
        ),
        click.option(
            "--expert",
            "-e",
            is_flag=True,
            help="Whether the model is an expert (only for testing).",
        ),
    ]
    return functools.reduce(lambda x, opt: opt(x), options, f)


def run_gym_options(f):
    options = [
        click.option(
            "--config",
            "-c",
            default=None,
            required=True,
            help="Path to configuration file.",
            type=str,
        ),
        click.option(
            "--imitation", "-i", is_flag=True, help="Whether to run imitation learning."
        ),
        click.option(
            "--scratch",
            "-s",
            is_flag=True,
            help="Whether to run experiment from scratch.",
        ),
        click.option(
            "--test", "-t", is_flag=True, help="Whether to run experiment in test mode."
        ),
        click.option(
            "--controller",
            "-cntr",
            is_flag=True,
            help="Whether to run experiment with a rule-based controller agent.",
        ),
        click.option(
            "--seed", "-se", required=True, help="Seed of experiment.", type=int
        ),
        click.option(
            "--model_path",
            "-m",
            required=False,
            help="Path of model if evaluating or finetuning.",
            type=str,
        ),
        click.option(
            "--expert",
            "-e",
            required=False,
            is_flag=True,
            help="Whether the model is an expert (only for testing).",
        ),
    ]
    return functools.reduce(lambda x, opt: opt(x), options, f)
