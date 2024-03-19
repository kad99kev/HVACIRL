from setuptools import find_packages, setup

requirements = [
    "wandb",
    "gymnasium",
    "torch",
    "tensorboard",
    "termcolor",
    "tqdm",
    "scikit-learn",
    "plotly",
    "kaleido",
]

setup(
    name="hvacirl",
    licence="GPLv3",
    version="0.1",
    url="https://github.com/kad99kev/HVACIRL",
    author="Kevlyn Kadamala",
    author_email="k.kadamala1@universityofgalway.ie",
    description="Enhancing HVAC Control Efficiency: A Hybrid Approach Using Imitation and Reinforcement Learning.",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hvacirl = hvacirl.cli:cli",
        ],
    },
)
