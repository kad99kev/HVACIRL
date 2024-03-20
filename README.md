# Enhancing HVAC Control Efficiency: A Hybrid Approach Using Imitation and Reinforcement Learning

## Setup
This repository runs in a Docker container configured by [Sinergym](https://ugr-sail.github.io/sinergym/compilation/v3.1.0/index.html).

Follow the instructions on [how to install Sinergym via Docker](https://ugr-sail.github.io/sinergym/compilation/v3.1.0/pages/installation.html#docker-container) and then follow the steps below.


## Installation

In a conda or virtual environment, run the following code.

```
git clone <this_repo_url>
pip install -e .
```

## Running an experiment.
Once the Docker container is built, there are different options available:
1. controller - Will run an experiment using a rule-based controller agent.
2. imitate - Will train an agent with imitation learning.
3. scratch - Will train a Deep RL agent from scratch (no fine-tuning).
4. finetune - Will finetune a Deep RL agent using pre-trained weights.
5. test- Will test any agent (trained via imitate, scratch or finetune).

The commands can be run as follows:
```
hvacirl scratch -c path/to/config -s 0
```

Run `hvacirl --help` for more information.

Example configuration file is given in `example_cfg.yaml`.

## Dataset generation.

To generate the dataset used for pre-training, run the `data_collector.ipynb` Jupyter Notebook. This will generate `.csv` files that can then be used for pre-training.
