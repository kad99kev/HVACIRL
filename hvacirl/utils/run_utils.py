import functools
import pathlib
import shutil
import subprocess

from hvacirl.config import parse_config


def run_util(
    config_path,
    seed,
    model_path=None,
    imitate=False,
    scratch=False,
    test=False,
    controller=False,
    expert=False,
):
    # Parse configuration and create folder.
    cfg = parse_config(config_path)
    cfg["run_name"] += "_" + str(seed)
    save_path = pathlib.Path("runs/" + cfg["run_name"])
    save_path.mkdir(parents=True, exist_ok=True)

    config_path = pathlib.Path(config_path)
    shutil.copy(config_path, save_path)
    container_cfg_path = save_path / config_path.name

    if test or not (imitate or scratch or controller):
        assert (
            model_path is not None
        ), "You need to specify a model path if you are testing or finetuning."
    
    subprocess.run(
        [
            f"hvacirl",
            f"run",
            "-se",
            f"{seed}",
            "-c",
            f"{container_cfg_path}",
        ]
        + (["-m", f"{model_path}"] if model_path else [])
        + (["-e"] if expert else [])
        + (["-i"] if imitate else [])
        + (["-s"] if scratch else [])
        + (["-t"] if test else [])
        + (["-cntr"] if controller else [])
    )
