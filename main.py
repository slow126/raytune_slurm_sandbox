import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from typing import Dict
from ray.tune.schedulers import ASHAScheduler
from ray.train import ScalingConfig, CheckpointConfig, RunConfig

from omegaconf import OmegaConf
import importlib
from ray.train.torch import TorchTrainer

from functools import partial

from pathlib import Path
import torch
import numba.cuda
from ray.tune import TuneConfig
import ray

from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

import sys
import os
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
if ROOT_DIR is not None:
    sys.path.append(ROOT_DIR)


from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

def load_yaml(yaml_path: str) -> Dict:
    # Use OmegaConf to load YAML with environment variable interpolation
    config = OmegaConf.load(yaml_path)
    # Convert to dict for backward compatibility
    return OmegaConf.to_container(config, resolve=True)

def load_class_from_string(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def train_func(config: Dict):
    # Extract model configuration and instantiate model
    model_config = config["model"]
    ModelClass = load_class_from_string(model_config["class_path"])
    model_init_args = model_config.get("init_args", {}).copy()
    model = ModelClass(**model_init_args)

    # Configure trainer
    trainer_config = config.get("trainer", {})
    
    lr_monitor = pl.callbacks.LearningRateMonitor()
    
    trainer = pl.Trainer(
        max_epochs=trainer_config.get("max_epochs", 100),
        accelerator="gpu",
        devices=trainer_config.get("devices", "auto"),
        logger=pl.loggers.TensorBoardLogger(
            save_dir=Path(trainer_config.get("default_root_dir", "lightning_logs")) / "tensorboard"
        ),
        callbacks=[RayTrainReportCallback()],
        precision=trainer_config.get("precision", "16-mixed"),
        gradient_clip_val=trainer_config.get("gradient_clip_val", 1.0),
        strategy=RayDDPStrategy(find_unused_parameters=True),
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=True,
    )
    
    # Train the model
    trainer = prepare_trainer(trainer)
    results = trainer.fit(model)
    print("TRAINING RESULTS: ")
    print(results)

    return {"val/loss": results["val/loss"]}

def simple_train_func(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        try:
            x = torch.randn(10, 10).to(device) # Simple GPU operation
            print("GPU operation successful")
        except Exception as e:
            print(f"GPU operation failed: {e}")
            raise
    else:
        print("Skipping GPU operation as CUDA is not available.")

    ray.train.report({"val/acc": 1}) # Required for Ray Tune


import multiprocessing
if __name__ == "__main__":
    # Get the data directory from .env file
    import os
    import dotenv
    from torch.utils.data import DataLoader
    multiprocessing.set_start_method("spawn", force=True)  # Set spawn mode
    dotenv.load_dotenv('.env')

    import argparse

    parser = argparse.ArgumentParser(description="Launch Ray Tune with specified resources and YAML config.")
    parser.add_argument("--yaml_path", type=str, default='configs/raytune.yaml', help="Path to the YAML config file")
    parser.add_argument("--node_ip", type=str, default=None, help="IP address of the node to run on")
    parser.add_argument("--node_port", type=int, default=6379, help="Port to run on")

    args = parser.parse_args()
    yaml_path = args.yaml_path
    node_ip = args.node_ip
    node_port = args.node_port

    path = Path(__file__).parent.joinpath('.').resolve()

    # Load base config
    from omegaconf import OmegaConf
    base_config_path = path.joinpath(yaml_path)
    base_config = OmegaConf.load(base_config_path)
    base_config_dict = OmegaConf.to_container(base_config, resolve=True)

    params_to_tune = base_config_dict['raytune']['params_to_tune']

    search_space = {}

    search_space = {}
    search_space_func = {
        "choice": tune.choice,
        "uniform": tune.uniform,
        "loguniform": tune.loguniform,
        "quniform": tune.quniform,
        "qloguniform": tune.qloguniform,
        "grid_search": tune.grid_search,
    }

    for param_name in params_to_tune.keys():
        func = search_space_func[params_to_tune[param_name]['type']]
        search_space[param_name] = func(params_to_tune[param_name]['values'])


    ray_results_dir = base_config_dict['raytune']['log_dir']
    # Convert ray_results_dir to absolute path
    if ray_results_dir is not None:
        ray_results_dir = os.path.abspath(ray_results_dir)
        print(f"Ray results directory: {ray_results_dir}")
        Path(ray_results_dir).mkdir(exist_ok=True, parents=True)
    else:
        ray_results_dir = None
        print("Ray results directory not set, using default")
    ray_init_args = {}
    if args.node_ip is not None:
        ray_init_args["address"] = f"ray://{args.node_ip}:{args.node_port}"
    ray_init_args["num_cpus"] = base_config_dict['raytune']['num_cpus']
    ray_init_args["runtime_env"] = {"env_vars": {"CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]}}


    ray.init(**ray_init_args) 

    workdir = os.getenv('WORKDIR', '')

    search_alg = OptunaSearch(metric="val/acc", mode="max")
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=base_config_dict['raytune']['concurrent_trials'])

    max_t = base_config_dict['raytune'].get('max_t', 40)
    grace_period = base_config_dict['raytune'].get('grace_period', 5)
    num_samples = base_config_dict['raytune'].get('num_samples', 100)
    reuse_actors = base_config_dict['raytune'].get('reuse_actors', False)
    num_to_keep = base_config_dict['raytune'].get('num_to_keep', num_samples)
    

    def inject_search_space(base_config_dict, search_space):
        for key, value in search_space.items():
            keys = key.split('.')
            current_dict = base_config_dict
            for i in range(len(keys) - 1):
                current_key = keys[i]
                if current_key not in current_dict:
                    current_dict[current_key] = {}
                current_dict = current_dict[current_key]
            current_dict[keys[-1]] = value
        return base_config_dict

    base_config_dict = inject_search_space(base_config_dict, search_space)

    scaling_config=ScalingConfig(
        num_workers=base_config_dict['raytune']['num_workers'],
        use_gpu=True,
        resources_per_worker={"CPU": base_config_dict['raytune']['cpus_per_worker'], "GPU": base_config_dict['raytune']['gpus_per_worker']},
    )
    run_config=RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=num_to_keep,
            checkpoint_score_attribute="val/loss",
            checkpoint_score_order="min",
        ),
    )

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=max_t,
        grace_period=grace_period,
    )

    # Import the ViT model
    from models.vit_lightning.vit_lightning import ViTLightning

    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": base_config_dict},
        tune_config=TuneConfig(
            metric="val/loss",
            mode="min",
            search_alg=search_alg,
            num_samples=num_samples,
            scheduler=scheduler,
            reuse_actors=reuse_actors,
        ),
    )

    results = tuner.fit()
    import pickle
    results_file = os.path.join(base_config_dict['raytune']['log_dir'], "raytune_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Ray Tune results saved to: {results_file}")

    best_result = results.get_best_result(metric="val/acc", mode="max")
    best_results_file = os.path.join(base_config_dict['raytune']['log_dir'], "best_raytune_result.pkl")
    with open(best_results_file, "wb") as f:
        pickle.dump(best_result, f)
    print(f"Ray Tune best result saved to: {best_results_file}")
    print(best_result)
