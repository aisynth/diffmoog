import os
import subprocess
import sys

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from shutil import rmtree

import torch

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from termcolor import colored

from dataset.synth_datamodule import ModularSynthDataModule
from utils.gpu_utils import get_device

from model.lit_module import LitModularSynth
from utils.train_utils import get_project_root


root = get_project_root()
EXP_ROOT = root.joinpath('experiments', 'current')
DATA_ROOT = root.joinpath('data')


def run(run_args):

    exp_name = run_args.experiment
    dataset_name = run_args.dataset

    cfg = configure_experiment(exp_name, dataset_name, run_args.config, run_args.debug)

    datamodule = ModularSynthDataModule(cfg.data_dir, cfg.model.batch_size, cfg.model.num_workers,
                                        added_noise_std=cfg.synth.added_noise_std)

    # todo: allow config of out of domain data
    datamodule.setup()

    device = get_device(run_args.gpu_index)

    lit_module = LitModularSynth(cfg, device)
    if cfg.model.get('ckpt_path', None):
        lit_module.load_from_checkpoint(checkpoint_path=cfg.model.ckpt_path, train_cfg=cfg, device=device)

    callbacks = [LearningRateMonitor(logging_interval='step')]

    tb_logger = TensorBoardLogger(cfg.logs_dir, name=exp_name)
    lit_module.tb_logger = tb_logger.experiment

    if len(datamodule.train_dataset.params) < 50:
        log_every_n_steps = len(datamodule.train_dataset.params)
    else:
        log_every_n_steps = 50
    trainer = Trainer(logger=tb_logger,
                      callbacks=callbacks,
                      max_epochs=cfg.model.num_epochs,
                      auto_select_gpus=True,
                      devices=[run_args.gpu_index],
                      accelerator="gpu",
                      detect_anomaly=True,
                      log_every_n_steps=log_every_n_steps,
                      check_val_every_n_epoch=1)
    trainer.fit(lit_module, datamodule=datamodule)


def configure_experiment(exp_name: str, dataset_name: str, config_name: str, debug: bool = False):

    exp_dir = os.path.join(EXP_ROOT, exp_name, '')
    data_dir = os.path.join(DATA_ROOT, dataset_name, '')
    config_path = os.path.join(root, 'configs', config_name)

    if os.path.isdir(exp_dir):
        if not debug:
            overwrite = input(colored(f"Folder {exp_dir} already exists. Overwrite previous experiment (Y/N)?"
                                      f"\n\tThis will delete all files related to the previous run!",
                                      'yellow'))
            if overwrite.lower() != 'y':
                print('Exiting...')
                exit()

        print("Deleting previous experiment...")
        rmtree(exp_dir)

    cfg = OmegaConf.load(config_path)

    cfg.exp_dir = exp_dir
    cfg.data_dir = data_dir
    cfg.ckpts_dir = os.path.join(exp_dir, 'checkpoints', '')
    cfg.logs_dir = os.path.join(exp_dir, 'tensorboard', '')

    config_dump_dir = os.path.join(cfg.exp_dir, 'config_dump', '')
    os.makedirs(config_dump_dir, exist_ok=True)

    config_dump_path = os.path.join(config_dump_dir, 'config.yaml')
    OmegaConf.save(cfg, config_dump_path)

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()

    args = " ".join(sys.argv[1:])
    txt_path = os.path.join(config_dump_dir, 'commit_and_args.txt')

    with open(txt_path, 'w') as f:
        f.write(f"Git commit: {commit}\n")
        f.write(f"Arguments: {args}\n")

    return cfg


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Train AI Synth')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing) -1 for cpu',
                        type=int, default=0)
    parser.add_argument('-e', '--experiment', required=True,
                        help='Experiment name', type=str)
    parser.add_argument('-d', '--dataset', required=True, type=str,
                        help='Dataset name')
    parser.add_argument('-c', '--config', required=True, type=str,
                        help='configuration file path')
    parser.add_argument('-de', '--debug', required=False, action='store_true',
                        help='run in debug mode', default=False)

    args = parser.parse_args()
    run(args)
