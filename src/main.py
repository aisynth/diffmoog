import os
import subprocess
import sys
import torch

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from shutil import rmtree

from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder
from pytorch_lightning.loggers import TensorBoardLogger

from termcolor import colored

from dataset.synth_datamodule import ModularSynthDataModule
from utils.gpu_utils import get_device

from model.lit_module import LitModularSynth
from utils.train_utils import get_project_root


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


root = get_project_root()
EXP_ROOT = root.joinpath('experiments', 'current')
DATA_ROOT = root.joinpath('data')


def run(run_args):

    exp_name = run_args.experiment
    dataset_name = run_args.dataset

    cfg = configure_experiment(exp_name, dataset_name, run_args.config, run_args.debug)

    datamodule = ModularSynthDataModule(cfg.data_dir,
                                        cfg.model.batch_size,
                                        cfg.model.num_workers,
                                        cfg.loss.in_domain_epochs,
                                        added_noise_std=cfg.synth.added_noise_std)

    # todo: allow config of out of domain data
    datamodule.setup()

    device = get_device(run_args.gpu_index)
    torch.set_float32_matmul_precision('medium')

    is_load_ckpt = False
    if cfg.model.get('ckpt_path', None):
        is_load_ckpt = True
        lit_module = LitModularSynth.load_from_checkpoint(checkpoint_path=cfg.model.ckpt_path, train_cfg=cfg,
                                                          device=device)
    else:
        lit_module = LitModularSynth(cfg, device)

    tb_logger = TensorBoardLogger(cfg.logs_dir, name=exp_name)
    lit_module.tb_logger = tb_logger.experiment

    next_version = get_next_version('./my_checkpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./my_checkpoints/version_{next_version}',  # Use next version for versioning
        filename='{epoch}-{train_loss:.2f}',  # the filename includes epoch number and validation loss
        save_top_k=-1,  # set to save all checkpoints
        verbose=True,
        monitor='train_loss',
        mode='min',
    )

    callbacks = [LearningRateMonitor(logging_interval='step'), checkpoint_callback]
    # callbacks = [LearningRateMonitor(logging_interval='step'), checkpoint_callback, FineTuneLearningRateFinder(milestones=(5, 10))]

    if len(datamodule.train_dataset.params) < 50:
        log_every_n_steps = len(datamodule.train_dataset.params)
    else:
        log_every_n_steps = 50

    seed_everything(42, workers=True)

    trainer = Trainer(logger=tb_logger,
                      callbacks=callbacks,
                      max_epochs=cfg.model.num_epochs,
                      accelerator="gpu",
                      detect_anomaly=True,
                      # log_every_n_steps=log_every_n_steps,
                      log_every_n_steps=1,
                      check_val_every_n_epoch=1,
                      accumulate_grad_batches=4,
                      reload_dataloaders_every_n_epochs=cfg.loss.in_domain_epochs)

    if is_load_ckpt:
        trainer.fit(lit_module, datamodule=datamodule, ckpt_path=cfg.model.ckpt_path)
    else:
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


def get_next_version(base_dir):
    existing_versions = [d for d in os.listdir(base_dir) if
                         os.path.isdir(os.path.join(base_dir, d)) and "version_" in d]
    existing_versions.sort(key=lambda x: int(x.split("_")[1]))

    if not existing_versions:
        return 0
    else:
        last_version = int(existing_versions[-1].split("_")[1])
        return last_version + 1


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
