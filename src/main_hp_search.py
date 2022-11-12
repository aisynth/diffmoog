import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from shutil import rmtree

import torch
import optuna
import argparse

from omegaconf import OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
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



# def objective(trial: optuna.trial.Trial) -> float:
#
#     # We optimize the number of layers, hidden units in each layer and dropouts.
#     # n_layers = trial.suggest_int("n_layers", 1, 3)
#     lr = trial.suggest_float("lr", 1e-6, 1e-2)
#     # output_dims = [
#     #     trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
#     # ]
#
#     model = LightningNet(dropout, output_dims)
#     datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)
#
#     trainer = pl.Trainer(
#         logger=True,
#         limit_val_batches=PERCENT_VALID_EXAMPLES,
#         enable_checkpointing=False,
#         max_epochs=EPOCHS,
#         gpus=1 if torch.cuda.is_available() else None,
#         callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
#     )
#     hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
#     trainer.logger.log_hyperparams(hyperparameters)
#     trainer.fit(model, datamodule=datamodule)
#
#     return trainer.callback_metrics["val_acc"].item()

#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
#     parser.add_argument(
#         "--pruning",
#         "-p",
#         action="store_true",
#         help="Activate the pruning feature. `MedianPruner` stops unpromising "
#         "trials at the early stages of training.",
#     )
#     args = parser.parse_args()
#
#     pruner: optuna.pruners.BasePruner = (
#         optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
#     )
#
#     study = optuna.create_study(direction="maximize", pruner=pruner)
#     study.optimize(objective, n_trials=100, timeout=600)
#
#     print("Number of finished trials: {}".format(len(study.trials)))
#
#     print("Best trial:")
#     trial = study.best_trial
#
#     print("  Value: {}".format(trial.value))
#
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))
#
#


def objective(trial: optuna.trial.Trial, run_args) -> float:

    exp_name = run_args.experiment
    dataset_name = run_args.dataset

    cfg = configure_experiment(exp_name, dataset_name, run_args.config, run_args.debug)

    lr = trial.suggest_float("lr", 1e-6, 1e-2)
    cfg.model.optimizer.base_lr = lr

    datamodule = ModularSynthDataModule(cfg.data_dir, cfg.model.batch_size, cfg.model.num_workers,
                                        added_noise_std=cfg.synth.added_noise_std)

    # todo: allow config of out of domain data
    datamodule.setup()

    device = get_device(run_args.gpu_index)

    lit_module = LitModularSynth(cfg, device, tuning_mode=True)
    if cfg.model.get('ckpt_path', None):
        lit_module.load_from_checkpoint(checkpoint_path=cfg.model.ckpt_path, train_cfg=cfg, device=device)

    callbacks = [LearningRateMonitor(logging_interval='step'),
                 PyTorchLightningPruningCallback(trial, monitor="train_lsd_val")]

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
                      check_val_every_n_epoch=500,
                      enable_checkpointing=False)

    hyperparameters = dict(lr=cfg.model.optimizer.base_lr)
    trainer.logger.log_hyperparams(hyperparameters)

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
    parser.add_argument('-p', '--pruning', required=False, action='store_true',
                        help='use pruning in optuna', default=False)

    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(lambda x: objective(x, args), n_trials=3, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    for trial in study.trials:
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
