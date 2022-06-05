import os

import torch

from config import SynthConfig, Config
from dataset.ai_synth_dataset import AiSynthDataset, create_data_loader
from model import helper
from model.model import SimpleSynthNetwork
from run_scripts.inference.inference_helper import inference_loop
from synth.synth_architecture import SynthModular

device = 'cuda:3'
preset = 'BASIC_FLOW'

## Create dataset
dataset_to_visualize = 'basic_flow_dataset'
split_to_visualize = 'test'
data_dir = os.path.join('.data', dataset_to_visualize, split_to_visualize, '')

wav_files_dir = os.path.join(data_dir, 'wav_files', '')
params_csv_path = os.path.join(data_dir, 'params_dataset.pkl')

ai_synth_dataset = AiSynthDataset(params_csv_path, wav_files_dir, device)
test_dataloader = create_data_loader(ai_synth_dataset, 32, 4, shuffle=False)


## Create synth
synth_cfg = SynthConfig()
cfg = Config()

transform = helper.mel_spectrogram_transform(cfg.sample_rate).to(device)
normalizer = helper.Normalizer(cfg.signal_duration_sec, synth_cfg)

synth_obj = SynthModular(synth_cfg=synth_cfg,
                         sample_rate=cfg.sample_rate,
                         signal_duration_sec=cfg.signal_duration_sec,
                         device=device,
                         preset='preset')


## Load model
model_ckpt_path = r'experiments/current/basic_flow/ckpts/synth_net_epoch0.pt'
model = SimpleSynthNetwork(preset, synth_cfg, cfg, device, backbone='resnet').to(device)
model.load_state_dict(torch.load(model_ckpt_path))

model.eval()


res = inference_loop(synth_cfg, test_dataloader, transform, model, normalizer.denormalize, device)

print(res)