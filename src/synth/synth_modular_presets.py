from synth.synth_architecture import SynthModularCell, SynthModular
from config import SynthConfig, Config

cfg = Config()
synth_cfg = SynthConfig()
synth_modular = SynthModular(synth_cfg=synth_cfg,
                             num_channels=synth_cfg.num_channels,
                             num_layers=synth_cfg.num_layers,
                             sample_rate=cfg.sample_rate,
                             signal_duration_sec=cfg.signal_duration_sec)
BASIC_FLOW = [
    SynthModularCell(index=(0, 0), operation='osc', default_connection=True, synth_config=synth_cfg),
    SynthModularCell(index=(0, 1), operation='fm', default_connection=True, synth_config=synth_cfg),
    SynthModularCell(index=(1, 0), operation='osc', default_connection=True, synth_config=synth_cfg),
    SynthModularCell(index=(1, 1), operation='fm', input_list=[[1, 0]], output=[0, 2], synth_config=synth_cfg),
    SynthModularCell(index=(1, 2), operation=None, input_list=None, synth_config=synth_cfg),
    SynthModularCell(index=(0, 2),
                     operation='mix',
                     input_list=[[0, 1], [1, 1]], synth_config=synth_cfg),
    SynthModularCell(index=(0, 3), operation='filter', default_connection=True, synth_config=synth_cfg),
    SynthModularCell(index=(0, 4), operation='env_adsr', default_connection=True, synth_config=synth_cfg)
]

FM = [
    SynthModularCell(index=(0, 0), operation='osc', default_connection=True, synth_config=synth_cfg),
    SynthModularCell(index=(0, 1), operation='fm', default_connection=True, synth_config=synth_cfg)
]