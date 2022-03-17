import os
from dataclasses import dataclass, field
from pathlib import Path


# from synth_config import BASIC_FLOW

@dataclass
class Config:
    sample_rate = 16000
    signal_duration_sec = 1.0

    " Mode - define a common configuration for the whole system     "
    "   0 -                     Use custom configurations           "
    "   Any other number -      Use predefined configuration preset. See below "
    mode: int = 1

    " The architecture of the system, that defines the data flow and the loss functions:                    "
    "   1. SPECTROGRAM_ONLY (input -> CNN -> parameters -> Synth -> output; Loss over spectrograms)         "
    "   2. PARAMETERS_ONLY (input -> CNN -> parameters; Loss over parameters)                               "
    "   3. FULL - (input -> CNN -> parameters -> Synth -> output; Loss over spectrograms AND parameters)    "
    "   4. SPEC_NO_SYNTH (input -> CNN -> parameters); Output inner product <probabilities, spectrograms>;   "
    "      Loss over spectrograms)                                                                          "
    "   5. REINFORCE - (input -> CNN -> parameters); Loss is computed to maximize rewards for correct       "
    "       classification. Using the classical REINFORCE algorithm                                         "
    architecture = 'SPECTROGRAM_ONLY'  # SPECTROGRAM_ONLY, PARAMETERS_ONLY, SPEC_NO_SYNTH or FULL (Spectrogram + parameters)

    " Spectrogram loss type options:" \
    "1. MSE" \
    "2. LSD (Log Spectral Distance)" \
    "3. KL (Kullback-Leibler)" \
    "4. EMD (earth movers distance)" \
    "5. MULTI-SPECTRAL"
    spectrogram_loss_type = 'MULTI-SPECTRAL'
    freq_param_loss_type = 'MSE'  # MSE or CE (Cross Entropy)

    " The model can output the oscillator frequency as:                                 "
    "   1. LOGITS (size is num of frequencies, for cross entropy loss)                  "
    "   2. PROBS (same as LOGITS, but softmax is applied)                               "
    "   3. WEIGHTED - inner product of <probabilities, original frequencies>. size is 1 "
    "   4. SINGLE - linear layer outputs single neuron. size is 1                       "
    model_frequency_output = 'SINGLE'
    transform = 'MEL_SPECTROGRAM'  # MEL_SPECTROGRAM or SPECTROGRAM- to be used in the data loader and at the synth output

    use_loaded_model = False

    save_model_path = Path(__file__).parent.parent.joinpath('trained_models', 'trained_synth_net.pt')
    load_model_path = Path(__file__).parent.parent.joinpath('trained_models', 'synth_net_epoch1.pt')

    txt_path = Path(__file__).parent.parent.joinpath('trained_models', 'loss_list.txt')
    numpy_path = Path(__file__).parent.parent.joinpath('trained_models', 'loss_list.npy')

    num_epochs_to_save_model = 2

    regression_loss_factor = 1e-1
    spectrogram_loss_factor = 1e-5
    freq_mse_loss_factor = 1e-3
    freq_reinforce_loss_factor = 1e5

    # multi-spectral loss configs
    multi_spectral_loss_type = 'L1'
    multi_spectral_mag_weight = 1/100
    multi_spectral_delta_time_weight = 1/100
    multi_spectral_delta_freq_weight = 1/100
    multi_spectral_cumsum_freq_weight = 1/27400
    multi_spectral_logmag_weight = 1

    # Debug
    debug_mode = False
    plot_spec = False
    print_train_batch_stats = False
    print_timings = True
    print_synth_param_stats = True
    print_accuracy_stats = False
    print_per_accuracy_stats_multiple_epochs = True

    log_spectrogram_mse_loss = False

    def __post_init__(self):

        if self.log_spectrogram_mse_loss:
            self.spectrogram_loss_factor = 1000

        if self.freq_param_loss_type == 'CE':
            self.model_frequency_output = 'LOGITS'

        if self.spectrogram_loss_type == 'MULTI-SPECTRAL':
            # one of ['BOTH', 'MEL_SPECTROGRAM', 'SPECTROGRAM']
            self.multi_spectral_loss_spec_type = 'MEL_SPECTROGRAM'

        if self.mode == 1:
            self.architecture = 'SPECTROGRAM_ONLY'
            self.spectrogram_loss_type = 'MULTI-SPECTRAL'
            self.model_frequency_output = 'SINGLE'




@dataclass
class DatasetConfig:
    dataset_size = 1000
    num_epochs_to_print_stats = 100
    num_epochs_to_save_model = 100
    train_parameters_file = Path(__file__).parent.parent.joinpath('dataset', 'train', 'params_dataset.pkl')
    train_audio_dir = Path(__file__).parent.parent.joinpath('dataset', 'train', 'wav_files')
    test_parameters_file = Path(__file__).parent.parent.joinpath('dataset', 'test', 'params_dataset.pkl')
    test_audio_dir = Path(__file__).parent.parent.joinpath('dataset', 'test', 'wav_files')
    inference_audio_dir = Path(__file__).parent.parent.joinpath('dataset', 'test', 'inference_wav_files')
    inference_plots_dir = Path(__file__).parent.parent.joinpath('dataset', 'test', 'inference_plots')
    train_dataset_dir_path = Path(__file__).parent.parent.joinpath('dataset', 'train')
    test_dataset_dir_path = Path(__file__).parent.parent.joinpath('dataset', 'test')


@dataclass
class ModelConfig:
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.0001
    optimizer_weight_decay = 0
    optimizer_scheduler_lr = 30
    optimizer_scheduler_gamma = 0.1
    reinforcement_epsilon = 0.15


@dataclass
class SynthConfig:
    preset = 'LFO'
    wave_type_dict = {"sine": 0,
                      "square": 1,
                      "sawtooth": 2}

    filter_type_dict = {"low_pass": 0,
                        "high_pass": 1}

    semitones_max_offset: int = 24
    middle_c_freq: float = 261.6255653005985
    max_amp = 1
    max_mod_index = 100
    max_lfo_freq = 20
    min_filter_freq = 0
    max_filter_freq = 20000

    # When predicting the oscillator frequency by regression, the defines are used to normalize the output from the model
    margin = 200
    # --------------------------------------
    # -----------Modular Synth--------------
    # --------------------------------------
    # Modular Synth attributes:
    num_channels = 4
    num_layers = 5

    # Modular synth possible modules from synth_modules.py
    modular_synth_operations = ['osc', 'fm', 'lfo', 'mix', 'filter', 'env_adsr']
    modular_synth_params = {'osc': ['amp', 'freq', 'waveform'],
                            'lfo': ['amp', 'freq', 'waveform'],
                            'fm': ['amp_c', 'freq_c', 'waveform', 'mod_index'],
                            'mix': None,
                            'filter': ['filter_freq', 'filter_type'],
                            'env_adsr': ['attack_t', 'decay_t', 'sustain_t', 'sustain_level', 'release_t']}

    def __post_init__(self):
        self.wave_type_dic_inv = {v: k for k, v in self.wave_type_dict.items()}
        self.filter_type_dic_inv = {v: k for k, v in self.filter_type_dict.items()}

        # build a list of possible frequencies
        self.semitones_list = [*range(-self.semitones_max_offset, self.semitones_max_offset + 1)]
        self.osc_freq_list = [self.middle_c_freq * (2 ** (1 / 12)) ** x for x in self.semitones_list]
        self.osc_freq_dic = {round(key, 4): value for value, key in enumerate(self.osc_freq_list)}
        self.osc_freq_dic_inv = {v: k for k, v in self.osc_freq_dic.items()}
        self.oscillator_freq = self.osc_freq_list[-1] + self.margin
