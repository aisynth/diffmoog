# multi-spectral loss configs
# multi_spectral_loss_type: str = 'L1'
# multi_spectral_mag_weight: float = 1/200
# multi_spectral_delta_time_weight: float = 0#1/200
# multi_spectral_delta_freq_weight: float = 0#1/10000
# multi_spectral_cumsum_freq_weight: float = 0#1/2000
# multi_spectral_cumsum_time_weight: float = 0#1/2000
# multi_spectral_logmag_weight: float = 0#10
# fft_sizes: tuple = (64, 128) #(2048, 1024, 512, 256, 128, 64)
# normalize_loss_by_nfft: bool = False


CUMSUM_TIME_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                    'multi_spectral_loss_type': 'L1',
                    'multi_spectral_cumsum_time_weight': 1/2000,
                    'normalize_loss_by_nfft': True}


CUMSUM_TIME_LOW_FFT_LOSS = {'fft_sizes': (128, 64),
                            'multi_spectral_loss_type': 'L1',
                            'multi_spectral_cumsum_time_weight': 1/2000,
                            'normalize_loss_by_nfft': True}

FM_ONLY_LOSS = {'fft_sizes': (128, 64),
                'multi_spectral_loss_type': 'L1',
                'multi_spectral_cumsum_time_weight': 1/2000,
                'multi_spectral_mag_weight': 1/200,
                'multi_spectral_mag_warmup': 0.25,
                'multi_spectral_mag_gradual': True,
                'normalize_loss_by_nfft': True}

loss_presets = {'cumsum_time': CUMSUM_TIME_LOSS,
                'cumsum_time_low_fft': CUMSUM_TIME_LOW_FFT_LOSS,
                'fm_only': FM_ONLY_LOSS}
