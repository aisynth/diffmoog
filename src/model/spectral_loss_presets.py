CUMSUM_TIME_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                    'multi_spectral_loss_type': 'L1',
                    'multi_spectral_cumsum_time_weight': 1/2000,
                    'normalize_loss_by_nfft': True}

CUMSUM_FREQ_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                    'multi_spectral_loss_type': 'L1',
                    'multi_spectral_cumsum_freq_weight': 1/2000,
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


CUMSUM_TIME_FREQ_W_LOGMAG = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                             'multi_spectral_loss_type': 'L1',
                             'multi_spectral_cumsum_time_weight': 1 / 2000,
                             'multi_spectral_cumsum_freq_weight': 1 / 5000,
                             'multi_spectral_mag_weight': 1/200,
                             'multi_spectral_mag_warmup': 0.25,
                             'multi_spectral_mag_gradual': True,
                             'normalize_loss_by_nfft': True}

loss_presets = {'cumsum_time': CUMSUM_TIME_LOSS,
                'cumsum_freq': CUMSUM_FREQ_LOSS,
                'cumsum_time_low_fft': CUMSUM_TIME_LOW_FFT_LOSS,
                'lfo_only': CUMSUM_TIME_LOSS,
                'fm_only': FM_ONLY_LOSS,
                'cumsum_time_freq_mag': CUMSUM_TIME_FREQ_W_LOGMAG}
