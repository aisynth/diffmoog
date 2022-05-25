CUMSUM_TIME_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                    'multi_spectral_loss_type': 'L1',
                    'multi_spectral_cumsum_time_weight': 1/2000,
                    'normalize_loss_by_nfft': False}

CUMSUM_TIME_LOW_FFT_LOSS = {'fft_sizes': (128, 64),
                            'multi_spectral_loss_type': 'L1',
                            'multi_spectral_cumsum_time_weight': 1/2000,
                            'normalize_loss_by_nfft': True}

CUMSUM_TIME_HIGH_FFT_LOSS = {'fft_sizes': (2048, 1024),
                             'multi_spectral_loss_type': 'L1',
                             'multi_spectral_cumsum_time_weight': 1/2000,
                             'normalize_loss_by_nfft': False}

FM_ONLY_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                'multi_spectral_loss_type': 'L1',
                'multi_spectral_cumsum_time_weight': 1/2000,
                'normalize_loss_by_nfft': False}

loss_presets = {'cumsum_time': CUMSUM_TIME_LOSS,
                'cumsum_time_low_fft': CUMSUM_TIME_LOW_FFT_LOSS,
                'cumsum_time_high_fft': CUMSUM_TIME_HIGH_FFT_LOSS,
                'lfo_only': CUMSUM_TIME_LOSS,
                'fm_only': FM_ONLY_LOSS}
