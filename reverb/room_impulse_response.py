import numpy
import torch

from scipy.io.wavfile import read

rir100 = read("room_size_100_no_amp.wav")
rir80 = read("room_size_80_no_amp.wav")
rir60 = read("room_size_60_no_amp.wav")
rir40 = read("room_size_40_no_amp.wav")
rir20 = read("room_size_20_no_amp.wav")
rir0 = read("room_size_0_no_amp.wav")

rir100 = torch.from_numpy(numpy.array(rir100[1], dtype='float32'))
rir80 = torch.from_numpy(numpy.array(rir80[1], dtype='float32'))
rir60 = torch.from_numpy(numpy.array(rir60[1], dtype='float32'))
rir40 = torch.from_numpy(numpy.array(rir40[1], dtype='float32'))
rir20 = torch.from_numpy(numpy.array(rir20[1], dtype='float32'))
rir0 = torch.from_numpy(numpy.array(rir0[1], dtype='float32'))

if __name__ == "__main__":
    dic = {'rir100': rir100, 'rir80': rir80, 'rir60': rir60, 'rir40': rir40, 'rir20': rir20, 'rir0': rir0}
    torch.save(dic, 'rir_for_reverb_no_amp')

    rirs = torch.load('rir_for_reverb_no_amp')
    print(rirs['rir100'])
