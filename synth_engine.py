import numpy as np
from scipy.signal import butter, lfilter

def synth_audio_from_params(params, sr=44100, duration=1.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    base_freq = 110 + params[0] * 880
    detune = (params[1] - 0.5) * 10
    waveform_type = int(params[2] * 3)

    def get_waveform(wave_type, freq):
        if wave_type == 0:
            return np.sin(2 * np.pi * freq * t)
        elif wave_type == 1:
            return np.sign(np.sin(2 * np.pi * freq * t))
        elif wave_type == 2:
            return 2 * (t * freq % 1) - 1
        else:
            return np.zeros_like(t)

    osc1 = get_waveform(waveform_type, base_freq)
    osc2 = get_waveform(waveform_type, base_freq + detune)
    mix = (osc1 + osc2) * 0.5

    attack = int(params[30] * sr * 0.5)
    decay = int(params[31] * sr * 0.5)
    sustain = params[32]
    release = int(params[33] * sr * 0.5)

    env = np.ones_like(t)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack)
    if decay > 0:
        env[attack:attack+decay] = np.linspace(1, sustain, decay)
    if release > 0:
        env[-release:] = np.linspace(sustain, 0, release)

    signal = mix * env

    cutoff = 500 + params[20] * 4000
    filtered = apply_filter(signal, cutoff, sr)

    filtered /= np.max(np.abs(filtered) + 1e-6)
    return filtered
def butter_lowpass(cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_filter(data, cutoff, sr):
    b, a = butter_lowpass(cutoff, sr)
    return lfilter(b, a, data)