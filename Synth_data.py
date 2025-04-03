import numpy as np


def generate_synth_data(n_samples=1000):
    params = np.zeros((n_samples, 128))

    for i in range(n_samples):
        # Oscillator 1: base freq, detune, waveform
        params[i, 0] = np.random.uniform(0.1, 0.9)        # Base pitch
        params[i, 1] = np.random.normal(0.5, 0.2)          # Detune
        params[i, 2] = np.random.choice([0, 0.33, 0.66])   # Waveform type (quantized)

        # Oscillator 2: slightly different for stereo or layering
        params[i, 10] = np.random.uniform(0.2, 1.0)        # Pitch offset
        params[i, 11] = np.random.normal(0.5, 0.3)         # Detune
        params[i, 12] = np.random.choice([0, 0.33, 0.66])  # Waveform type

        # Filter settings (emphasize mid-range more)
        params[i, 20] = np.random.beta(2, 5)  # Skew toward lower cutoff
        params[i, 21] = np.random.uniform(0.1, 1.0)  # Resonance

        # Envelope (ADSR with variation)
        params[i, 30] = np.random.uniform(0.01, 0.3)  # Attack
        params[i, 31] = np.random.uniform(0.01, 0.5)  # Decay
        params[i, 32] = np.random.uniform(0.5, 1.0)   # Sustain
        params[i, 33] = np.random.uniform(0.05, 0.5)  # Release

        # Modulation (LFOs or FM depth)
        params[i, 40] = np.random.choice([0.0, 0.5, 1.0])  # On/off states
        params[i, 41] = np.random.normal(0.5, 0.3)

        # Effects (reverb, distortion, etc. - simplified here)
        params[i, 50:90] = np.random.uniform(0, 1, 40)
        params[i, 90:128] = np.random.normal(0.5, 0.15, 38)

        # Clip values to 0–1 range
        params[i] = np.clip(params[i], 0.0, 1.0)

    return params