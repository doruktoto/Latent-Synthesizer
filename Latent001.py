import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

# Define the VAE model
class SynthesizerVAE(keras.Model):
    def __init__(self, latent_dim=2, **kwargs):
        super(SynthesizerVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_inputs = keras.Input(shape=(128,))
        x = layers.Dense(256, activation="relu")(encoder_inputs)
        x = layers.Dense(128, activation="relu")(x)
        
        # Mean and variance for latent space
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        
        # Sampling function
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
        
        # Decoder
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(128, activation="relu")(latent_inputs)
        x = layers.Dense(256, activation="relu")(x)
        decoder_outputs = layers.Dense(128, activation="sigmoid")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        
        # Store for later
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampling([z_mean, z_log_var])
            reconstruction = self.decoder(z)
            
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            ) * 128
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        reconstruction = self.decoder(z)
        return reconstruction

# Function to generate synthetic training data
def generate_synth_data(n_samples=1000):
    """Generate synthetic parameter data for training"""
    params = np.zeros((n_samples, 128))
    
    # Generate random synth parameters
    for i in range(n_samples):
        # Oscillator parameters
        params[i, 0:10] = np.random.uniform(0, 1, 10)  # Osc 1 params
        params[i, 10:20] = np.random.uniform(0, 1, 10)  # Osc 2 params
        
        # Filter parameters
        params[i, 20:30] = np.random.uniform(0, 1, 10)
        
        # Envelope parameters
        params[i, 30:40] = np.random.uniform(0, 1, 10)  # ADSR
        
        # LFO parameters
        params[i, 40:50] = np.random.uniform(0, 1, 10)
        
        # Effects parameters
        params[i, 50:128] = np.random.uniform(0, 1, 78)
        
    return params

# Function to synthesize audio from parameters (simplified example)
def synth_audio_from_params(params, sr=44100, duration=1.0):
    """Simplified function to generate audio from synth parameters"""
    # This is a placeholder for actual synthesis
    # In a real implementation, you would use a synth engine to generate audio
    
    # Extract some basic parameters
    freq = 110 + params[0] * 1000  # Base frequency
    mod_freq = 1 + params[10] * 20  # Modulation frequency
    mod_depth = params[11] * 0.5    # Modulation depth
    
    # Generate time array
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Simple FM synthesis
    carrier = np.sin(2 * np.pi * freq * t)
    modulator = mod_depth * np.sin(2 * np.pi * mod_freq * t)
    signal = np.sin(2 * np.pi * freq * t + modulator)
    
    # Apply simple envelope
    attack = int(params[30] * sr * 0.5)
    decay = int(params[31] * sr * 0.5)
    sustain = params[32]
    release = int(params[33] * sr * 0.5)
    
    env = np.ones_like(t)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack)
    if release > 0:
        env[-release:] = np.linspace(sustain, 0, release)
    
    return signal * env

# Main function to train the model and explore the latent space
def create_synth_latent_space(latent_dim=2, n_samples=1000):
    # Generate training data
    print("Generating synthetic training data...")
    synth_params = generate_synth_data(n_samples)
    
    # Create and train the VAE model
    print("Training VAE model...")
    vae = SynthesizerVAE(latent_dim=latent_dim)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(synth_params, epochs=20, batch_size=32, verbose=1)
    
    # Create a grid in latent space for exploration
    print("Creating latent space visualization...")
    grid_size = 10
    latent_grid = np.zeros((grid_size * grid_size, latent_dim))
    
    # Create a 2D grid for exploration
    if latent_dim >= 2:
        x_values = np.linspace(-3, 3, grid_size)
        y_values = np.linspace(-3, 3, grid_size)
        
        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                latent_grid[i * grid_size + j, 0] = x
                latent_grid[i * grid_size + j, 1] = y
    
    # Decode the grid points to get synth parameters
    decoded_params = vae.decoder.predict(latent_grid)
    
    # Visualize the latent space (for 2D)
    if latent_dim == 2:
        plt.figure(figsize=(12, 10))
        plt.scatter(latent_grid[:, 0], latent_grid[:, 1], c='blue', alpha=0.5)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Synthesizer Latent Space')
        plt.grid(True)
        plt.savefig('synth_latent_space.png')
        plt.show()
    
    return vae, decoded_params, latent_grid

# Function to play sound from a point in latent space
def play_from_latent_point(vae, x, y, sr=44100, duration=1.0):
    """Generate and play audio from a point in latent space"""
    # Create latent vector
    latent_point = np.array([[x, y]])
    
    # Decode to get synth parameters
    decoded_params = vae.decoder.predict(latent_point)[0]
    
    # Generate audio
    audio = synth_audio_from_params(decoded_params, sr=sr, duration=duration)
    
    # Play audio
    sd.play(audio, sr)
    sd.wait()
    
    return audio

# Interactive exploration function
def explore_latent_space(vae):
    """Simple function to explore the latent space interactively"""
    print("Latent Space Explorer")
    print("Enter x, y coordinates between -3 and 3 to hear sounds")
    print("Enter 'q' to quit")
    
    while True:
        input_str = input("Enter coordinates (x,y): ")
        if input_str.lower() == 'q':
            break
        
        try:
            x, y = map(float, input_str.split(','))
            if abs(x) > 3 or abs(y) > 3:
                print("Values should be between -3 and 3")
                continue
                
            print(f"Playing sound at ({x}, {y})...")
            audio = play_from_latent_point(vae, x, y)
        except ValueError:
            print("Invalid input. Please enter two numbers separated by a comma.")

if __name__ == "__main__":
    # Create and train the model
    vae, decoded_params, latent_grid = create_synth_latent_space(latent_dim=2, n_samples=2000)
    
    print("Model training completed")
    
    # Save the model
    try:
        vae.save_weights('synth_vae.weights.h5')
        print("Model weights saved successfully")
    except Exception as e:
        print(f"Error saving model weights: {e}")
    
    # Explore the latent space
    try:
        print("Starting latent space exploration...")
        explore_latent_space(vae)
        print("Exploration completed")
    except Exception as e:
        print(f"Error during exploration: {e}")