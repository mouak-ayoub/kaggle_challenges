
import numpy as np
from scipy.io.wavfile import write


def generate_audio_from_function( duration=1.0, fs=44100):
    """
    Generate an audio signal from a mathematical function.

    Parameters:
    - func: A function that takes a time array and returns an audio signal.
    - duration: Duration of the audio in seconds.
    - fs: Sampling frequency.

    Returns:
    - y: Generated audio signal.
    """

    # Parameters
    sample_rate = 44100  # Sampling rate in Hz
    duration = 2.0  # Duration in seconds
    frequency = 440000.0  # Frequency of the sine wave in Hz (A4 note)

    # Generate time vector
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Generate sine wave
    sine_wave = 0.05 * np.sin(2 * np.pi * 0.001* frequency * t)   # Amplitude scaled to 0.5

    # Save as a WAV file
    write("function_wave_7.wav", sample_rate, (sine_wave * 32767).astype(np.int16))  # Convert to 16-bit PCM

    print("Sine wave audio generated and saved as 'sine_wave.wav'")

generate_audio_from_function()

