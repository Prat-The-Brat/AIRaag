import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load the Saptak.wav file
file_path = 'Data/Saptak.wav'
song, sr = librosa.load(file_path)

# Display the waveform
plt.figure()
librosa.display.waveshow(song, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.show(block=False)
input("Press Enter to continue...")
plt.close()  # Close the waveform figure

# Calculate and display the STFT
hop_length = 512
n_fft = 2048
stft = librosa.stft(song, hop_length=hop_length, n_fft=n_fft)
stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

plt.figure(figsize=(10, 6))
librosa.display.specshow(
    stft_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.xlabel("Time")
plt.ylabel("Frequency (Hz)")
plt.title("Short Time Fourier Transform (STFT) Heatmap")
plt.show(block=False)
input("Press Enter to continue...")
plt.close()  # Close the STFT figure

# Calculate and display the Mel filter banks graph
n_mels = 40
mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

plt.figure(figsize=(8, 6))
librosa.display.specshow(
    mel, sr=sr, hop_length=hop_length, x_axis='linear', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mel Filter Banks")
plt.title("Mel Filter Banks Graph")
plt.show(block=False)
input("Press Enter to continue...")
plt.close()  # Close the Mel filter banks figure

# Calculate and display the Mel-Frequency Cepstral Coefficients (MFCC)
mfcc_song = librosa.feature.mfcc(
    y=song, sr=sr, hop_length=hop_length, n_mfcc=13)

plt.figure(figsize=(10, 6))
librosa.display.specshow(
    mfcc_song, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar()
plt.xlabel("Time")
plt.ylabel("MFCC Coefficients")
plt.title("Mel-Frequency Cepstral Coefficients (MFCC)")
plt.show()
