"""
Core audio processing utilities for gunshot detection.
"""
import numpy as np
import librosa
import scipy.signal as signal
from scipy.signal import butter, sosfilt, hilbert
import soundfile as sf
import os

class AudioProcessor:
    """Core audio processing class with various utilities."""
    
    def __init__(self, sr=16000, frame_size=512, hop_length=256):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.hann_window = np.hanning(frame_size)
        
    def load_audio(self, file_path, target_sr=None):
        """Load audio file with optional resampling."""
        if target_sr is None:
            target_sr = self.sr
            
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    
    def normalize_audio(self, audio, target_db=-3.0):
        """Normalize audio to target dB level."""
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return audio
            
        current_db = 20 * np.log10(rms)
        gain = 10 ** ((target_db - current_db) / 20)
        return audio * gain
    
    def apply_bandpass_filter(self, audio, lowcut=50, highcut=8000, order=4):
        """Apply bandpass filter to audio."""
        nyquist = 0.5 * self.sr
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(order, [low, high], btype='band', output='sos')
        filtered = sosfilt(sos, audio)
        return filtered
    
    def apply_preemphasis(self, audio, coeff=0.97):
        """Apply pre-emphasis filter."""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    
    def frame_audio(self, audio, frame_size=None, hop_length=None):
        """Split audio into frames."""
        if frame_size is None:
            frame_size = self.frame_size
        if hop_length is None:
            hop_length = self.hop_length
            
        num_frames = (len(audio) - frame_size) // hop_length + 1
        frames = []
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_size
            frame = audio[start:end]
            
            if len(frame) < frame_size:
                # Pad last frame
                frame = np.pad(frame, (0, frame_size - len(frame)))
                
            frames.append(frame)
            
        return np.array(frames)
    
    def apply_window(self, frames, window_type='hann'):
        """Apply window function to frames."""
        if window_type == 'hann':
            window = np.hanning(frames.shape[1])
        elif window_type == 'hamming':
            window = np.hamming(frames.shape[1])
        else:
            window = np.ones(frames.shape[1])
            
        return frames * window
    
    def compute_energy(self, audio):
        """Compute energy of audio signal."""
        return np.sum(audio**2)
    
    def compute_rms(self, audio):
        """Compute RMS value."""
        return np.sqrt(np.mean(audio**2))
    
    def compute_zero_crossing_rate(self, audio):
        """Compute zero-crossing rate."""
        return np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
    
    def compute_spectral_centroid(self, audio):
        """Compute spectral centroid."""
        spectrum = np.abs(np.fft.rfft(audio))
        frequencies = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        if np.sum(spectrum) == 0:
            return 0
            
        return np.sum(frequencies * spectrum) / np.sum(spectrum)
    
    def compute_spectral_rolloff(self, audio, roll_percent=0.85):
        """Compute spectral rolloff point."""
        spectrum = np.abs(np.fft.rfft(audio))
        frequencies = np.fft.rfftfreq(len(audio), 1/self.sr)
        cumulative_sum = np.cumsum(spectrum)
        total_energy = np.sum(spectrum)
        
        for i, cum_val in enumerate(cumulative_sum):
            if cum_val >= roll_percent * total_energy:
                return frequencies[i]
        return frequencies[-1]
    
    def compute_mfcc(self, audio, n_mfcc=13):
        """Compute MFCC features."""
        return librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
    
    def compute_mel_spectrogram(self, audio, n_mels=40):
        """Compute mel spectrogram."""
        return librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=n_mels)
    
    def compute_chromagram(self, audio):
        """Compute chromagram."""
        return librosa.feature.chroma_stft(y=audio, sr=self.sr)
    
    def detect_onset_times(self, audio, threshold=0.1):
        """Detect onset times in audio."""
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=self.sr, 
            hop_length=self.hop_length,
            threshold=threshold
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)
        return onset_times
    
    def extract_envelope(self, audio, lowcut=10):
        """Extract amplitude envelope using Hilbert transform."""
        analytic_signal = hilbert(audio)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Smooth envelope with lowpass filter
        nyquist = 0.5 * self.sr
        low = lowcut / nyquist
        sos = butter(2, low, btype='low', output='sos')
        smoothed_envelope = sosfilt(sos, amplitude_envelope)
        
        return smoothed_envelope
    
    def save_audio(self, audio, file_path, normalize=True):
        """Save audio to file."""
        if normalize:
            audio = self.normalize_audio(audio)
        sf.write(file_path, audio, self.sr)
        
    def resample_audio(self, audio, original_sr, target_sr):
        """Resample audio to target sample rate."""
        return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    
    def trim_silence(self, audio, top_db=30):
        """Trim silence from audio."""
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
