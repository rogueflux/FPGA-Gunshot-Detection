"""
Feature extraction utilities for gunshot detection.
"""
import numpy as np
import librosa
import librosa.display
from scipy import stats
from scipy.signal import spectrogram
from .audio_processor import AudioProcessor

class FeatureExtractor:
    """Extract various audio features for gunshot detection."""
    
    def __init__(self, sr=16000, frame_size=512, hop_length=256):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.audio_processor = AudioProcessor(sr, frame_size, hop_length)
        
    def extract_all_features(self, audio, include_temporal=True, 
                           include_spectral=True, include_mfcc=True):
        """Extract comprehensive set of features."""
        features = {}
        
        if include_temporal:
            features.update(self.extract_temporal_features(audio))
            
        if include_spectral:
            features.update(self.extract_spectral_features(audio))
            
        if include_mfcc:
            features.update(self.extract_mfcc_features(audio))
            
        return features
    
    def extract_temporal_features(self, audio):
        """Extract temporal domain features."""
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(audio)
        features['std'] = np.std(audio)
        features['max'] = np.max(audio)
        features['min'] = np.min(audio)
        features['range'] = features['max'] - features['min']
        
        # Energy-based features
        features['energy'] = self.audio_processor.compute_energy(audio)
        features['rms'] = self.audio_processor.compute_rms(audio)
        features['energy_db'] = 10 * np.log10(features['energy'] + 1e-10)
        
        # Zero-crossing rate
        features['zcr'] = self.audio_processor.compute_zero_crossing_rate(audio)
        
        # Statistical moments
        features['skewness'] = stats.skew(audio)
        features['kurtosis'] = stats.kurtosis(audio)
        
        # Envelope features
        envelope = self.audio_processor.extract_envelope(audio)
        features['envelope_mean'] = np.mean(envelope)
        features['envelope_std'] = np.std(envelope)
        features['envelope_peak'] = np.max(envelope)
        features['attack_time'] = self._compute_attack_time(envelope)
        features['decay_time'] = self._compute_decay_time(envelope)
        
        return features
    
    def extract_spectral_features(self, audio):
        """Extract spectral domain features."""
        features = {}
        
        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(audio))
        frequencies = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        # Basic spectral features
        features['spectral_centroid'] = self.audio_processor.compute_spectral_centroid(audio)
        features['spectral_rolloff'] = self.audio_processor.compute_spectral_rolloff(audio)
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sr
        ).mean()
        features['spectral_flatness'] = librosa.feature.spectral_flatness(y=audio).mean()
        features['spectral_contrast'] = librosa.feature.spectral_contrast(y=audio, sr=self.sr).mean()
        
        # Spectral statistics
        features['spectral_mean'] = np.mean(spectrum)
        features['spectral_std'] = np.std(spectrum)
        features['spectral_skewness'] = stats.skew(spectrum)
        features['spectral_kurtosis'] = stats.kurtosis(spectrum)
        
        # Peak features
        peak_freqs, peak_mags = self._find_spectral_peaks(spectrum, frequencies)
        features['peak_frequency'] = peak_freqs[0] if len(peak_freqs) > 0 else 0
        features['peak_magnitude'] = peak_mags[0] if len(peak_mags) > 0 else 0
        features['num_peaks'] = len(peak_freqs)
        
        # Band energy ratios
        features['low_freq_ratio'] = self._compute_band_energy_ratio(spectrum, frequencies, 0, 500)
        features['mid_freq_ratio'] = self._compute_band_energy_ratio(spectrum, frequencies, 500, 3000)
        features['high_freq_ratio'] = self._compute_band_energy_ratio(spectrum, frequencies, 3000, 8000)
        
        return features
    
    def extract_mfcc_features(self, audio, n_mfcc=13):
        """Extract MFCC features and statistics."""
        mfccs = self.audio_processor.compute_mfcc(audio, n_mfcc=n_mfcc)
        
        features = {}
        for i in range(n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_max'] = np.max(mfccs[i])
            features[f'mfcc_{i}_min'] = np.min(mfccs[i])
            
        # Delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        for i in range(n_mfcc):
            features[f'mfcc_delta_{i}_mean'] = np.mean(mfcc_delta[i])
            features[f'mfcc_delta2_{i}_mean'] = np.mean(mfcc_delta2[i])
            
        return features
    
    def extract_gunshot_specific_features(self, audio):
        """Extract features specifically tuned for gunshot detection."""
        features = {}
        
        # Temporal features specific to gunshots
        envelope = self.audio_processor.extract_envelope(audio)
        
        # Attack characteristics (gunshots have fast attack)
        attack_time = self._compute_attack_time(envelope)
        features['attack_time'] = attack_time
        features['is_fast_attack'] = 1 if attack_time < 0.005 else 0
        
        # Decay characteristics
        decay_time = self._compute_decay_time(envelope)
        features['decay_time'] = decay_time
        features['decay_slope'] = self._compute_decay_slope(envelope)
        
        # Spectral features specific to gunshots
        spectrum = np.abs(np.fft.rfft(audio))
        frequencies = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        # Gunshots often have energy in specific bands
        low_band = self._compute_band_energy(spectrum, frequencies, 50, 500)
        mid_band = self._compute_band_energy(spectrum, frequencies, 500, 3000)
        high_band = self._compute_band_energy(spectrum, frequencies, 3000, 10000)
        
        total_energy = low_band + mid_band + high_band
        if total_energy > 0:
            features['low_band_ratio'] = low_band / total_energy
            features['mid_band_ratio'] = mid_band / total_energy
            features['high_band_ratio'] = high_band / total_energy
        else:
            features['low_band_ratio'] = 0
            features['mid_band_ratio'] = 0
            features['high_band_ratio'] = 0
            
        # Transient detection
        features['transient_score'] = self._compute_transient_score(audio)
        
        return features
    
    def extract_frame_based_features(self, audio):
        """Extract features from audio frames."""
        frames = self.audio_processor.frame_audio(audio)
        windowed_frames = self.audio_processor.apply_window(frames)
        
        frame_features = []
        for frame in windowed_frames:
            frame_feat = self.extract_all_features(frame, include_mfcc=False)
            frame_features.append(frame_feat)
            
        # Aggregate frame features
        aggregated = {}
        for key in frame_features[0].keys():
            values = [f[key] for f in frame_features]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_max'] = np.max(values)
            aggregated[f'{key}_min'] = np.min(values)
            
        return aggregated
    
    def _compute_attack_time(self, envelope):
        """Compute attack time (time to reach 90% of peak)."""
        peak_idx = np.argmax(envelope)
        peak_value = envelope[peak_idx]
        
        if peak_value == 0:
            return 0
            
        threshold = 0.9 * peak_value
        for i in range(peak_idx, -1, -1):
            if envelope[i] < threshold:
                attack_samples = peak_idx - i
                return attack_samples / self.sr
                
        return peak_idx / self.sr
    
    def _compute_decay_time(self, envelope):
        """Compute decay time (time from peak to 10% of peak)."""
        peak_idx = np.argmax(envelope)
        peak_value = envelope[peak_idx]
        
        if peak_value == 0:
            return 0
            
        threshold = 0.1 * peak_value
        for i in range(peak_idx, len(envelope)):
            if envelope[i] < threshold:
                decay_samples = i - peak_idx
                return decay_samples / self.sr
                
        return (len(envelope) - peak_idx) / self.sr
    
    def _compute_decay_slope(self, envelope):
        """Compute slope of decay portion."""
        peak_idx = np.argmax(envelope)
        if peak_idx >= len(envelope) - 1:
            return 0
            
        decay_portion = envelope[peak_idx:]
        if len(decay_portion) < 2:
            return 0
            
        # Linear fit to decay portion
        x = np.arange(len(decay_portion))
        slope, _ = np.polyfit(x, decay_portion, 1)
        return slope
    
    def _find_spectral_peaks(self, spectrum, frequencies, prominence=0.1):
        """Find prominent peaks in spectrum."""
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(spectrum, prominence=prominence)
        peak_freqs = frequencies[peaks]
        peak_mags = spectrum[peaks]
        
        # Sort by magnitude
        if len(peaks) > 0:
            sorted_idx = np.argsort(peak_mags)[::-1]
            peak_freqs = peak_freqs[sorted_idx]
            peak_mags = peak_mags[sorted_idx]
            
        return peak_freqs[:5], peak_mags[:5]  # Return top 5 peaks
    
    def _compute_band_energy(self, spectrum, frequencies, low_freq, high_freq):
        """Compute energy in frequency band."""
        mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        return np.sum(spectrum[mask]**2)
    
    def _compute_band_energy_ratio(self, spectrum, frequencies, low_freq, high_freq):
        """Compute ratio of energy in frequency band to total energy."""
        band_energy = self._compute_band_energy(spectrum, frequencies, low_freq, high_freq)
        total_energy = np.sum(spectrum**2)
        
        if total_energy == 0:
            return 0
        return band_energy / total_energy
    
    def _compute_transient_score(self, audio):
        """Compute transient detection score."""
        # Using spectral flux as transient indicator
        frames = self.audio_processor.frame_audio(audio)
        windowed_frames = self.audio_processor.apply_window(frames)
        
        spectral_flux = []
        prev_spectrum = None
        
        for frame in windowed_frames:
            spectrum = np.abs(np.fft.rfft(frame))
            
            if prev_spectrum is not None:
                flux = np.sum((spectrum - prev_spectrum)**2)
                spectral_flux.append(flux)
                
            prev_spectrum = spectrum
            
        if len(spectral_flux) == 0:
            return 0
            
        return np.max(spectral_flux)
