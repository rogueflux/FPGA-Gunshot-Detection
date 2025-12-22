"""
Inference script for gunshot detection model.
"""
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from .attention_model import AudioMobileNet1D
from ..audio_processor.feature_extractor import FeatureExtractor

class GunshotDetector:
    """Gunshot detection inference class."""
    
    def __init__(self, model_path=None, device=None, threshold=0.65):
        """
        Initialize gunshot detector.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to trained model
        device : str, optional
            Device to run inference on
        threshold : float
            Detection threshold (0-1)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Load model
        self.model = AudioMobileNet1D(num_classes=2, input_channels=1)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("Warning: No model loaded. Using random weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(sr=16000)
        
        # Detection history
        self.detection_history = []
        
    def load_model(self, model_path):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        print(f"Model loaded from {model_path}")
        
    def preprocess_audio(self, audio, sr=16000, target_sr=16000):
        """
        Preprocess audio for inference.
        
        Parameters:
        -----------
        audio : numpy array or str
            Audio signal or path to audio file
        sr : int
            Original sample rate
        target_sr : int
            Target sample rate
            
        Returns:
        --------
        processed_audio : numpy array
            Preprocessed audio
        """
        if isinstance(audio, str):
            # Load audio file
            audio, sr = librosa.load(audio, sr=None, mono=True)
            
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Ensure minimum length
        min_length = 16000  # 1 second at 16kHz
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)))
            
        return audio
    
    def extract_features(self, audio):
        """
        Extract features from audio.
        
        Parameters:
        -----------
        audio : numpy array
            Audio signal
            
        Returns:
        --------
        features : torch tensor
            Extracted features
        """
        # Extract MFCC features (13 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13, n_fft=512, hop_length=256)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=16000, n_fft=512, hop_length=256)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=16000, n_fft=512, hop_length=256)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=16000, n_fft=512, hop_length=256)
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=512, hop_length=256)
        
        # Stack features
        features = np.vstack([
            mfcc,
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            zcr
        ])
        
        # Ensure correct shape
        if features.shape[1] < 63:  # Model expects 63 time frames
            features = np.pad(features, ((0, 0), (0, 63 - features.shape[1])))
        else:
            features = features[:, :63]
            
        # Add channel dimension
        features = features[np.newaxis, ...]  # (1, num_features, time)
        
        return torch.FloatTensor(features)
    
    def predict(self, audio, return_confidence=False):
        """
        Predict if audio contains gunshot.
        
        Parameters:
        -----------
        audio : numpy array or str
            Audio signal or path to audio file
        return_confidence : bool
            Whether to return confidence score
            
        Returns:
        --------
        is_gunshot : bool or tuple
            True if gunshot detected, or (is_gunshot, confidence)
        """
        # Preprocess audio
        processed_audio = self.preprocess_audio(audio)
        
        # Extract features
        features = self.extract_features(processed_audio)
        features = features.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            gunshot_prob = probabilities[0, 1].item()  # Probability of gunshot class
            
        is_gunshot = gunshot_prob > self.threshold
        
        # Record detection
        detection_record = {
            'timestamp': datetime.now().isoformat(),
            'is_gunshot': is_gunshot,
            'confidence': gunshot_prob,
            'threshold': self.threshold
        }
        self.detection_history.append(detection_record)
        
        if return_confidence:
            return is_gunshot, gunshot_prob
        return is_gunshot
    
    def predict_batch(self, audio_list):
        """
        Predict for batch of audio files.
        
        Parameters:
        -----------
        audio_list : list
            List of audio signals or file paths
            
        Returns:
        --------
        results : list
            List of (is_gunshot, confidence) tuples
        """
        results = []
        
        for audio in audio_list:
            try:
                is_gunshot, confidence = self.predict(audio, return_confidence=True)
                results.append((is_gunshot, confidence))
            except Exception as e:
                print(f"Error processing audio: {e}")
                results.append((False, 0.0))
                
        return results
    
    def stream_predict(self, audio_chunk, sr=16000):
        """
        Real-time prediction for audio stream.
        
        Parameters:
        -----------
        audio_chunk : numpy array
            Audio chunk from stream
        sr : int
            Sample rate
            
        Returns:
        --------
        is_gunshot : bool
            True if gunshot detected
        confidence : float
            Detection confidence
        """
        return self.predict(audio_chunk, return_confidence=True)
    
    def analyze_audio_file(self, audio_path, plot=False):
        """
        Comprehensive analysis of audio file.
        
        Parameters:
        -----------
        audio_path : str
            Path to audio file
        plot : bool
            Whether to plot analysis results
            
        Returns:
        --------
        analysis_result : dict
            Analysis results
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract features
        features = self.extract_features(audio)
        
        # Run inference
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            gunshot_prob = probabilities[0, 1].item()
            
        # Additional feature analysis
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        
        # Compute statistics
        analysis_result = {
            'file_path': audio_path,
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'gunshot_probability': gunshot_prob,
            'is_gunshot': gunshot_prob > self.threshold,
            'mfcc_mean': mfcc.mean(axis=1).tolist(),
            'mfcc_std': mfcc.std(axis=1).tolist(),
            'spectral_centroid_mean': float(spectral_centroid.mean()),
            'spectral_rolloff_mean': float(spectral_rolloff.mean()),
            'rms': float(librosa.feature.rms(y=audio).mean()),
            'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(audio).mean())
        }
        
        # Plot if requested
        if plot:
            self._plot_analysis(audio, sr, analysis_result)
            
        return analysis_result
    
    def _plot_analysis(self, audio, sr, analysis_result):
        """Plot audio analysis results."""
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        # Waveform
        axes[0, 0].plot(np.arange(len(audio)) / sr, audio)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Waveform')
        axes[0, 0].grid(True)
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[0, 1])
        axes[0, 1].set_title('Spectrogram')
        plt.colorbar(img, ax=axes[0, 1])
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[1, 0])
        axes[1, 0].set_title('MFCC')
        plt.colorbar(img, ax=axes[1, 0])
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        axes[1, 1].plot(spectral_centroid.T)
        axes[1, 1].set_xlabel('Time (frames)')
        axes[1, 1].set_ylabel('Frequency (Hz)')
        axes[1, 1].set_title('Spectral Centroid')
        axes[1, 1].grid(True)
        
        # Detection probability
        axes[2, 0].bar(['Non-Gunshot', 'Gunshot'], 
                      [1 - analysis_result['gunshot_probability'], 
                       analysis_result['gunshot_probability']],
                      color=['green', 'red'])
        axes[2, 0].set_ylim([0, 1])
        axes[2, 0].set_ylabel('Probability')
        axes[2, 0].set_title(f'Detection: {analysis_result["gunshot_probability"]:.2%}')
        axes[2, 0].grid(True, axis='y')
        
        # Feature summary
        features_text = f"""
        Duration: {analysis_result['duration']:.2f}s
        RMS: {analysis_result['rms']:.4f}
        Zero-Crossing Rate: {analysis_result['zero_crossing_rate']:.4f}
        Spectral Centroid: {analysis_result['spectral_centroid_mean']:.1f} Hz
        """
        axes[2, 1].text(0.1, 0.5, features_text, fontsize=10, 
                       verticalalignment='center')
        axes[2, 1].set_title('Feature Summary')
        axes[2, 1].axis('off')
        
        # Add detection result
        detection_text = "GUNSHOT DETECTED!" if analysis_result['is_gunshot'] else "No gunshot detected"
        detection_color = 'red' if analysis_result['is_gunshot'] else 'green'
        
        fig.suptitle(f'{detection_text}\nConfidence: {analysis_result["gunshot_probability"]:.2%}', 
                    fontsize=14, color=detection_color, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    def save_detection_history(self, filepath='detection_history.json'):
        """Save detection history to file."""
        with open(filepath, 'w') as f:
            json.dump(self.detection_history, f, indent=2)
        print(f"Detection history saved to {filepath}")
        
    def get_statistics(self):
        """Get detection statistics."""
        if not self.detection_history:
            return None
            
        detections = [d for d in self.detection_history if d['is_gunshot']]
        total = len(self.detection_history)
        gunshots = len(detections)
        
        if gunshots > 0:
            avg_confidence = np.mean([d['confidence'] for d in detections])
        else:
            avg_confidence = 0
            
        return {
            'total_predictions': total,
            'gunshots_detected': gunshots,
            'non_gunshots': total - gunshots,
            'detection_rate': gunshots / total if total > 0 else 0,
            'average_confidence': avg_confidence
        }


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gunshot Detection Inference')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to audio file or directory')
    parser.add_argument('--threshold', type=float, default=0.65,
                       help='Detection threshold (0-1)')
    parser.add_argument('--plot', action='store_true',
                       help='Plot analysis results')
    parser.add_argument('--batch', action='store_true',
                       help='Process batch of files')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = GunshotDetector(model_path=args.model, threshold=args.threshold)
    
    if os.path.isdir(args.audio):
        # Process directory
        audio_files = [os.path.join(args.audio, f) for f in os.listdir(args.audio) 
                      if f.endswith(('.wav', '.mp3', '.flac'))]
        
        print(f"Processing {len(audio_files)} audio files...")
        results = detector.predict_batch(audio_files)
        
        # Print results
        for audio_file, (is_gunshot, confidence) in zip(audio_files, results):
            status = "✅ GUNSHOT" if is_gunshot else "❌ No gunshot"
            print(f"{audio_file}: {status} (Confidence: {confidence:.2%})")
            
    else:
        # Process single file
        result = detector.analyze_audio_file(args.audio, plot=args.plot)
        
        print("\n" + "="*50)
        print(f"Analysis Results for: {args.audio}")
        print("="*50)
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Gunshot Probability: {result['gunshot_probability']:.2%}")
        print(f"Detection: {'✅ GUNSHOT DETECTED' if result['is_gunshot'] else '❌ No gunshot'}")
        print(f"RMS: {result['rms']:.4f}")
        print(f"Spectral Centroid: {result['spectral_centroid_mean']:.1f} Hz")
        print("="*50)
        
    # Print statistics
    stats = detector.get_statistics()
    if stats:
        print(f"\nStatistics:")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Gunshots detected: {stats['gunshots_detected']}")
        print(f"  Detection rate: {stats['detection_rate']:.2%}")
        print(f"  Average confidence: {stats['average_confidence']:.2%}")


if __name__ == "__main__":
    main()
