"""
Data loading utilities for gunshot detection dataset.
"""
import os
import numpy as np
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GunshotDataset(Dataset):
    """Dataset class for gunshot detection."""
    
    def __init__(self, data_dir, transform=None, target_sr=16000, 
                 max_duration=2.0, mode='train'):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing audio data
            Expected structure:
                data_dir/
                    gunshots/
                        *.wav
                    non_gunshots/
                        *.wav
        transform : callable, optional
            Optional transform to apply to samples
        target_sr : int
            Target sample rate
        max_duration : float
            Maximum duration in seconds (longer files will be truncated)
        mode : str
            Mode: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.mode = mode
        
        # Load audio files and labels
        self.audio_files = []
        self.labels = []
        self.file_info = []
        
        self._load_data()
        
        print(f"Dataset loaded: {len(self.audio_files)} samples")
        print(f"  Gunshots: {sum(self.labels)}")
        print(f"  Non-gunshots: {len(self.labels) - sum(self.labels)}")
        
    def _load_data(self):
        """Load audio files and labels."""
        gunshot_dir = os.path.join(self.data_dir, 'gunshots')
        non_gunshot_dir = os.path.join(self.data_dir, 'non_gunshots')
        background_dir = os.path.join(self.data_dir, 'background')
        
        # Load gunshot samples
        if os.path.exists(gunshot_dir):
            for fname in os.listdir(gunshot_dir):
                if fname.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(gunshot_dir, fname)
                    self.audio_files.append(file_path)
                    self.labels.append(1)  # Gunshot label
                    self.file_info.append({
                        'type': 'gunshot',
                        'filename': fname,
                        'path': file_path
                    })
        
        # Load non-gunshot samples
        if os.path.exists(non_gunshot_dir):
            for fname in os.listdir(non_gunshot_dir):
                if fname.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(non_gunshot_dir, fname)
                    self.audio_files.append(file_path)
                    self.labels.append(0)  # Non-gunshot label
                    self.file_info.append({
                        'type': 'non_gunshot',
                        'filename': fname,
                        'path': file_path
                    })
        
        # Load background noise samples
        if os.path.exists(background_dir):
            for fname in os.listdir(background_dir):
                if fname.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(background_dir, fname)
                    self.audio_files.append(file_path)
                    self.labels.append(0)  # Non-gunshot label
                    self.file_info.append({
                        'type': 'background',
                        'filename': fname,
                        'path': file_path
                    })
        
        # If no data found, try alternative structure
        if len(self.audio_files) == 0:
            self._load_alternative_structure()
            
    def _load_alternative_structure(self):
        """Try alternative directory structure."""
        # Look for audio files in data_dir directly
        for fname in os.listdir(self.data_dir):
            if fname.endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(self.data_dir, fname)
                self.audio_files.append(file_path)
                
                # Infer label from filename
                if 'gun' in fname.lower() or 'shot' in fname.lower():
                    self.labels.append(1)
                    file_type = 'gunshot'
                else:
                    self.labels.append(0)
                    file_type = 'non_gunshot'
                    
                self.file_info.append({
                    'type': file_type,
                    'filename': fname,
                    'path': file_path
                })
                
        print(f"Loaded {len(self.audio_files)} files from alternative structure")
        
    def _load_audio(self, file_path):
        """Load and preprocess audio file."""
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=30)
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Truncate or pad to fixed duration
            target_samples = int(self.max_duration * self.target_sr)
            
            if len(audio) > target_samples:
                # Take middle portion
                start = (len(audio) - target_samples) // 2
                audio = audio[start:start + target_samples]
            elif len(audio) < target_samples:
                # Pad with zeros
                pad_len = target_samples - len(audio)
                audio = np.pad(audio, (0, pad_len), mode='constant')
                
            return audio
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return silent audio
            return np.zeros(int(self.max_duration * self.target_sr))
        
    def _extract_features(self, audio):
        """Extract features from audio."""
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.target_sr, 
            n_mfcc=13,
            n_fft=512,
            hop_length=256
        )
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.target_sr, n_fft=512, hop_length=256
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.target_sr, n_fft=512, hop_length=256
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.target_sr, n_fft=512, hop_length=256
        )
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=512, hop_length=256
        )
        
        # Stack features
        features = np.vstack([
            mfcc,
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            zcr
        ])
        
        # Ensure correct shape (18 features x 63 time frames)
        if features.shape[1] < 63:
            features = np.pad(features, ((0, 0), (0, 63 - features.shape[1])))
        else:
            features = features[:, :63]
            
        return features
        
    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.audio_files)
        
    def __getitem__(self, idx):
        """Get sample by index."""
        # Load audio
        audio = self._load_audio(self.audio_files[idx])
        
        # Extract features
        features = self._extract_features(audio)
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        
        # Get label
        label = torch.LongTensor([self.labels[idx]])[0]
        
        # Apply transformations if any
        if self.transform:
            features = self.transform(features)
            
        return features, label
        
    def get_class_weights(self):
        """Compute class weights for imbalanced dataset."""
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        
        weights = total_samples / (len(class_counts) * class_counts)
        weights = weights / weights.sum() * len(class_counts)
        
        return torch.FloatTensor(weights)
        
    def get_file_info(self, idx):
        """Get file information for sample."""
        return self.file_info[idx]
        
    def analyze_dataset(self):
        """Analyze dataset statistics."""
        print("\n" + "="*50)
        print("Dataset Analysis")
        print("="*50)
        
        durations = []
        sample_rates = []
        
        for file_path in tqdm(self.audio_files, desc="Analyzing files"):
            try:
                info = sf.info(file_path)
                durations.append(info.duration)
                sample_rates.append(info.samplerate)
            except:
                pass
                
        if durations:
            print(f"Number of files: {len(self.audio_files)}")
            print(f"Total duration: {sum(durations):.2f} seconds")
            print(f"Average duration: {np.mean(durations):.2f} seconds")
            print(f"Min duration: {np.min(durations):.2f} seconds")
            print(f"Max duration: {np.max(durations):.2f} seconds")
            print(f"Sample rates: {set(sample_rates)}")
            
        # Class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"\nClass distribution:")
        for label, count in zip(unique, counts):
            class_name = "Gunshot" if label == 1 else "Non-gunshot"
            print(f"  {class_name}: {count} samples ({count/len(self.labels):.1%})")
            
        return {
            'total_files': len(self.audio_files),
            'total_duration': sum(durations) if durations else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'class_distribution': dict(zip(unique, counts))
        }


class AudioAugmentation:
    """Audio data augmentation class."""
    
    def __init__(self, sr=16000):
        self.sr = sr
        
    def add_noise(self, audio, noise_level=0.005):
        """Add Gaussian noise to audio."""
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
        
    def time_shift(self, audio, shift_max=0.2):
        """Shift audio in time."""
        shift = np.random.randint(int(self.sr * shift_max))
        if shift > 0:
            audio = np.roll(audio, shift)
            audio[:shift] = 0
        return audio
        
    def time_stretch(self, audio, rate=1.0):
        """Time stretch audio."""
        if rate == 1.0:
            return audio
        return librosa.effects.time_stretch(audio, rate=rate)
        
    def pitch_shift(self, audio, n_steps=0):
        """Pitch shift audio."""
        if n_steps == 0:
            return audio
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        
    def apply_random_gain(self, audio, min_gain=0.5, max_gain=1.5):
        """Apply random gain."""
        gain = np.random.uniform(min_gain, max_gain)
        return audio * gain
        
    def apply_augmentation(self, audio, augmentation_list=None):
        """Apply random augmentations."""
        if augmentation_list is None:
            augmentation_list = ['noise', 'shift', 'gain']
            
        augmented = audio.copy()
        
        for aug in augmentation_list:
            if aug == 'noise' and np.random.random() > 0.5:
                augmented = self.add_noise(augmented)
            elif aug == 'shift' and np.random.random() > 0.5:
                augmented = self.time_shift(augmented)
            elif aug == 'stretch' and np.random.random() > 0.3:
                rate = np.random.uniform(0.8, 1.2)
                augmented = self.time_stretch(augmented, rate)
            elif aug == 'pitch' and np.random.random() > 0.3:
                n_steps = np.random.uniform(-2, 2)
                augmented = self.pitch_shift(augmented, n_steps)
            elif aug == 'gain' and np.random.random() > 0.5:
                augmented = self.apply_random_gain(augmented)
                
        return augmented


def get_data_loaders(data_dir, batch_size=32, train_ratio=0.7, 
                    val_ratio=0.15, test_ratio=0.15, augment=True,
                    num_workers=4, pin_memory=True):
    """
    Create train, validation, and test data loaders.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data
    batch_size : int
        Batch size
    train_ratio : float
        Ratio of training data
    val_ratio : float
        Ratio of validation data
    test_ratio : float
        Ratio of test data
    augment : bool
        Whether to augment training data
    num_workers : int
        Number of data loading workers
    pin_memory : bool
        Whether to pin memory for GPU training
        
    Returns:
    --------
    train_loader, val_loader, test_loader : DataLoader objects
    """
    # Create full dataset
    full_dataset = GunshotDataset(data_dir, mode='full')
    
    # Split indices
    n_total = len(full_dataset)
    indices = list(range(n_total))
    
    # Stratified split
    labels = full_dataset.labels
    train_idx, temp_idx = train_test_split(
        indices, 
        train_size=train_ratio, 
        stratify=labels,
        random_state=42
    )
    
    # Split remaining into validation and test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_test_ratio,
        stratify=[labels[i] for i in temp_idx],
        random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(train_idx)} samples ({len(train_idx)/n_total:.1%})")
    print(f"  Validation: {len(val_idx)} samples ({len(val_idx)/n_total:.1%})")
    print(f"  Test: {len(test_idx)} samples ({len(test_idx)/n_total:.1%})")
    
    # Create subset datasets
    class SubsetDataset(Dataset):
        def __init__(self, dataset, indices, transform=None, augment=False):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
            self.augment = augment
            self.augmentor = AudioAugmentation(sr=dataset.target_sr) if augment else None
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            dataset_idx = self.indices[idx]
            features, label = self.dataset[dataset_idx]
            
            if self.augment and self.augmentor and label == 1:  # Only augment gunshots
                # Convert features back to audio for augmentation
                audio = self._features_to_audio(features.numpy())
                audio = self.augmentor.apply_augmentation(audio)
                features = torch.FloatTensor(self.dataset._extract_features(audio))
                
            if self.transform:
                features = self.transform(features)
                
            return features, label
            
        def _features_to_audio(self, features):
            """Convert features back to audio (simplified)."""
            # This is a simplified version - in practice, you'd use Griffin-Lim or similar
            # For now, just create a dummy audio signal
            return np.random.randn(32000) * 0.1
    
    # Create datasets
    train_dataset = SubsetDataset(
        full_dataset, train_idx, 
        augment=augment and train_ratio > 0
    )
    val_dataset = SubsetDataset(full_dataset, val_idx)
    test_dataset = SubsetDataset(full_dataset, test_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def create_synthetic_dataset(output_dir, num_samples=1000):
    """
    Create synthetic gunshot dataset for testing.
    
    Parameters:
    -----------
    output_dir : str
        Output directory
    num_samples : int
        Number of samples to generate
    """
    import os
    os.makedirs(os.path.join(output_dir, 'gunshots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'non_gunshots'), exist_ok=True)
    
    sr = 16000
    duration = 2.0
    
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    for i in range(num_samples):
        # Generate gunshot samples
        if i < num_samples // 2:
            # Gunshot: fast attack, exponential decay
            t = np.linspace(0, duration, int(sr * duration))
            
            # Create impulse
            attack_time = np.random.uniform(0.001, 0.01)
            decay_time = np.random.uniform(0.1, 0.5)
            
            attack_samples = int(attack_time * sr)
            decay_samples = int(decay_time * sr)
            
            audio = np.zeros(len(t))
            
            # Attack portion
            attack = np.linspace(0, 1, attack_samples)
            
            # Decay portion
            decay = np.exp(-np.linspace(0, 5, decay_samples))
            
            # Combine
            impulse = np.concatenate([attack, decay])
            
            # Place impulse randomly in audio
            start_idx = np.random.randint(0, len(audio) - len(impulse))
            end_idx = start_idx + len(impulse)
            
            if end_idx <= len(audio):
                audio[start_idx:end_idx] = impulse[:end_idx-start_idx]
                
            # Add some high frequency content
            freq = np.random.uniform(1000, 5000)
            audio += 0.1 * np.sin(2 * np.pi * freq * t)
            
            # Add noise
            audio += np.random.normal(0, 0.01, len(audio))
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Save
            file_path = os.path.join(output_dir, 'gunshots', f'gunshot_{i:04d}.wav')
            sf.write(file_path, audio, sr)
            
        else:
            # Non-gunshot: random noise or sine waves
            t = np.linspace(0, duration, int(sr * duration))
            
            # Random choice of noise type
            noise_type = np.random.choice(['white', 'pink', 'sine', 'silent'])
            
            if noise_type == 'white':
                audio = np.random.normal(0, 0.1, len(t))
            elif noise_type == 'pink':
                # Simplified pink noise
                audio = np.random.normal(0, 0.1, len(t))
                b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
                a = [1, -2.494956002, 2.017265875, -0.522189400]
                audio = scipy.signal.lfilter(b, a, audio)
            elif noise_type == 'sine':
                freq = np.random.uniform(100, 1000)
                audio = 0.5 * np.sin(2 * np.pi * freq * t)
            else:  # silent
                audio = np.zeros(len(t))
                
            # Add occasional clicks
            if np.random.random() < 0.3:
                click_pos = np.random.randint(0, len(audio))
                click_duration = np.random.randint(10, 100)
                audio[click_pos:click_pos+click_duration] += 0.5
                
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Save
            file_path = os.path.join(output_dir, 'non_gunshots', f'non_gunshot_{i:04d}.wav')
            sf.write(file_path, audio, sr)
            
    print(f"Synthetic dataset created in {output_dir}")


def main():
    """Test data loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test data loading')
    parser.add_argument('--data_dir', type=str, default='data/audio_samples',
                       help='Data directory')
    parser.add_argument('--create_synthetic', action='store_true',
                       help='Create synthetic dataset')
    
    args = parser.parse_args()
    
    if args.create_synthetic:
        create_synthetic_dataset(args.data_dir)
        
    # Test dataset loading
    dataset = GunshotDataset(args.data_dir)
    dataset.analyze_dataset()
    
    # Test data loader
    train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, batch_size=4)
    
    # Show a batch
    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels: {labels.tolist()}")
        
        if batch_idx >= 2:  # Show only first 3 batches
            break


if __name__ == "__main__":
    main()
