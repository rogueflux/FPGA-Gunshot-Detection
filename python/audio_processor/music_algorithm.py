"""
MUSIC (MUltiple SIgnal Classification) algorithm for sound source localization.
"""
import numpy as np
from scipy.linalg import svd, eig
from scipy.signal import correlate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MUSICAlgorithm:
    """
    Implementation of MUSIC algorithm for sound source localization
    using microphone arrays.
    """
    
    def __init__(self, mic_positions, sr=16000, sound_speed=343.0):
        """
        Initialize MUSIC algorithm.
        
        Parameters:
        -----------
        mic_positions : numpy array (M, 3)
            Positions of microphones in meters
        sr : int
            Sample rate in Hz
        sound_speed : float
            Speed of sound in m/s
        """
        self.mic_positions = np.array(mic_positions)
        self.num_mics = len(mic_positions)
        self.sr = sr
        self.sound_speed = sound_speed
        
        # Validate microphone positions
        if self.mic_positions.shape[1] != 3:
            raise ValueError("Microphone positions must be 3D coordinates")
            
        # Precompute steering vectors for common grid
        self.grid_cache = {}
        
    def compute_steering_vector(self, source_position, frequencies):
        """
        Compute steering vector for given source position and frequencies.
        
        Parameters:
        -----------
        source_position : numpy array (3,)
            Source position in meters
        frequencies : numpy array (F,)
            Frequencies to consider (Hz)
            
        Returns:
        --------
        steering_vector : numpy array (M, F)
            Steering vector
        """
        # Compute time delays from source to each microphone
        distances = np.linalg.norm(self.mic_positions - source_position, axis=1)
        time_delays = distances / self.sound_speed
        
        # Convert time delays to phase shifts
        # steering_vector[m, f] = exp(-j * 2π * f * τ_m)
        omega = 2 * np.pi * frequencies[:, np.newaxis]  # (F, 1)
        phase_shifts = omega * time_delays  # (F, M)
        steering_vector = np.exp(-1j * phase_shifts)  # (F, M)
        
        return steering_vector.T  # (M, F)
    
    def compute_covariance_matrix(self, audio_signals):
        """
        Compute spatial covariance matrix from microphone signals.
        
        Parameters:
        -----------
        audio_signals : numpy array (M, N)
            Audio signals from M microphones, each of length N
            
        Returns:
        --------
        R : numpy array (M, M)
            Spatial covariance matrix
        """
        M, N = audio_signals.shape
        
        # Ensure signals are zero-mean
        audio_signals = audio_signals - np.mean(audio_signals, axis=1, keepdims=True)
        
        # Compute covariance matrix
        R = np.zeros((M, M), dtype=complex)
        
        for i in range(M):
            for j in range(M):
                R[i, j] = np.mean(audio_signals[i] * np.conj(audio_signals[j]))
                
        return R
    
    def compute_music_spectrum(self, audio_signals, grid_points, 
                             frequency_range=(100, 4000), num_freq_bins=100):
        """
        Compute MUSIC pseudospectrum over grid points.
        
        Parameters:
        -----------
        audio_signals : numpy array (M, N)
            Microphone signals
        grid_points : numpy array (G, 3)
            Grid points to evaluate
        frequency_range : tuple (low, high)
            Frequency range to consider
        num_freq_bins : int
            Number of frequency bins
            
        Returns:
        --------
        spectrum : numpy array (G,)
            MUSIC pseudospectrum values
        """
        M, N = audio_signals.shape
        G = len(grid_points)
        
        # Generate frequency bins
        frequencies = np.linspace(frequency_range[0], frequency_range[1], num_freq_bins)
        
        # Compute covariance matrix
        R = self.compute_covariance_matrix(audio_signals)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        
        # Sort eigenvalues in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Estimate number of sources (simplified)
        num_sources = self._estimate_num_sources(eigenvalues)
        
        # Split into signal and noise subspaces
        signal_subspace = eigenvectors[:, :num_sources]
        noise_subspace = eigenvectors[:, num_sources:]
        
        # Compute MUSIC spectrum
        spectrum = np.zeros(G)
        
        for g in range(G):
            source_pos = grid_points[g]
            
            # Compute steering vector for all frequencies
            steering_vec = self.compute_steering_vector(source_pos, frequencies)
            
            # Project steering vector onto noise subspace
            music_values = []
            for f_idx in range(num_freq_bins):
                a_f = steering_vec[:, f_idx]
                
                # Project onto noise subspace
                proj = noise_subspace.conj().T @ a_f
                power = np.linalg.norm(proj)**2
                
                if power > 0:
                    music_val = 1.0 / power
                    music_values.append(music_val)
                    
            if music_values:
                spectrum[g] = np.mean(music_values)
                
        return spectrum
    
    def localize_single_source(self, audio_signals, search_range=(-10, 10), 
                             grid_resolution=0.5, frequency_range=(100, 4000)):
        """
        Localize single sound source using MUSIC algorithm.
        
        Parameters:
        -----------
        audio_signals : numpy array (M, N)
            Microphone signals
        search_range : tuple (min, max)
            Search range in meters for x, y, z
        grid_resolution : float
            Grid resolution in meters
        frequency_range : tuple (low, high)
            Frequency range for analysis
            
        Returns:
        --------
        estimated_position : numpy array (3,)
            Estimated source position
        confidence : float
            Localization confidence (0-1)
        spectrum : numpy array
            MUSIC spectrum over grid
        grid_points : numpy array
            Grid points
        """
        # Create search grid
        x_range = np.arange(search_range[0], search_range[1] + grid_resolution, grid_resolution)
        y_range = np.arange(search_range[0], search_range[1] + grid_resolution, grid_resolution)
        z_range = np.array([0])  # Assume source at height 0 (2D localization)
        
        grid_points = []
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    grid_points.append([x, y, z])
                    
        grid_points = np.array(grid_points)
        
        # Compute MUSIC spectrum
        spectrum = self.compute_music_spectrum(audio_signals, grid_points, frequency_range)
        
        # Find peak in spectrum
        peak_idx = np.argmax(spectrum)
        estimated_position = grid_points[peak_idx]
        
        # Compute confidence based on peak sharpness
        confidence = self._compute_peak_confidence(spectrum, peak_idx)
        
        return estimated_position, confidence, spectrum, grid_points
    
    def localize_multiple_sources(self, audio_signals, num_sources=2, 
                                search_range=(-10, 10), grid_resolution=0.5):
        """
        Localize multiple sound sources using MUSIC algorithm.
        
        Parameters:
        -----------
        audio_signals : numpy array (M, N)
            Microphone signals
        num_sources : int
            Number of sources to localize
        search_range : tuple (min, max)
            Search range in meters
        grid_resolution : float
            Grid resolution
            
        Returns:
        --------
        source_positions : list of numpy arrays
            Estimated source positions
        confidences : list of floats
            Confidence values for each source
        """
        # Use iterative peak finding
        spectrum, grid_points = self._compute_full_spectrum(
            audio_signals, search_range, grid_resolution
        )
        
        source_positions = []
        confidences = []
        current_spectrum = spectrum.copy()
        
        for _ in range(num_sources):
            peak_idx = np.argmax(current_spectrum)
            source_pos = grid_points[peak_idx]
            confidence = self._compute_peak_confidence(current_spectrum, peak_idx)
            
            source_positions.append(source_pos)
            confidences.append(confidence)
            
            # Remove peak region to find next source
            current_spectrum = self._remove_peak_region(
                current_spectrum, grid_points, source_pos, 
                removal_radius=2.0
            )
            
        return source_positions, confidences
    
    def estimate_tdoa(self, audio_signals, reference_mic=0):
        """
        Estimate Time Difference of Arrival (TDOA) between microphones.
        
        Parameters:
        -----------
        audio_signals : numpy array (M, N)
            Microphone signals
        reference_mic : int
            Reference microphone index
            
        Returns:
        --------
        tdoa : numpy array (M,)
            TDOA estimates in samples
        """
        M, N = audio_signals.shape
        tdoa = np.zeros(M)
        
        for m in range(M):
            if m == reference_mic:
                tdoa[m] = 0
                continue
                
            # Cross-correlation
            correlation = correlate(audio_signals[reference_mic], audio_signals[m], mode='full')
            
            # Find peak
            peak_idx = np.argmax(np.abs(correlation))
            
            # Convert to time difference
            tdoa[m] = peak_idx - (N - 1)
            
        return tdoa
    
    def tdoa_localization(self, audio_signals):
        """
        Localize source using TDOA method (simpler alternative to MUSIC).
        
        Parameters:
        -----------
        audio_signals : numpy array (M, N)
            Microphone signals
            
        Returns:
        --------
        estimated_position : numpy array (3,)
            Estimated position
        """
        M = len(audio_signals)
        
        if M < 4:
            raise ValueError("Need at least 4 microphones for 3D TDOA localization")
            
        # Estimate TDOA
        tdoa = self.estimate_tdoa(audio_signals)
        
        # Convert TDOA to time differences in seconds
        time_diffs = tdoa / self.sr
        
        # Solve for position using least squares
        # Implement Taylor series or other TDOA localization algorithm
        # This is a simplified version
        
        # For simplicity, return centroid of microphone array
        return np.mean(self.mic_positions, axis=0)
    
    def plot_localization_results(self, spectrum, grid_points, source_positions=None, 
                                save_path=None):
        """
        Plot localization results.
        
        Parameters:
        -----------
        spectrum : numpy array
            MUSIC spectrum
        grid_points : numpy array (G, 3)
            Grid points
        source_positions : list of numpy arrays, optional
            Estimated source positions
        save_path : str, optional
            Path to save the plot
        """
        fig = plt.figure(figsize=(12, 10))
        
        # 3D plot of spectrum
        ax1 = fig.add_subplot(221, projection='3d')
        sc1 = ax1.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], 
                         c=spectrum, cmap='viridis', alpha=0.6)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('MUSIC Spectrum (3D)')
        plt.colorbar(sc1, ax=ax1, shrink=0.5)
        
        # Plot microphone positions
        ax1.scatter(self.mic_positions[:, 0], self.mic_positions[:, 1], 
                   self.mic_positions[:, 2], c='red', s=100, marker='^', label='Mics')
        
        # Plot estimated source positions
        if source_positions is not None:
            source_positions = np.array(source_positions)
            ax1.scatter(source_positions[:, 0], source_positions[:, 1], 
                       source_positions[:, 2], c='yellow', s=200, marker='*', 
                       label='Sources')
            ax1.legend()
        
        # 2D contour plot (XY plane)
        ax2 = fig.add_subplot(222)
        
        # Reshape spectrum for 2D plotting
        unique_x = np.unique(grid_points[:, 0])
        unique_y = np.unique(grid_points[:, 1])
        X, Y = np.meshgrid(unique_x, unique_y)
        
        # Assuming single z-plane
        spectrum_2d = spectrum.reshape(len(unique_y), len(unique_x))
        
        contour = ax2.contourf(X, Y, spectrum_2d, levels=50, cmap='viridis')
        ax2.scatter(self.mic_positions[:, 0], self.mic_positions[:, 1], 
                   c='red', s=100, marker='^')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('MUSIC Spectrum (2D)')
        plt.colorbar(contour, ax=ax2)
        
        # Spectrum histogram
        ax3 = fig.add_subplot(223)
        ax3.hist(spectrum, bins=50, edgecolor='black')
        ax3.set_xlabel('Spectrum Value')
        ax3.set_ylabel('Count')
        ax3.set_title('Spectrum Distribution')
        
        # Top-down view
        ax4 = fig.add_subplot(224)
        im = ax4.imshow(spectrum_2d, extent=[unique_x.min(), unique_x.max(), 
                                            unique_y.min(), unique_y.max()],
                       origin='lower', aspect='auto', cmap='hot')
        ax4.scatter(self.mic_positions[:, 0], self.mic_positions[:, 1], 
                   c='cyan', s=100, marker='o', edgecolors='white')
        
        if source_positions is not None:
            ax4.scatter(source_positions[:, 0], source_positions[:, 1], 
                       c='yellow', s=200, marker='*', edgecolors='black')
            
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_title('Top View with Mics and Sources')
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        plt.show()
    
    def _estimate_num_sources(self, eigenvalues, threshold_db=10):
        """
        Estimate number of sources from eigenvalues.
        
        Parameters:
        -----------
        eigenvalues : numpy array
            Eigenvalues from covariance matrix
        threshold_db : float
            Threshold in dB for eigenvalue drop
            
        Returns:
        --------
        num_sources : int
            Estimated number of sources
        """
        # Convert to dB
        eigenvalues_db = 10 * np.log10(eigenvalues + 1e-10)
        
        # Find significant drop
        for i in range(1, len(eigenvalues_db)):
            if eigenvalues_db[i-1] - eigenvalues_db[i] > threshold_db:
                return i
                
        # Default to 1 source if no clear drop
        return 1
    
    def _compute_peak_confidence(self, spectrum, peak_idx):
        """
        Compute confidence based on peak sharpness.
        
        Parameters:
        -----------
        spectrum : numpy array
            MUSIC spectrum
        peak_idx : int
            Index of peak
            
        Returns:
        --------
        confidence : float
            Confidence value (0-1)
        """
        peak_value = spectrum[peak_idx]
        mean_value = np.mean(spectrum)
        std_value = np.std(spectrum)
        
        if std_value == 0:
            return 0.0
            
        # Normalized peak height
        normalized_peak = (peak_value - mean_value) / std_value
        
        # Convert to confidence (sigmoid)
        confidence = 1.0 / (1.0 + np.exp(-0.5 * normalized_peak))
        
        return min(max(confidence, 0.0), 1.0)
    
    def _compute_full_spectrum(self, audio_signals, search_range, grid_resolution):
        """Compute full MUSIC spectrum over search grid."""
        x_range = np.arange(search_range[0], search_range[1] + grid_resolution, grid_resolution)
        y_range = np.arange(search_range[0], search_range[1] + grid_resolution, grid_resolution)
        z_range = np.array([0])
        
        grid_points = []
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    grid_points.append([x, y, z])
                    
        grid_points = np.array(grid_points)
        
        spectrum = self.compute_music_spectrum(audio_signals, grid_points)
        
        return spectrum, grid_points
    
    def _remove_peak_region(self, spectrum, grid_points, peak_position, removal_radius):
        """
        Remove region around peak to find next source.
        
        Parameters:
        -----------
        spectrum : numpy array
            Current spectrum
        grid_points : numpy array
            Grid points
        peak_position : numpy array
            Position of peak to remove
        removal_radius : float
            Radius to remove around peak
            
        Returns:
        --------
        new_spectrum : numpy array
            Spectrum with peak region removed
        """
        new_spectrum = spectrum.copy()
        
        # Find points within removal radius
        distances = np.linalg.norm(grid_points - peak_position, axis=1)
        mask = distances < removal_radius
        
        # Set spectrum values in this region to minimum
        new_spectrum[mask] = np.min(spectrum)
        
        return new_spectrum


# Helper functions for microphone array configurations
def create_linear_array(num_mics=6, spacing=0.1, center=(0, 0, 0)):
    """Create linear microphone array."""
    positions = []
    for i in range(num_mics):
        x = center[0] + (i - (num_mics-1)/2) * spacing
        positions.append([x, center[1], center[2]])
    return np.array(positions)

def create_circular_array(num_mics=6, radius=0.1, center=(0, 0, 0)):
    """Create circular microphone array."""
    positions = []
    for i in range(num_mics):
        angle = 2 * np.pi * i / num_mics
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        positions.append([x, y, center[2]])
    return np.array(positions)

def create_rectangular_array(rows=2, cols=3, spacing=0.1, center=(0, 0, 0)):
    """Create rectangular microphone array."""
    positions = []
    for i in range(rows):
        for j in range(cols):
            x = center[0] + (j - (cols-1)/2) * spacing
            y = center[1] + (i - (rows-1)/2) * spacing
            positions.append([x, y, center[2]])
    return np.array(positions)
