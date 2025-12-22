import sounddevice as sd
import numpy as np
from collections import deque
from scipy.signal import butter, sosfilt
import soundfile as sf
import os

# -----------------------------
# Audio PreProcessor class
# -----------------------------
class AudioPreProcessor:
    def __init__(self, sr=16000, frame_size=320, hop_length=160,
                 lowcut=500.0, highcut=7000.0, filter_order=4,
                 save_dir="detected_impulses", save_audio=True):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.save_audio = save_audio # flag

        nyquist = 0.5 * sr
        low = lowcut / nyquist
        high = highcut / nyquist
        self.sos = butter(filter_order, [low, high], btype='band', output='sos')
        self.zi = np.zeros((self.sos.shape[0], 2))

        # Energy tracking
        self.energy_history = deque(maxlen=100)
        self.noise_floor_alpha = 0.99
        self.current_noise_floor = 0.0

        # thresholds
        self.energy_threshold_multiplier = 1.0
        self.noise_margin_db = 3

        # impulse durations
        self.min_impulse_duration_ms = 3
        self.max_impulse_duration_ms = 100
        self.min_impulse_samples = int(self.min_impulse_duration_ms * sr / 1000)
        self.max_impulse_samples = int(self.max_impulse_duration_ms * sr / 1000)

        self.impulse_active = False
        self.impulse_start_idx = 0
        self.impulse_buffer = []

        # conditionally save impulses
        if self.save_audio:
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            self.impulse_counter = 0

    # -----------------------------
    # Bandpass filtering
    # -----------------------------
    def bandpass_filter(self, audio):
        filtered, self.zi = sosfilt(self.sos, audio, zi=self.zi)
        return filtered

    # -----------------------------
    # Features
    # -----------------------------
    def compute_frame_energy(self, frame):
        energy = np.sum(frame**2)
        return -80.0 if energy <= 0 else 10 * np.log10(energy)

    def compute_spectral_flatness(self, frame):
        spectrum = np.abs(np.fft.rfft(frame)) + 1e-8
        gmean = np.exp(np.mean(np.log(spectrum)))
        amean = np.mean(spectrum)
        return gmean / amean

    def update_noise_floor(self, energy):
        if self.current_noise_floor == 0:
            self.current_noise_floor = energy
        else:
            self.current_noise_floor = (
                self.noise_floor_alpha * self.current_noise_floor +
                (1 - self.noise_floor_alpha) * energy
            )
        self.current_noise_floor = min(self.current_noise_floor, -20)

    # -----------------------------
    # Impulse detection
    # -----------------------------
    def detect_energy_impulse(self, frame, frame_idx):
        frame_energy = self.compute_frame_energy(frame)
        self.energy_history.append(frame_energy)

        if not self.impulse_active and len(self.energy_history) > 10:
            avg_energy = np.mean(list(self.energy_history)[-10:])
            if frame_energy < avg_energy + 10:
                self.update_noise_floor(frame_energy)

        if len(self.energy_history) < 10:
            return None

        recent_energies = list(self.energy_history)[-50:]
        energy_mean = np.mean(recent_energies)
        energy_std = np.std(recent_energies)
        threshold = max(self.current_noise_floor + 3, energy_mean + 0.5 * energy_std)

        if not self.impulse_active:
            if frame_energy > threshold:
                self.impulse_active = True
                self.impulse_start_idx = frame_idx
                self.impulse_buffer = [frame]
        else:
            self.impulse_buffer.append(frame)
            if frame_energy < threshold or len(self.impulse_buffer) > self.max_impulse_samples:
                impulse_duration_samples = len(self.impulse_buffer) * self.hop_length
                impulse_duration_ms = impulse_duration_samples * 1000 / self.sr
                self.impulse_active = False

                if (self.min_impulse_duration_ms <= impulse_duration_ms <= self.max_impulse_duration_ms):
                    impulse_audio = np.concatenate(self.impulse_buffer)
                    flatness = self.compute_spectral_flatness(impulse_audio)
                    peak_db = max([self.compute_frame_energy(f) for f in self.impulse_buffer])

                    # Only consider reasonable impulses
                    if flatness < 0.8 and peak_db > 0:
                        if self.save_audio:
                            self.impulse_counter += 1
                            out_path = os.path.join(self.save_dir, f"impulse_{self.impulse_counter}.wav")
                            sf.write(out_path, impulse_audio, self.sr)
                            print(f"✅ Impulse {self.impulse_counter} @ {self.impulse_start_idx*self.hop_length/self.sr:.3f}s | "
                                f"Peak={peak_db:.1f}dB | Saved: {out_path}")
                        else:
                            print(f"✅ Impulse detected @ {self.impulse_start_idx*self.hop_length/self.sr:.3f}s | "
                                f"Peak={peak_db:.1f}dB")
                        return {
                            'audio': impulse_audio,
                            'timestamp': self.impulse_start_idx * self.hop_length / self.sr,
                            'peak_energy_db': peak_db
                        }
                self.impulse_buffer = []
        return None

    # -----------------------------
    # Stream processor
    # -----------------------------
    def process_stream(self, audio_chunk):
        filtered_audio = self.bandpass_filter(audio_chunk)
        impulses = []
        num_frames = (len(filtered_audio) - self.frame_size) // self.hop_length + 1

        for i in range(num_frames):
            start_idx = i * self.hop_length
            end_idx = start_idx + self.frame_size
            frame = filtered_audio[start_idx:end_idx]
            if len(frame) < self.frame_size:
                continue
            windowed = frame * np.hanning(len(frame))
            impulse = self.detect_energy_impulse(windowed, i)
            if impulse is not None:
                impulses.append(impulse)
        return impulses, filtered_audio

# -----------------------------
# Merge impulses
# -----------------------------
def merge_impulses(impulses, merge_window=0.12):
    if not impulses:
        return []
    merged = [impulses[0]]
    for imp in impulses[1:]:
        last = merged[-1]
        if imp['timestamp'] - last['timestamp'] < merge_window:
            last['peak_energy_db'] = max(last['peak_energy_db'], imp['peak_energy_db'])
        else:
            merged.append(imp)
    return merged

# -----------------------------
# Real-time listening
# -----------------------------
duration = 2      # seconds per recording
sample_rate = 44100
preprocessor = AudioPreProcessor(sr=sample_rate, save_audio=True)

print("🎤 Listening for gunshots... Press Ctrl+C to stop.")

try:
    while True:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        try:
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            print("🛑 Recording stopped.")
            break

        audio = recording.flatten()
        impulses, _ = preprocessor.process_stream(audio)
        impulses = merge_impulses(impulses)

        if impulses:
            print(f"🔫 Gunshot-like impulses detected: {len(impulses)}")
        else:
            print("❌ No gunshot detected")

except KeyboardInterrupt:
    print("🛑 Stopped by user.")
