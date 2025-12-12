import numpy as np, os
from matplotlib import pyplot as plt
from scipy import signal

class SignalGenerator:
    @staticmethod
    def create_chirp_signal(base_frequency: int, half_bandwidth: int, time_vector: np.ndarray, duration: int) -> np.ndarray:
        return np.sin(2 * np.pi * (base_frequency + half_bandwidth * time_vector / duration) * time_vector)

    @staticmethod
    def create_sine_signal(base_frequency: int, time_vector: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * time_vector * base_frequency)

class WindowGenerator:
    @staticmethod
    def create_rectangular_window(window_size: int, manual: bool = False) -> np.ndarray:
        return np.ones(window_size) if manual else signal.windows.boxcar(window_size)

    @staticmethod
    def create_hanning_window(window_size: int, manual: bool = False) -> np.ndarray:
        return 0.5 - 0.5 * np.cos((2 * np.pi * np.arange(window_size)) / (window_size - 1)) \
            if manual else signal.windows.hann(window_size)

    @staticmethod
    def create_hamming_window(window_size: int, manual: bool = False) -> np.ndarray:
        return 0.54 - 0.46 * np.cos((2 * np.pi * np.arange(window_size)) / (window_size - 1)) \
            if manual else signal.windows.hamming(window_size)

    @staticmethod
    def create_blackman_window(window_size: int, manual: bool = False) -> np.ndarray:
        return 0.42 - 0.5 * np.cos((2 * np.pi * np.arange(window_size)) / (window_size - 1)) \
            + 0.08 * np.cos((4 * np.pi * np.arange(window_size)) / (window_size - 1)) \
            if manual else signal.windows.blackman(window_size)

class STFTCalculator:
    @staticmethod
    def calculate_STFT(signal: np.ndarray, window: np.ndarray, n_overlap: int, fs: int) -> np.ndarray:
        N = len(window)
        hop_size = N - n_overlap
        scaling_factor = np.sum(window)

        stft_list = []
        time_indices = []

        for i in range(0, len(signal) - N + 1, hop_size):
            segment = signal[i: i + N]
            windowed_segment = window * segment

            dft_result = np.fft.fft(windowed_segment)
            magnitude = np.abs(dft_result[:N // 2 + 1])

            stft_list.append(magnitude / scaling_factor)

            center_sample = i + N / 2
            time_indices.append(center_sample)

        stft_matrix = np.array(stft_list).T

        frequencies = np.fft.rfftfreq(N, d = 1/fs)
        times = np.array(time_indices) / fs

        return frequencies, times, stft_matrix


def main():
    T = 2
    fs = 1000
    N = 128         # Window Size
    n_overlap = 64

    # Create Time Vector
    time_vector = np.arange(0, T, 1 / fs) # I look into a single time duration :)
    print(50 * "=", "\n", "Time Vector:", time_vector)

    result_dir = "Results"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "Signal Components"), exist_ok=True)
    # ===================================================================================
    # Create Signals
    # -----------------------------------------------------------------------------------
    chirp1_signal = SignalGenerator.create_chirp_signal(
        base_frequency = 20,
        time_vector = time_vector,
        half_bandwidth = 100,
        duration = T
    )
    chirp2_signal = SignalGenerator.create_chirp_signal(
        base_frequency = 80,
        time_vector = time_vector,
        half_bandwidth = 70,
        duration = T
    )
    sine1_signal = SignalGenerator.create_sine_signal(
        base_frequency = 40,
        time_vector = time_vector,
    )
    sine2_signal = SignalGenerator.create_sine_signal(
        base_frequency = 70,
        time_vector = time_vector,
    )

    received_signal = sine1_signal + sine2_signal + chirp1_signal + chirp2_signal

    signals_to_plot = [
        (sine1_signal, "Sine Wave 1 (40 Hz)", "signal_sine1.png"),
        (sine2_signal, "Sine Wave 2 (70 Hz)", "signal_sine2.png"),
        (chirp1_signal, "Chirp Signal 1 (Starts 20 Hz, Half BW: 100 Hz)", "signal_chirp1.png"),
        (chirp2_signal, "Chirp Signal 2 (Starts 80 Hz, Half BW: 70 Hz)", "signal_chirp2.png"),
        (received_signal, "Combined Received Signal", "signal_combined.png")
    ]
    for sig_data, title, filename in signals_to_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, sig_data)

        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(result_dir,"Signal Components", filename)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

        plt.close()
    # ===================================================================================
    # Create Windows Needed
    # -----------------------------------------------------------------------------------
    window_types = [
        ("Rectangular", WindowGenerator.create_rectangular_window),
        ("Hanning", WindowGenerator.create_hanning_window),
        ("Hamming", WindowGenerator.create_hamming_window),
        ("Blackman", WindowGenerator.create_blackman_window)
    ]
    all_windows = {}
    print(50 * "=", "\n", "Manual Windows:")
    for name, func in window_types:
        manual_win = func(window_size=N, manual=True)
        # library_win = func(N=N, manual=False)
        print(f"{name}:\n", manual_win)
        all_windows[name] = manual_win

    # ===================================================================================
    # Calculate STFT Using Windows
    # -----------------------------------------------------------------------------------
    plt.figure(figsize=(15, 10))

    for i, (name, win_func) in enumerate(window_types):
        freqs, times, Sxx = STFTCalculator.calculate_STFT(received_signal, all_windows[name], n_overlap, fs=1000)

        plt.subplot(2, 2, i + 1)
        plt.pcolormesh(times, freqs, Sxx, shading='gouraud', cmap='viridis')

        plt.colorbar(label='Magnitude (Linear)')
        plt.title(f"Spectrogram: {name}")
        plt.ylabel("Frequency Bin")
        plt.xlabel("Time Step (Frame)")

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "ManualSpectrogram.png"))
    plt.show()

    plt.figure(figsize=(15, 10))

    for i, (name, win_func) in enumerate(window_types):
        f, t, Sxx = signal.spectrogram(
            received_signal,
            fs=fs,
            window=all_windows[name],
            nperseg=N,
            noverlap=n_overlap,
            mode='magnitude',
            scaling='spectrum'
        )

        plt.subplot(2, 2, i + 1)
        plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')

        plt.colorbar(label='Magnitude (Linear)')
        plt.title(f"SciPy Spectrogram: {name}")
        plt.ylabel("Frequency Bin")
        plt.xlabel("Time Step (Frame)")

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "LibrarySpectrogram.png"))
    plt.show()


if __name__ == "__main__":
    main()