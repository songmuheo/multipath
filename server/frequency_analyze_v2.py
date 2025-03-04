import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import os

# --------------------------------
# 경로 및 파일 설정
# --------------------------------
base_folder_name = 'logs/2024_12_10_16_45'


kt_log_file_name = os.path.join(base_folder_name, 'kt_log.csv')
lg_log_file_name = os.path.join(base_folder_name, 'lg_log.csv')

file_dict = {
    'kt': kt_log_file_name,
    'lg': lg_log_file_name
}

fft_folder = os.path.join(base_folder_name, 'graphs_spike_analysis', 'fft')
sfft_folder = os.path.join(base_folder_name, 'graphs_spike_analysis', 'sfft')
wavelet_folder = os.path.join(base_folder_name, 'graphs_spike_analysis', 'wavelet')
raw_data_folder = os.path.join(base_folder_name, 'graphs_spike_analysis', 'raw_data')

# Create directories
for folder in [fft_folder, sfft_folder, wavelet_folder, raw_data_folder]:
    os.makedirs(folder, exist_ok=True)

# --------------------------------
# 분석 함수 정의
# --------------------------------
def detect_spikes(sequence_numbers, data, threshold=50):
    """
    스파이크를 감지합니다.
    - threshold: 변화율(차분 값)의 임계값
    """
    diffs = np.abs(np.diff(data))
    spike_indices = np.where(diffs > threshold)[0] + 1  # 변화율 초과 부분
    spike_sequences = sequence_numbers[spike_indices]
    return spike_sequences, spike_indices


def plot_spike_data(sequence_numbers, data, spike_indices, save_path):
    """
    스파이크 데이터를 시각화합니다.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sequence_numbers, data, label='Latency')
    plt.scatter(sequence_numbers[spike_indices], data[spike_indices], color='red', label='Spikes', zorder=5)
    plt.title('Latency with Detected Spikes')
    plt.xlabel('Sequence Number')
    plt.ylabel('Latency (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Spike visualization saved to {save_path}")


def perform_fft(data, save_path_csv, save_path_png):
    """
    FFT 수행 및 결과 저장
    """
    N = len(data)
    fft_vals = np.fft.fft(data)
    fft_amp = np.abs(fft_vals)[:N // 2]
    freqs = np.fft.fftfreq(N, d=1)[:N // 2]

    # Save raw data
    fft_df = pd.DataFrame({'Frequency': freqs, 'Amplitude': fft_amp})
    fft_df.to_csv(save_path_csv, index=False)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_amp, marker='o', linestyle='-', markersize=1)
    plt.title('FFT Analysis')
    plt.xlabel('Frequency (1/Sequence Number)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(save_path_png, dpi=150)
    plt.close()


def perform_sfft(sequence_numbers, data, window_size, step_size, save_path_csv, save_path_png):
    """
    SFFT 수행 및 결과 저장
    """
    N = len(data)
    sfft_magnitudes = []
    for start in range(0, N - window_size, step_size):
        segment = data[start:start + window_size]
        segment_fft = np.fft.fft(segment)
        sfft_magnitudes.append(np.abs(segment_fft[:window_size // 2]))

    sfft_df = pd.DataFrame(np.array(sfft_magnitudes).T)
    sfft_df.to_csv(save_path_csv, index=False)

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(
        np.array(sfft_magnitudes).T,
        aspect='auto',
        extent=[sequence_numbers[0], sequence_numbers[-1], 0, window_size // 2],
        cmap='viridis'
    )
    plt.colorbar(label='Magnitude')
    plt.title('Short-Time FFT (SFFT)')
    plt.xlabel('Sequence Number')
    plt.ylabel('Frequency (1/Sequence Number)')
    plt.savefig(save_path_png, dpi=150)
    plt.close()


def perform_wavelet(data, wavelet_name, scales, save_path_csv, save_path_png):
    """
    웨이브릿 변환 수행 및 결과 저장
    """
    coefficients, frequencies = pywt.cwt(data, scales, wavelet_name, sampling_period=1)
    wavelet_df = pd.DataFrame(coefficients.real)
    wavelet_df.to_csv(save_path_csv, index=False)

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(
        np.abs(coefficients),
        aspect='auto',
        extent=[0, len(data), frequencies.max(), frequencies.min()],
        cmap='viridis'
    )
    plt.colorbar(label='Magnitude')
    plt.title('Wavelet Transform Scalogram')
    plt.xlabel('Data Index')
    plt.ylabel('Frequency (1/Sequence Number)')
    plt.savefig(save_path_png, dpi=150)
    plt.close()


# --------------------------------
# 데이터 처리 및 분석
# --------------------------------
window_size = 128
step_size = 64
wavelet_name = 'cmor1.5-1.0'
scales = np.arange(1, 128)

for key, file_path in file_dict.items():
    # CSV 파일 불러오기
    df = pd.read_csv(file_path)
    df = df.sort_values('sequence_number')  # Sequence Number 기준 정렬
    sequence_numbers = df['sequence_number'].values
    latency = df['network_latency_ms'].values

    # 스파이크 감지
    spike_sequences, spike_indices = detect_spikes(sequence_numbers, latency, threshold=50)

    # 스파이크 시각화
    spike_plot_path = os.path.join(raw_data_folder, f'{key}_spike_visualization.png')
    plot_spike_data(sequence_numbers, latency, spike_indices, spike_plot_path)

    # 각 스파이크 주변 데이터를 분석
    for i, spike_idx in enumerate(spike_indices):
        start_idx = max(0, spike_idx - window_size // 2)
        end_idx = min(len(latency), spike_idx + window_size // 2)
        segment = latency[start_idx:end_idx]

        # FFT 분석
        fft_csv_path = os.path.join(raw_data_folder, f'{key}_spike_{i}_fft.csv')
        fft_png_path = os.path.join(fft_folder, f'{key}_spike_{i}_fft.png')
        perform_fft(segment, fft_csv_path, fft_png_path)

        # SFFT 분석
        sfft_csv_path = os.path.join(raw_data_folder, f'{key}_spike_{i}_sfft.csv')
        sfft_png_path = os.path.join(sfft_folder, f'{key}_spike_{i}_sfft.png')
        perform_sfft(sequence_numbers[start_idx:end_idx], segment, window_size, step_size, sfft_csv_path, sfft_png_path)

        # Wavelet 분석
        wavelet_csv_path = os.path.join(raw_data_folder, f'{key}_spike_{i}_wavelet.csv')
        wavelet_png_path = os.path.join(wavelet_folder, f'{key}_spike_{i}_wavelet.png')
        perform_wavelet(segment, wavelet_name, scales, wavelet_csv_path, wavelet_png_path)
