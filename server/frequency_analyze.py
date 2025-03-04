import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import os

# --------------------------------
# 1) 경로 및 파일 설정
# --------------------------------
base_folder_name = 'logs/2024_12_10_16_45'
graph_folder_name = os.path.join(base_folder_name, 'graphs')
os.makedirs(graph_folder_name, exist_ok=True)  # 폴더 없으면 생성

kt_log_file_name = os.path.join(base_folder_name, 'kt_log.csv')
lg_log_file_name = os.path.join(base_folder_name, 'lg_log.csv')

# --------------------------------
# 2) 파일별 반복 처리를 위한 설정
# --------------------------------
file_dict = {
    'kt': kt_log_file_name,
    'lg': lg_log_file_name
}

# --------------------------------
# 3) 기본 파라미터(윈도우 크기 및 스텝 크기)
# --------------------------------
window_size = 128  # 윈도우 크기
step_size = 64     # 윈도우 이동 간격

# --------------------------------
# 4) 반복문을 이용한 처리
# --------------------------------
for key, file_path in file_dict.items():
    # (1) CSV 파일 불러오기
    df = pd.read_csv(file_path)

    # (2) Sequence Number와 Latency 데이터 추출
    sequence_numbers = df['sequence_number'].values
    latency = df['network_latency_ms'].values

    # --------------------------------
    # FFT 수행
    # --------------------------------
    N = len(latency)
    fft_vals = np.fft.fft(latency)
    fft_amp = np.abs(fft_vals)  # 복소수 -> 진폭
    freqs = np.fft.fftfreq(N, d=1)  # Sequence Number 기준 주파수

    # 양의 주파수만 보기
    mask = freqs >= 0
    freqs_pos = freqs[mask]
    fft_amp_pos = fft_amp[mask]

    # FFT 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_pos, fft_amp_pos, marker='o', linestyle='-', markersize=1)
    plt.title(f'FFT of Latency ({key.upper()})')
    plt.xlabel('Frequency (1/Sequence Number)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    graph_file_name = os.path.join(graph_folder_name, f'{key}_fft_spectrum.png')
    plt.savefig(graph_file_name, dpi=150)
    plt.close()

    # FFT 데이터 저장
    fft_data = pd.DataFrame({'freq': freqs_pos, 'amplitude': fft_amp_pos})
    raw_data_file_name = os.path.join(graph_folder_name, f'{key}_fft_data.csv')
    fft_data.to_csv(raw_data_file_name, index=False)

    print(f"[{key.upper()}] FFT complete. Graph: {graph_file_name}, Data: {raw_data_file_name}")

    # --------------------------------
    # Short-Time Fourier Transform (SFFT)
    # --------------------------------
    sfft_times = []
    sfft_magnitudes = []

    for start in range(0, N - window_size, step_size):
        segment = latency[start:start + window_size]
        segment_fft = np.fft.fft(segment)
        segment_amp = np.abs(segment_fft[:window_size // 2])

        sfft_times.append(sequence_numbers[start])
        sfft_magnitudes.append(segment_amp)

    # SFFT 그래프 생성
    plt.figure(figsize=(12, 8))
    plt.imshow(
        np.array(sfft_magnitudes).T,
        aspect='auto',
        extent=[sequence_numbers[0], sequence_numbers[-1], 0, window_size // 2],
        cmap='viridis'
    )
    plt.colorbar(label='Magnitude')
    plt.title(f'Short-Time FFT (SFFT) of Latency ({key.upper()})')
    plt.xlabel('Sequence Number')
    plt.ylabel('Frequency (1/Sequence Number)')
    graph_file_name = os.path.join(graph_folder_name, f'{key}_sfft.png')
    plt.savefig(graph_file_name, dpi=150)
    plt.close()
    print(f"[{key.upper()}] SFFT complete. Graph: {graph_file_name}")

    # --------------------------------
    # Wavelet Transform
    # --------------------------------
    wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(latency, scales, wavelet, sampling_period=1)

    # Wavelet Scalogram 생성
    plt.figure(figsize=(12, 8))
    plt.imshow(np.abs(coefficients), aspect='auto', extent=[sequence_numbers[0], sequence_numbers[-1], frequencies.max(), frequencies.min()], cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title(f'Wavelet Transform Scalogram (Latency) ({key.upper()})')
    plt.xlabel('Sequence Number')
    plt.ylabel('Frequency (1/Sequence Number)')
    graph_file_name = os.path.join(graph_folder_name, f'{key}_wavelet.png')
    plt.savefig(graph_file_name, dpi=150)
    plt.close()

    print(f"[{key.upper()}] Wavelet Transform complete. Graph: {graph_file_name}")
