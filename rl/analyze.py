import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from codec_module import Encoder, Decoder
import glob
import os

# Directory and Encoder/Decoder setup
frame_dir = 'data/frames/frames_og/'
encoder = Encoder(640, 480)
decoder = Decoder()

# Data storage
results = []

# Edge Detection using Canny Edge Detection
def calculate_edge_intensity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)  # Adjust thresholds if needed
    edge_intensity = np.mean(edges)  # Mean edge intensity across all pixels
    return edge_intensity

# Frequency Domain Analysis using Fourier Transform
def calculate_frequency_energy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    high_freq_energy = np.mean(magnitude_spectrum[200:, 200:])  # Analyze specific high-frequency area
    return high_freq_energy

# Image Sharpness using Laplacian Variance
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

# Calculate SSIM between original and decoded frames
def calculate_ssim(original, decoded):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    decoded_gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(original_gray, decoded_gray, full=True)
    return score

# Process each frame in the directory
for frame_path in sorted(glob.glob(os.path.join(frame_dir, '*.png')), key=lambda x: int(os.path.basename(x).split('_')[0])):
    sequence_number = int(os.path.basename(frame_path).split('_')[0])

    # Read and encode frame
    original_frame = cv2.imread(frame_path)
    encoded_data = encoder.encode_frame(frame_path, True)
    
    # Decode the frame
    decoded_frame = decoder.decode_frame(encoded_data, len(encoded_data), 640, 480)
    decoded_frame = np.array(decoded_frame).reshape((480, 640, 3))  # Reshape decoded data if necessary

    # Calculate edge intensity
    edge_intensity = calculate_edge_intensity(original_frame)

    # Calculate high-frequency energy in the frequency domain
    frequency_energy = calculate_frequency_energy(original_frame)

    # Calculate image sharpness
    sharpness = calculate_sharpness(original_frame)

    # Calculate SSIM
    ssim_value = calculate_ssim(original_frame, decoded_frame)

    # Append results
    results.append([sequence_number, edge_intensity, frequency_energy, sharpness, ssim_value])

# Save results to DataFrame
df = pd.DataFrame(results, columns=['Sequence Number', 'Edge Intensity', 'Frequency Energy', 'Sharpness', 'SSIM'])

# Plot each metric against SSIM
metrics = ['Edge Intensity', 'Frequency Energy', 'Sharpness']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    plt.scatter(df[metric], df['SSIM'], s=2)
    plt.xlabel(metric)
    plt.ylabel('SSIM')
    plt.title(f'Relationship between {metric} and SSIM')
    plt.grid()
    plt.savefig(f'analyze/results/{metric}_SSIM', format='png')
    plt.show()

# Plot SSIM over Sequence Number
plt.figure(figsize=(10, 6))
plt.plot(df['Sequence Number'], df['SSIM'], label='SSIM', color='blue', markersize=2)
plt.xlabel('Sequence Number')
plt.ylabel('SSIM')
plt.title('SSIM over Sequence')
plt.grid()
plt.legend()
plt.savefig(f'analyze/results/SSIM_over_sequence', format='png')
plt.show()

# Save DataFrame to CSV for logging
df.to_csv('analyze/results/frame_analysis_log.csv', index=False)
