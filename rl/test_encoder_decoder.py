import os
import cv2
import numpy as np
from codec_module import Encoder, Decoder
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import matplotlib.pyplot as plt

def main():
    metric = 'ssim'
    frames_dir = 'data/frames/frames_og/'
    frame_files = os.listdir(frames_dir)
    
    # Extract sequence number and sort files based on it
    frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[0]))
    frames_to_process = frame_files[:]  # Process frames 0 to 10

    # GOP 설정 리스트
    # gop_values = [1, 3, 5, 7, 10, 15, 20]
    gop_values = [1]
    ssim_avg = []
    psnr_avg = []

    for gop in gop_values:
        print(f"\nProcessing with GOP = {gop}")
        
        encoder = Encoder(640, 480)
        decoder = Decoder()

        encoded_frames = []
        decoded_frames = []
        if metric == 'ssim':
            ssim_values = []
        else:
            psnr_values = []
        data_sizes = []

        for idx, file_name in enumerate(frames_to_process):
            frame_path = os.path.join(frames_dir, file_name)
            print(frame_path)

            # I-frame 여부 결정
            is_i_frame = (idx % gop == 0)
            encoded_data = encoder.encode_frame(frame_path, is_i_frame)
            encoded_frames.append(encoded_data)

            # Data size
            data_size = len(encoded_data)
            data_sizes.append(data_size)

            # Decode frame
            if encoded_data:
                decoded_frame = decoder.decode_frame(encoded_data, len(encoded_data), 640, 480)
                decoded_frames.append(decoded_frame)

                if metric == 'ssim':
                    # Calculate SSIM
                    # Load original frame
                    original_frame = np.array(Image.open(frame_path).convert('L'))
                    # Convert decoded frame to grayscale
                    decoded_gray = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2GRAY)
                    ssim_value = ssim(original_frame, decoded_gray)
                    ssim_values.append(ssim_value)
                else:
                    # Calculate PSNR
                    # Load original frame in color
                    original_frame = np.array(Image.open(frame_path).convert('RGB'))
                    psnr_value = cv2.PSNR(original_frame, decoded_frame)
                    psnr_values.append(psnr_value)

            else:
                print(f"Encoded data is empty for frame {idx}.")
                decoded_frames.append(None)
                
                if metric == 'ssim':
                    ssim_values.append(0)
                else:
                    psnr_values.append(0)
            if metric == 'ssim':
                print(f"Frame {idx}: Data Size = {data_size} bytes, SSIM = {ssim_values[-1]:.4f}")
            else:
                print(f"Frame {idx}: Data Size = {data_size} bytes, PSNR = {psnr_values[-1]:.4f}")

        if metric == 'ssim':
            # SSIM 평균 계산
            average_ssim = np.mean(ssim_values)
            print(f"\nAverage SSIM for all frames with GOP={gop}: {average_ssim:.4f}")
            ssim_avg.append(average_ssim)
        else:
            # PSNR 평균 계산
            average_psnr = np.mean(psnr_values)
            print(f"\nAverage PSNR for all frames with GOP={gop}: {average_psnr:.4f}")
            psnr_avg.append(average_psnr)

        if metric == 'ssim':
            # SSIM 값을 프레임별로 그래프로 표시
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(ssim_values)), ssim_values, marker='o', linestyle='-', color='b', markersize=2)
            plt.xlabel('Frame Number')
            plt.ylabel('SSIM')
            plt.title(f'SSIM per Frame (GOP={gop})')

            # 그래프를 파일로 저장
            plt.savefig(f'ssim_per_frame_gop_{gop}.png', format='png')
            plt.close()  # 그래프를 닫아 메모리 절약
        else:
            # PSNR 값을 프레임별로 그래프로 표시
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(psnr_values)), psnr_values, marker='o', linestyle='-', color='b', markersize=2)
            plt.xlabel('Frame Number')
            plt.ylabel('PSNR')
            plt.title(f'PSNR per Frame (GOP={gop})')

            # 그래프를 파일로 저장
            plt.savefig(f'psnr_per_frame_gop_{gop}.png', format='png')
            plt.close()  # 그래프를 닫아 메모리 절약
    if metric == 'ssim':
        print(ssim_avg)
    else:
        print(psnr_avg)

if __name__ == '__main__':
    main()
