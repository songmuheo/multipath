import os
import csv
import numpy as np
import pandas as pd
import cv2
from codec_module import Encoder, Decoder
from skimage.metrics import structural_similarity as ssim

class PerformanceEvaluator:
    def __init__(self, frames_dir, env_logs_dir, latency_threshold=40):
        self.frames_dir = frames_dir
        self.env_logs_dir = env_logs_dir
        self.latency_threshold = latency_threshold
        self.encoder = Encoder(640, 480)
        self.decoder = Decoder()
        self.frames = self.load_frames()
        self.kt_log = self.load_env_log('kt_log.csv')
        self.lg_log = self.load_env_log('lg_log.csv')

    def load_frames(self):
        """Load all frame paths and sort them by sequence number."""
        files = os.listdir(self.frames_dir)
        frame_paths = {int(f.split('_')[0]): os.path.join(self.frames_dir, f) for f in files}
        return dict(sorted(frame_paths.items()))

    def load_env_log(self, filename):
        """Load the log file and map sequence_number to network_latency_ms."""
        path = os.path.join(self.env_logs_dir, filename)
        df = pd.read_csv(path, usecols=['sequence_number', 'network_latency_ms'])
        return df.set_index('sequence_number')['network_latency_ms'].to_dict()

    def get_latency(self, latency_map, sequence_number):
        """Get the network latency for a given sequence number."""
        return latency_map.get(sequence_number, None)

    def calculate_ssim(self, decoded_frame, original_frame_path):
        """Calculate SSIM between the decoded frame and original frame."""
        original_frame = cv2.imread(original_frame_path, cv2.IMREAD_GRAYSCALE)
        if decoded_frame is None or original_frame is None:
            return 0.0
        decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2GRAY)
        return ssim(original_frame, decoded_frame)

    def evaluate(self):
        """Evaluate the performance based on the user's requirements."""
        total_frames = len(self.frames)
        i_frame_count = 0
        p_frame_count = 0
        total_data_size = 0
        total_ssim = 0
        total_received_frames = 0
        path_counts = {'lg': 0, 'kt': 0, 'both': 0}
        
        prev_ssim = 1.0  # Initialize with a high SSIM for the first frame

        latency_lg = self.get_latency(self.lg_log, list(self.frames.keys())[0])
        latency_kt = self.get_latency(self.kt_log, list(self.frames.keys())[0])

        for seq_num, frame_path in self.frames.items():
            is_i_frame = prev_ssim <= 0.9
            if is_i_frame:
                i_frame_count += 1
            else:
                p_frame_count += 1

            # Encode the frame
            encoded_data = self.encoder.encode_frame(frame_path, is_i_frame)
            data_size = len(encoded_data)

            # Update latency values
            recent_lg_latency = self.get_latency(self.lg_log, seq_num)
            recent_kt_latency = self.get_latency(self.kt_log, seq_num)

            if recent_lg_latency is not None:
                latency_lg = 0.2 * recent_lg_latency + 0.7 * latency_lg
            if recent_kt_latency is not None:
                latency_kt = 0.2 * recent_kt_latency + 0.7 * latency_kt

            if latency_lg > self.latency_threshold and latency_kt > self.latency_threshold:
                path_used = 'both'
                data_size *= 2
                path_counts['both'] += 1
            elif latency_lg < latency_kt:
                path_used = 'lg'
                path_counts['lg'] += 1
            else:
                path_used = 'kt'
                path_counts['kt'] += 1

            total_data_size += data_size

            # Determine if we decode the frame
            decode_flag = False
            if path_used in ['lg', 'both']:
                latency = self.get_latency(self.lg_log, seq_num)
                if latency is not None and latency < self.latency_threshold:
                    decode_flag = True
            if path_used in ['kt', 'both']:
                latency = self.get_latency(self.kt_log, seq_num)
                if latency is not None and latency < self.latency_threshold:
                    decode_flag = True

            if decode_flag:
                decoded_frame = self.decoder.decode_frame(encoded_data, len(encoded_data), 640, 480)
                ssim_score = self.calculate_ssim(decoded_frame, frame_path)
                total_ssim += ssim_score
                prev_ssim = ssim_score
                total_received_frames += 1
            else:
                prev_ssim = 0.0  # No frame was decoded, so SSIM is 0

        gop = (i_frame_count + p_frame_count) / i_frame_count if i_frame_count > 0 else 0
        avg_data_size = total_data_size / total_frames
        avg_ssim = total_ssim / total_frames
        prr = total_received_frames / total_frames

        print(f"GoP: {gop}")
        print(f"# of Path Usage: {path_counts}")
        print(f"Average Data Size: {avg_data_size} bytes")
        print(f"Average SSIM: {avg_ssim}")
        print(f"Packet Reception Ratio (PRR): {prr}")

        return {
            'GoP': gop,
            '# of path': path_counts,
            'Average Data Size': avg_data_size,
            'Average SSIM': avg_ssim,
            'PRR': prr
        }


if __name__ == "__main__":
    frames_dir = "/home/songmu/Multipath/rl/data/frames/frames_og/"
    env_logs_dir = "/home/songmu/Multipath/rl/data/env_logs/"
    evaluator = PerformanceEvaluator(frames_dir, env_logs_dir)
    results = evaluator.evaluate()
