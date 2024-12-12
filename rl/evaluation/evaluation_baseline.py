import os
import numpy as np
import pandas as pd
from codec_module import Encoder, Decoder
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt

class PerformanceEvaluator:
    def __init__(self, frames_dir, env_logs_dir, latency_threshold=40):
        self.frames_dir = frames_dir
        self.env_logs_dir = env_logs_dir
        self.latency_threshold = latency_threshold
        self.sequence_numbers = []
        self.seq_num_to_frame_path = {}
        self.get_sequence_numbers()
        self.adjust_sequence_numbers()
        self.methods = ['kt', 'lg', 'combine']
        self.gops = [1, 10, 30]

    def get_sequence_numbers(self):
        """Map sequence numbers to file paths."""
        files = os.listdir(self.frames_dir)
        seq_num_to_frame_path = {}
        for f in files:
            seq_num = int(f.split('_')[0])
            seq_num_to_frame_path[seq_num] = os.path.join(self.frames_dir, f)
        self.sequence_numbers = sorted(seq_num_to_frame_path.keys())
        self.seq_num_to_frame_path = seq_num_to_frame_path

    def adjust_sequence_numbers(self):
        """Adjust sequence numbers to use the last 20%."""
        total_sequences = len(self.sequence_numbers)
        split_index = int(total_sequences * 0.8)
        self.sequence_numbers = self.sequence_numbers[split_index:]

    def load_env_log(self, method):
        """Load environment log for the specified method."""
        log_path = os.path.join(self.env_logs_dir, f"{method}_log.csv")
        df = pd.read_csv(log_path)
        df = df[['sequence_number', 'network_latency_ms']]
        return df.set_index('sequence_number')

    def compute_prr(self, latency_series):
        """Compute Packet Reception Ratio (PRR)."""
        total_packets = len(latency_series)
        received_packets = (latency_series <= self.latency_threshold).sum()
        return received_packets / total_packets

    def evaluate(self):
        """Evaluate performance."""
        results = {'method': [], 'gop': [], 'prr': [], 'avg_ssim': [], 'avg_datasize': []}

        for method in self.methods:
            print(f"Evaluating method: {method}")
            latency_df = self.load_env_log(method)

            # self.sequence_numbers를 인덱스로 가지도록 latency_df 재정렬
            # 로그에 없는 sequence_number는 NaN으로 처리
            latency_series = latency_df.reindex(self.sequence_numbers)['network_latency_ms']

            # 수신하지 않은 (로그에 없는) 프레임을 loss로 처리하기 위해 NaN 값을 임계치보다 큰 값으로 채움
            latency_series = latency_series.fillna(self.latency_threshold + 1)

            prr = self.compute_prr(latency_series)
            print(f"PRR for {method}: {prr}")

            for gop in self.gops:
                print(f"  Evaluating GoP size: {gop}")
                encoder = Encoder(640, 480)
                decoder = Decoder()
                encoder.reset()
                decoder.reset()

                ssim_scores = []
                datasize_list = []
                is_i_frame_list = self.get_is_i_frame_list(gop, len(self.sequence_numbers))

                # 실제 평가할 sequence_numbers 기준으로 loop
                for idx, seq_num in enumerate(self.sequence_numbers):
                    frame_path = self.seq_num_to_frame_path.get(seq_num)
                    if frame_path is None:
                        # 프레임 파일 자체가 없는 경우도 loss로 간주
                        ssim_scores.append(0)
                        datasize_list.append(0)
                        continue

                    is_i_frame = is_i_frame_list[idx]
                    encoded_data = encoder.encode_frame(frame_path, is_i_frame)
                    data_size = len(encoded_data)
                    if method == 'combine':
                        data_size *= 2  # Double data size for 'combine'
                    datasize_list.append(data_size)

                    network_latency = latency_series.loc[seq_num]
                    if network_latency > self.latency_threshold:
                        # Packet loss (네트워크 지연 초과 혹은 수신하지 않은 경우)
                        ssim_scores.append(0)
                        continue

                    try:
                        decoded_frame = decoder.decode_frame(encoded_data, len(encoded_data), 640, 480)
                        if decoded_frame is None:
                            ssim_scores.append(0)
                            continue
                        original_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                        decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2GRAY)
                        ssim_score = ssim(original_frame, decoded_frame)
                        ssim_scores.append(ssim_score)
                    except:
                        ssim_scores.append(0)

                avg_ssim = np.mean(ssim_scores) if len(ssim_scores) > 0 else 0
                avg_datasize = np.mean(datasize_list) if len(datasize_list) > 0 else 0
                print(f"    Average SSIM for GoP {gop}: {avg_ssim}")
                print(f"    Average Data Size for GoP {gop}: {avg_datasize} bytes")

                # Append results
                results['method'].append(method)
                results['gop'].append(gop)
                results['prr'].append(prr)
                results['avg_ssim'].append(avg_ssim)
                results['avg_datasize'].append(avg_datasize)

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv("evaluation_results.csv", index=False)
        print("Results saved to evaluation_results.csv")

        self.plot_results(results)


    def get_is_i_frame_list(self, gop_size, total_frames):
        """Generate I-frame flags based on GoP size."""
        is_i_frame_list = [(i % gop_size == 0) for i in range(total_frames)]
        return is_i_frame_list

    def plot_results(self, results):
        """Plot results."""
        results_df = pd.DataFrame(results)

        # PRR Plot
        plt.figure()
        prr_means = results_df.groupby('method')['prr'].mean()
        prr_means.plot(kind='bar')
        plt.title("PRR by Method")
        plt.xlabel("Method")
        plt.ylabel("PRR")
        plt.savefig("prr_results.png")
        print("PRR plot saved as prr_results.png")

        # SSIM and Data Size Plots by GoP
        for gop in self.gops:
            subset = results_df[results_df['gop'] == gop]

            # SSIM Plot
            plt.figure()
            subset.set_index('method')['avg_ssim'].plot(kind='bar')
            plt.title(f"SSIM by Method (GoP={gop})")
            plt.xlabel("Method")
            plt.ylabel("Average SSIM")
            plt.savefig(f"ssim_results_gop_{gop}.png")
            print(f"SSIM plot saved as ssim_results_gop_{gop}.png")

            # Data Size Plot
            plt.figure()
            subset.set_index('method')['avg_datasize'].plot(kind='bar')
            plt.title(f"Data Size by Method (GoP={gop})")
            plt.xlabel("Method")
            plt.ylabel("Average Data Size (bytes)")
            plt.savefig(f"datasize_results_gop_{gop}.png")
            print(f"Data Size plot saved as datasize_results_gop_{gop}.png")


if __name__ == "__main__":
    frames_dir = "/home/songmu/multipath/client/logs/2024_12_09_16_07/frames_with_sequence"
    env_logs_dir = "/home/songmu/multipath/server/logs/2024_12_09_16_02"
    evaluator = PerformanceEvaluator(frames_dir, env_logs_dir)
    evaluator.evaluate()
