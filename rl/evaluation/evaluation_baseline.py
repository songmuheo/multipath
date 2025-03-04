import os
import numpy as np
import pandas as pd
from codec_module import Encoder, Decoder
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt

class PerformanceEvaluator:
    def __init__(self, frames_dir, env_logs_dir, latency_threshold, training_data_split):
        self.frames_dir = frames_dir
        self.env_logs_dir = env_logs_dir
        self.latency_threshold = latency_threshold
        self.sequence_numbers = []
        self.seq_num_to_frame_path = {}
        self.get_sequence_numbers()
        self.adjust_sequence_numbers(training_data_split)
        # self.methods = ['kt', 'lg', 'combine']
        self.methods = ['combine', 'kt', 'lg']
        self.gops = [1, 10, 15, 30, 60]

    def get_sequence_numbers(self):
        """Map sequence numbers to file paths."""
        files = os.listdir(self.frames_dir)
        seq_num_to_frame_path = {}
        for f in files:
            seq_num = int(f.split('_')[0])
            seq_num_to_frame_path[seq_num] = os.path.join(self.frames_dir, f)
        self.sequence_numbers = sorted(seq_num_to_frame_path.keys())
        self.seq_num_to_frame_path = seq_num_to_frame_path

    def adjust_sequence_numbers(self, training_data_split):
        """Adjust sequence numbers to use the last (1 - training_data_split) portion."""
        total_sequences = len(self.sequence_numbers)
        split_index = int(total_sequences * training_data_split)
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

    def get_frame_type(self, encoded_data):
        """
        H.264 NAL 유닛을 분석하여 프레임 타입을 반환합니다.
        - I-frame: 3-byte Start Code (\x00\x00\x01)
        - P-frame: 4-byte Start Code (\x00\x00\x00\x01)
        """
        if len(encoded_data) < 5:
            return "Unknown"

        i = 0
        while i < len(encoded_data) - 5:
            # I-frame (3-byte Start Code)
            if encoded_data[i:i+3] == b'\x00\x00\x01' and (encoded_data[i+3] & 0x1F) == 5:
                return "I-frame"
            
            # P-frame (4-byte Start Code)
            if encoded_data[i:i+4] == b'\x00\x00\x00\x01' and (encoded_data[i+4] & 0x1F) == 1:
                return "P-frame"

            i += 1  # 한 바이트씩 이동하며 탐색

        return "Unknown"



    def evaluate(self):
        """Evaluate performance."""
        results = {'method': [], 'gop': [], 'prr': [], 'avg_ssim': [], 'avg_datasize': []}

        for method in self.methods:
            print(f"Evaluating method: {method}")
            latency_df = self.load_env_log(method)

            # self.sequence_numbers를 인덱스로 가지도록 latency_df 재정렬
            latency_series = latency_df.reindex(self.sequence_numbers)['network_latency_ms']

            # 로그에 없는 (NaN) 값을 임계치보다 큰 값으로 채워서 수신되지 않은 것으로 처리
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

                # 추가: raw data 저장을 위한 딕셔너리
                raw_data = {
                    'sequence_number': [],
                    'network_latency_ms': [],
                    'ssim': [],
                    'datasize': [],
                    'frame type': []
                }

                is_i_frame_list = self.get_is_i_frame_list(gop, len(self.sequence_numbers))

                for idx, seq_num in enumerate(self.sequence_numbers):
                    frame_path = self.seq_num_to_frame_path.get(seq_num)
                    is_i_frame = is_i_frame_list[idx]
                    # 인코딩
                    encoded_data = encoder.encode_frame(frame_path, is_i_frame)
                    frame_type_str = self.get_frame_type(encoded_data)  # 프레임 타입 직접 분석

                    # frame_type_str = "I-frame" if is_i_frame else "P-frame"  # **추가**

                    data_size = len(encoded_data)
                    if method == 'combine':
                        data_size *= 2
                    datasize_list.append(data_size)

                    # 네트워크 레이턴시 확인
                    network_latency = latency_series.loc[seq_num]

                    # **frame loss 확인 (csv에 sequence_number 존재 여부 체크)**
                    if pd.isna(network_latency):
                        print(f"Frame loss detected at sequence {seq_num}")
                        ssim_scores.append(0)
                        raw_data['sequence_number'].append(seq_num)
                        raw_data['network_latency_ms'].append(float('nan'))  # NaN으로 기록
                        raw_data['ssim'].append(0)
                        raw_data['datasize'].append(data_size)
                        raw_data['frame type'].append(frame_type_str)
                        continue

                    # packet loss 판정 (latency 임계값 초과)
                    if network_latency > self.latency_threshold:
                        ssim_scores.append(0)
                        raw_data['sequence_number'].append(seq_num)
                        raw_data['network_latency_ms'].append(network_latency)
                        raw_data['ssim'].append(0)
                        raw_data['datasize'].append(data_size)
                        raw_data['frame type'].append(frame_type_str)
                        continue
                    
                    # 디코딩 & SSIM 계산
                    try:
                        decoded_frame = decoder.decode_frame(encoded_data, len(encoded_data), 640, 480)
                        if decoded_frame is None:
                            # print(f"[Warning] Decoded frame is None at sequence {seq_num} ({frame_type_str})")
                            ssim_scores.append(0)
                            raw_data['sequence_number'].append(seq_num)
                            raw_data['network_latency_ms'].append(network_latency)
                            raw_data['ssim'].append(0)
                            raw_data['datasize'].append(data_size)
                            raw_data['frame type'].append(frame_type_str)
                            continue
                        if decoded_frame.size == 0:
                            # print(f"[Warning] Decoded frame is empty at sequence {seq_num} ({frame_type_str})")
                            ssim_scores.append(0)
                            raw_data['sequence_number'].append(seq_num)
                            raw_data['network_latency_ms'].append(network_latency)
                            raw_data['ssim'].append(0)
                            raw_data['datasize'].append(data_size)
                            raw_data['frame type'].append(frame_type_str)
                            continue

                        original_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                        decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2GRAY)
                        ssim_score = ssim(original_frame, decoded_frame)
                        ssim_scores.append(ssim_score)

                        # raw data 기록
                        raw_data['sequence_number'].append(seq_num)
                        raw_data['network_latency_ms'].append(network_latency)
                        raw_data['ssim'].append(ssim_score)
                        raw_data['datasize'].append(data_size)
                        raw_data['frame type'].append(frame_type_str)

                    except Exception as e:
                        # **에러 발생 시 I-frame/P-frame 정보 로깅**
                        print(f"Decoding error at sequence {seq_num} ({frame_type_str}): {e}")
                        ssim_scores.append(0)
                        raw_data['sequence_number'].append(seq_num)
                        raw_data['network_latency_ms'].append(network_latency)
                        raw_data['ssim'].append(0)
                        raw_data['datasize'].append(data_size)
                        raw_data['frame type'].append(frame_type_str)

                # 평균값 계산
                avg_ssim = np.mean(ssim_scores) if len(ssim_scores) > 0 else 0
                avg_datasize = np.mean(datasize_list) if len(datasize_list) > 0 else 0
                print(f"    Average SSIM for GoP {gop}: {avg_ssim}")
                print(f"    Average Data Size for GoP {gop}: {avg_datasize} bytes")

                # 결과 기록
                results['method'].append(method)
                results['gop'].append(gop)
                results['prr'].append(prr)
                results['avg_ssim'].append(avg_ssim)
                results['avg_datasize'].append(avg_datasize)

                # [추가] raw data를 CSV 파일로 저장
                csv_save_folder = os.path.join(self.env_logs_dir, 'baseline')
                # csv_save_folder = os.path.join(csv_save_folder, str(int(training_data_split * 100)))
                # csv_save_folder = os.path.join(csv_save_folder, str(int(training_data_split * 100)))

                # os.makedirs(os.path.dirname(csv_save_folder), exist_ok=True)
                raw_data_df = pd.DataFrame(raw_data)
                raw_data_csv_path = os.path.join(
                    csv_save_folder,
                    f"raw_data_{method}_gop{gop}_threshold_{self.latency_threshold}.csv"
                )
                raw_data_df.to_csv(raw_data_csv_path, index=False)
                print(f"      Raw data saved to {raw_data_csv_path}")

        # 종합 결과 저장
        results_df = pd.DataFrame(results)
        summary_csv_path = os.path.join(csv_save_folder, f"evaluation_results_threshold_{self.latency_threshold}.csv")
        results_df.to_csv(summary_csv_path, index=False)
        print(f"Summary results saved to {summary_csv_path}")

        # 그래프 플롯
        # self.plot_results(results, self.env_logs_dir)

    def get_is_i_frame_list(self, gop_size, total_frames):
        """Generate I-frame flags based on GoP size."""
        is_i_frame_list = [(i % gop_size == 0) for i in range(total_frames)]
        return is_i_frame_list

    def plot_results(self, results, path):
        """Plot results."""
        results_df = pd.DataFrame(results)

        # PRR Plot
        plt.figure()
        prr_means = results_df.groupby('method')['prr'].mean()
        prr_means.plot(kind='bar')
        plt.title("PRR by Method")
        plt.xlabel("Method")
        plt.ylabel("PRR")
        baseline_folder_path = os.path.join(path, 'baseline')
        graph_folder_path = os.path.join(baseline_folder_path, 'graph')
        os.makedirs(os.path.dirname(graph_folder_path), exist_ok=True)
        prr_plot_path = os.path.join(graph_folder_path, f"prr_results_threshold_{self.latency_threshold}.png")
        plt.savefig(prr_plot_path)
        print(f"PRR plot saved as {prr_plot_path}")
        plt.close()

        # SSIM / Data Size Plot
        for gop in self.gops:
            subset = results_df[results_df['gop'] == gop]

            # SSIM Plot
            plt.figure()
            subset.set_index('method')['avg_ssim'].plot(kind='bar')
            plt.title(f"SSIM by Method (GoP={gop})")
            plt.xlabel("Method")
            plt.ylabel("Average SSIM")
            ssim_plot_path = os.path.join(graph_folder_path, f"ssim_results_gop_{gop}_threshold_{self.latency_threshold}.png")
            plt.savefig(ssim_plot_path)
            print(f"SSIM plot saved as {ssim_plot_path}")
            plt.close()

            # Data Size Plot
            plt.figure()
            subset.set_index('method')['avg_datasize'].plot(kind='bar')
            plt.title(f"Data Size by Method (GoP={gop})")
            plt.xlabel("Method")
            plt.ylabel("Average Data Size (bytes)")
            datasize_plot_path = os.path.join(graph_folder_path, f"datasize_results_gop_{gop}_threshold_{self.latency_threshold}.png")
            plt.savefig(datasize_plot_path)
            print(f"Data Size plot saved as {datasize_plot_path}")
            plt.close()


if __name__ == "__main__":
    frames_dir = "/home/songmu/multipath/client/logs/2025_02_17_14_14/frames_with_sequence"
    env_logs_dir = "/home/songmu/multipath/server/logs/2025_02_17_13_57"
    # frames_dir = "/home/songmu/multipath/client/logs/2024_12_10_17_00/frames_with_sequence"
    # env_logs_dir = "/home/songmu/multipath/server/logs/2024_12_10_16_45"

    latency_threshold = 100
    training_data_split = 0.6

    evaluator = PerformanceEvaluator(frames_dir, env_logs_dir, latency_threshold, training_data_split)
    evaluator.evaluate()
