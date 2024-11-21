# environment/env.py
import os
import pandas as pd
import numpy as np
import cv2
from codec_module import Encoder, Decoder
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import math

class StreamingEnvironment:
    def __init__(self, config):
        self.config = config
        self.latency_threshold = self.config['latency_threshold']
        self.ssim_threshold = self.config['ssim_threshold']
        self.gamma_reward = self.config['gamma_reward']
        self.max_datasize = self.config['max_datasize']
        self.kt_log = pd.read_csv(config['kt_log_dir'])
        self.lg_log = pd.read_csv(config['lg_log_dir'])
        self.encoder = Encoder(640, 480)
        self.decoder = Decoder()
        # frame을 training/evaluation으로 나눔
        self.frames_dir = config['frame_dir']
        self.sequence_numbers = self.get_sequence_numbers()
        self.adjust_sequence_numbers()
        # Normalize 하기 위함
        self.latency_values = self.get_all_latency_values()
        self.max_latency = np.percentile(self.latency_values, 95)  # 95th percentile        
        self.current_episode = 0
        self.kt_latency_history = []
        self.lg_latency_history = []
        self.current_step = 0
        self.state = None
        self.done = False

    def get_all_latency_values(self):
        kt_latencies = self.kt_log['network_latency_ms'].values
        lg_latencies = self.lg_log['network_latency_ms'].values
        return np.concatenate([kt_latencies, lg_latencies])
    
    def get_sequence_numbers(self):
        files = os.listdir(self.frames_dir)
        seq_nums = [int(f.split('_')[0]) for f in files]
        return sorted(seq_nums)

    def adjust_sequence_numbers(self):
        mode = self.config['mode']
        total_sequences = len(self.sequence_numbers)
        split_index = int(total_sequences * self.config['training_data_split'])
        if mode == 'train':
            self.sequence_numbers = self.sequence_numbers[:split_index]
        elif mode == 'eval':
            self.sequence_numbers = self.sequence_numbers[split_index:]

    def reset(self):
        self.current_step = 0
        self.done = False
        self.encoder.reset()
        self.decoder.reset()
        self.kt_latency_history.clear()
        self.lg_latency_history.clear()
        self.kt_last_latency = 0
        self.lg_last_latency = 0
        self.last_frame_loss = False
        self.state = self.get_initial_state()
        self.current_episode += 1

        return self.state

    def get_initial_state(self):
        # 초기 상태를 0으로 설정
        return [0]*5

    def step(self, action):
        # Action 매핑
        actions = self.config['actions']

        path_choice, frame_type = actions[action]
        seq_num = self.sequence_numbers[self.current_step]
        frame_path = self.get_frame_path(seq_num)

        # 인코딩 (인코더 상태 유지)
        if frame_type == 'I-frame': # I-frame으로 인코딩
            encoded_data = self.encoder.encode_frame(frame_path, True)
        else: # P-frame으로 인코딩
            encoded_data = self.encoder.encode_frame(frame_path, False)

        # 네트워크 전송 시뮬레이션
        latencies = self.get_network_latencies(seq_num, path_choice)
        # 선택한 action에서, frame을 latency_threshold 내에 수신했으면 True
        frame_received = any(latency < self.latency_threshold for latency in latencies.values())

        # Latency 업데이트
        self.update_latencies(latencies)

        # Loss 여부 판단 - 못받았으면 True, 받았으면 False : 이건 설정하기 나름(State 정의)
        self.last_frame_loss = not frame_received

        # Reward 계산
        reward_ssim = self.get_reward_ssim(frame_received, encoded_data, frame_path)
        data_size = self.get_data_size(encoded_data, path_choice)
        reward = self.calc_reward(reward_ssim, data_size)

        # State 업데이트
        self.state = self.get_next_state()
    
        # Step 진행
        self.current_step += 1
        if self.current_step >= len(self.sequence_numbers):
            self.done = True

        return self.state, reward, self.done, {'seq_num': seq_num, 'ssim': reward_ssim, 'datasize': data_size}

    def calc_reward(self, reward_ssim, data_size):
        if reward_ssim > self.ssim_threshold:
            reward = math.log(reward_ssim + (1 - self.ssim_threshold)) + self.gamma_reward * math.log(self.max_datasize / data_size)
        else:
            reward = math.log(reward_ssim + (1 - self.ssim_threshold))
        return reward
    
    def get_reward_ssim(self, frame_received, encoded_data, frame_path):
        # 디코딩 및 SSIM 계산
        if frame_received:
            decoded_frame = self.decoder.decode_frame(encoded_data, len(encoded_data), 640, 480)
            # 디코딩이 성공한 경우에만 SSIM 계산
            if decoded_frame is not None and decoded_frame.size > 0:
                reward_ssim = self.calculate_ssim(frame_path, decoded_frame)
            # 디코딩 실패 시 SSIM을 0으로 처리
            else:
                reward_ssim = 0
        # frame이 수신되지 않은 경우
        else:
            reward_ssim = 0
        return reward_ssim
    
    def get_data_size(self, encoded_data, path_choice):
        data_size = len(encoded_data)
        if path_choice == 'Both':
            data_size *= 2
        return data_size

    def get_frame_path(self, seq_num):
        # 프레임 경로 반환
        for file in os.listdir(self.frames_dir):
            if file.startswith(f"{seq_num}_"):
                return os.path.join(self.frames_dir, file)
        return None
    
    def get_network_latencies(self, seq_num, path_choice):
        latencies = {'KT': float('inf'), 'LG': float('inf')}
        if path_choice in ['KT', 'Both']:
            kt_row = self.kt_log[self.kt_log['sequence_number'] == seq_num]
            if not kt_row.empty:
                latencies['KT'] = kt_row.iloc[0]['network_latency_ms']
        if path_choice in ['LG', 'Both']:
            lg_row = self.lg_log[self.lg_log['sequence_number'] == seq_num]
            if not lg_row.empty:
                latencies['LG'] = lg_row.iloc[0]['network_latency_ms']
        return latencies

    def update_latencies(self, latencies):
        # 'KT' 경로의 유효한 지연 값만 기록
        if latencies['KT'] != float('inf') and latencies['KT'] != 0:
            self.kt_last_latency = latencies['KT']
            self.kt_latency_history.append(self.kt_last_latency)
            if len(self.kt_latency_history) > 10:
                self.kt_latency_history.pop(0)
        
        # 'LG' 경로의 유효한 지연 값만 기록
        if latencies['LG'] != float('inf') and latencies['LG'] != 0:
            self.lg_last_latency = latencies['LG']
            self.lg_latency_history.append(self.lg_last_latency)
            if len(self.lg_latency_history) > 10:
                self.lg_latency_history.pop(0)

    def calculate_ssim(self, original_frame_path, decoded_frame):
        original = np.array(Image.open(original_frame_path).convert('L'))
        # decoded = decoded_frame  # 이미 numpy 배열로 전달됨
        # 디코딩된 프레임을 그레이스케일로 변환
        decoded_gray = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2GRAY)

        # SSIM 계산
        return ssim(original, decoded_gray)
    
    # def calculate_psnr(self, original_frame_path, decoded_frame):
    #     original = cv2.imread(original_frame_path)
    #     decoded = cv2.

    def get_next_state(self):
        # 최근 KT, LG Latency - 0 ~ 1 사이로 정규화
        kt_latency = min(self.kt_last_latency, self.max_latency) / self.max_latency
        lg_latency = min(self.lg_last_latency, self.max_latency) / self.max_latency
        state = [
            kt_latency,
            lg_latency,
        ]

        # 최근 10개 Latency의 평균 변화율 계산
        kt_change_rate = self.calculate_change_rate(self.kt_latency_history)
        lg_change_rate = self.calculate_change_rate(self.lg_latency_history)
        state.extend([kt_change_rate, lg_change_rate])

        # 최근 프레임의 Loss 여부
        state.append(int(self.last_frame_loss))

        return state

    def calculate_change_rate(self, latency_history):
        # 변화율 계산에 사용할 데이터 포인트 수
        n = len(latency_history) - 1
        
        # 유효한 데이터 포인트가 없으면 0을 반환
        if n <= 0:
            return 0

        # 각 연속된 값의 변화율을 계산
        change_rates = [
            (latency_history[i + 1] - latency_history[i]) / latency_history[i]
            for i in range(-n, 0)
        ]
        
        # 변화율의 평균을 반환
        return np.mean(change_rates)