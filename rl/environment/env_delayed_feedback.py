# environment/env_delayed_feedback.py
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
        # 1) Hyperparameters & configurations
        self.initialize_hyperparameters()
        # 2) Episode tracking
        self.initialize_episode_tracking()
        # 3) Load network logs (KT, LG)
        self.load_network_logs()
        # 4) Encoder & Decoder
        self.encoder = Encoder(640, 480)
        self.decoder = Decoder()
        # 5) Prepare frame sequences
        self.prepare_frame_sequences()
        # 6) State variables & cache
        self.initialize_state_variables()
        # 7) Normalization parameters
        self.compute_normalization_params()
        # 8) Frame cache for SSIM calculation
        self.initialize_frame_cache()

        # Delayed feedback 관련: 
        # 전체 프레임 개수만큼 state를 저장할 버퍼
        self.delayed_state_buffer = [None] * len(self.sequence_numbers)
        # 학습 혹은 평가 구간에서 실제 step이 될 때 사용하는 index
        self.current_step = 0


    def initialize_hyperparameters(self):
        """Initialize hyperparameters from config."""
        self.latency_threshold = self.config['latency_threshold']
        self.ssim_threshold = self.config['ssim_threshold']
        self.gamma_reward = self.config['gamma_reward']
        self.history_length = self.config['history_length']
        self.actions = self.config['actions']
        self.state_num = self.config['state_num']

        # 새로 추가된 지연 피드백 스텝
        self.delay_steps = self.config.get('delay_steps', 2)  # 기본값 2

        # max_frames_since_iframe, max_frames_since_loss, datasize 범위
        self.max_frames_since_iframe = self.config['max_frames_since_iframe']
        self.max_frames_since_loss = self.config['max_frames_since_loss']
        self.min_datasize = self.config['min_datasize']
        self.max_datasize = self.config['max_datasize']

        # 데이터 분할 관련
        self.mode = self.config['mode']  # 'train' or 'eval'
        self.training_data_split = self.config['training_data_split']


    def initialize_episode_tracking(self):
        """Initialize episode and step counters."""
        self.current_episode = 0
        self.done = False

    def load_network_logs(self):
        """Load network logs from CSV files and set 'sequence_number' as index."""
        self.kt_log = pd.read_csv(self.config['kt_log_dir']).set_index('sequence_number')
        self.lg_log = pd.read_csv(self.config['lg_log_dir']).set_index('sequence_number')

    def prepare_frame_sequences(self):
        """Prepare frame sequences and create mappings from sequence numbers to frame paths."""
        self.frames_dir = self.config['frame_dir']
        self.sequence_numbers, self.seq_num_to_frame_path = self.get_sequence_numbers()
        self.adjust_sequence_numbers()  # train/eval split

    def initialize_frame_cache(self):
        """Initialize frame cache for original grayscale frames (for SSIM)."""
        self.frame_cache = {}

    def initialize_state_variables(self):
        """Initialize state-related variables."""
        # step, state 등은 reset()에서 다시 지정
        self.state = None
        self.kt_latency_history = []
        self.lg_latency_history = []
        self.kt_packet_loss_history = []
        self.lg_packet_loss_history = []
        self.datasize_history = []

        self.kt_last_latency = 0
        self.lg_last_latency = 0
        self.last_encoding_type = 0  # 0: I-frame, 1: P-frame
        self.frames_since_last_iframe = 0
        self.last_datasize = 0
        # self.last_ssim_value = 0
        self.frames_since_last_loss = 0

    def compute_normalization_params(self):
        """Compute normalization parameters for latency and variance."""
        self.latency_values = self.get_all_latency_values()
        # latency
        self.min_latency = np.min(self.latency_values)
        self.max_latency = np.percentile(self.latency_values, 99)  # 99th percentile
        # variance
        variance_values = self.compute_latency_variances()
        self.min_variance = np.min(variance_values) if len(variance_values) > 0 else 0
        self.max_variance = np.percentile(variance_values, 98) if len(variance_values) > 0 else 1

    def get_all_latency_values(self):
        """Retrieve all latency values from network logs."""
        kt_latencies = self.kt_log['network_latency_ms'].values
        lg_latencies = self.lg_log['network_latency_ms'].values
        return np.concatenate([kt_latencies, lg_latencies])

    def compute_latency_variances(self):
        """Compute variance of latencies over a sliding window."""
        latency_values = self.latency_values
        window_size = self.history_length
        if len(latency_values) < window_size:
            return np.array([])
        variance_values = np.array([
            np.var(latency_values[i:i+window_size]) 
            for i in range(len(latency_values) - window_size + 1)
        ])
        return variance_values

    def get_sequence_numbers(self):
        """Extract sequence numbers and create mapping from frame filenames."""
        files = os.listdir(self.frames_dir)
        seq_num_to_frame_path = {}
        for f in files:
            seq_num = int(f.split('_')[0])
            seq_num_to_frame_path[seq_num] = os.path.join(self.frames_dir, f)
        seq_nums = sorted(seq_num_to_frame_path.keys())
        return seq_nums, seq_num_to_frame_path

    def adjust_sequence_numbers(self):
        """Adjust sequence numbers based on mode (train/eval)."""
        total_sequences = len(self.sequence_numbers)
        split_index = int(total_sequences * self.training_data_split)

        if self.mode == 'train':
            self.sequence_numbers = self.sequence_numbers[:split_index]
        elif self.mode == 'eval':
            # warmup을 위해 뒤에 추가로 빼줌 (training 부분을 warm up을 위한 부분으로 사용 - 초기 state)
            self.sequence_numbers = self.sequence_numbers[split_index- self.history_length - self.delay_steps:]
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

    def reset(self):
        """
        Reset the environment.

        - warm-up 단계에서 history_length + delay_steps 만큼 내부 업데이트를 수행해
          충분한 state 기록을 쌓고,
        - 마지막 warm-up 프레임은 강제 I-frame으로 인코딩하여 decoder 초기화.
        - 그 이후 current_step 부터 실제 RL의 step 진행.

        Returns:
            state (list): delay_steps만큼 지연된(또는 초기값) state 반환
        """
        self.done = False
        self.current_episode += 1

        # 인코더 / 디코더 / 히스토리 변수들 초기화
        self.encoder.reset()
        self.decoder.reset()
        self.initialize_state_variables()

        # delayed_state_buffer 초기화 (에피소드마다 새롭게)
        self.delayed_state_buffer = [None] * len(self.sequence_numbers)

        # warm-up (0 ~ warmup_end-1)
        warmup_end = self.history_length + self.delay_steps
        if warmup_end > len(self.sequence_numbers):
            warmup_end = len(self.sequence_numbers)

        # warm-up 단계: 임의 액션으로 내부적으로 step 진행해 state 값 갱신
        # (학습에 사용되지 않음)
        for i in range(warmup_end):
            # 마지막 warm-up 프레임은 강제 I-frame
            # if i == 0 or i == (warmup_end - 1):
            #     force_i_frame = True
            # else:
            #     force_i_frame = False
            force_i_frame = True
            default_action = self.get_default_action(force_i_frame=force_i_frame)

            # 실제 프레임 처리 (네트워크 시뮬레이션 등)
            self.process_frame(action=default_action, step_index=i)

            # **process_frame** 후, 현재 환경 내부 상태를 build
            current_state = self.build_current_state()
            self.delayed_state_buffer[i] = current_state

            # warm-up 후, 실제 학습/평가를 시작할 step 설정
            self.current_step = warmup_end

        # 초기 state: delay_steps 전의 state가 유효하다면 그걸 쓰고,
        # 유효하지 않다면 그냥 0으로 된 state를 준다.
        if self.current_step - self.delay_steps >= 0:
            init_state = self.delayed_state_buffer[self.current_step - self.delay_steps]
        else:
            init_state = self.get_initial_state()

        return init_state

    def get_default_action(self, force_i_frame=True):
        """
        Warm-up 단계에서 사용할 기본 액션.
        - vector 모드라면 (path_index, frame_index) 튜플을 반환합니다.
        - scalar 모드라면 기존과 같이 정수 인덱스를 반환합니다.
        """
        if self.config.get('action_type', 'scalar') == 'vector':
            # 예시: 기본 액션으로 'Both'와 I-frame을 사용
            # config에서 vector 모드라면, 예를 들어
            #   actions:
            #       path: [KT, LG, Both]
            #       frame: [I-frame, P-frame]
            # 라고 정의되어 있을 때,
            # force_i_frame True이면 (Both, I-frame) => (2, 0)
            # False이면 (Both, P-frame) => (2, 1) 로 설정
            if force_i_frame:
                return (2, 0)
            else:
                return (2, 1)
        # 예시: 항상 KT + I-frame (강제) or KT + P-frame
        # 실제로는 필요 시 바꿔도 무방
        if force_i_frame:
            return 4  # 예: 0번 액션이 (Both, I-frame) 이라고 가정
        else:
            return 5  # 예: 1번 액션이 (Both, P-frame) 이라고 가정

    def get_initial_state(self):
        """Return a dummy initial state (all zeros)."""
        return [0.0] * self.state_num

    def step(self, action):
        """
        실제 RL에서 호출되는 step 함수.
        - 현재 step에 해당하는 frame을 action에 따라 인코딩/전송/디코딩
        - reward 계산
        - "지연 피드백" 적용: next_state는 (current_step - delay_steps)에 저장해둔 state 사용
        """
        if self.done:
            # 이미 에피소드가 끝났으면 더 이상 step 진행 안 함
            return self.get_initial_state(), 0.0, True, {}

        # 현재 step에 대하여 환경 로직(인코딩, 네트워크 시뮬 등) 수행
        seq_num, reward, frame_received, frame_type_str = self.process_frame(action, self.current_step)

        # 이 step에 대해 새로 계산된 state를 delayed_state_buffer에 저장
        current_state = self.build_current_state()
        self.delayed_state_buffer[self.current_step] = current_state

        # 다음 step에서 에이전트에게 줄 state = delay_steps 전에 미리 저장된 state
        # 만약 아직 buffer에 없다면(초기 구간), 0으로 된 state 반환
        next_state_idx = self.current_step - self.delay_steps
        if next_state_idx >= 0:
            next_state = self.delayed_state_buffer[next_state_idx]
        else:
            next_state = self.get_initial_state()

        # step 증가 및 done 여부 판단
        self.current_step += 1
        if self.current_step >= len(self.sequence_numbers):
            self.done = True

        info = {
            'seq_num': seq_num,
            'ssim': self.last_ssim_value,
            'datasize': self.last_datasize,
            'frame_loss': frame_received,
            'frame_type': frame_type_str
        }

        return next_state, reward, self.done, info

    def process_frame(self, action, step_index):
        """
        주어진 step_index의 frame에 대해,
        (1) 액션(action) -> (path_choice, frame_type) 매핑
        (2) 인코딩 (I/P)
        (3) 전송 (네트워크 지연/패킷손실 시뮬레이션)
        (4) 디코딩 -> SSIM/데이터 사이즈/보상 계산
        (5) 내부 state 변수 갱신
        """
        seq_num = self.sequence_numbers[step_index]
        frame_path = self.get_frame_path(seq_num)

        path_choice, frame_type = self.map_action(action)
        is_i_frame = (frame_type == 'I-frame')

        # 인코딩 타입, I-frame 이후 경과 프레임 수 갱신
        self.update_encoding_type(is_i_frame)

        # 인코딩
        encoded_data = self.encoder.encode_frame(frame_path, is_i_frame)

        # encoded_data를 이용해 프레임 타입 추출
        frame_type_str = self.get_frame_type(encoded_data)

        # 네트워크 전송 시뮬
        latencies = self.simulate_network_transmission(seq_num, path_choice)

        # 프레임 수신 여부 판정
        frame_received = self.determine_frame_reception(latencies, path_choice)

        # 히스토리 업데이트 (지연/패킷손실 등)
        self.update_histories(latencies, path_choice)

        # 보상 계산
        reward = self.calculate_reward(frame_received, encoded_data, frame_path, path_choice)

        self.datasize_history.append(self.last_datasize)
        if len(self.datasize_history) > self.history_length:
            self.datasize_history.pop(0)

        return seq_num, reward, frame_received, frame_type_str

    def build_current_state(self):
        """
        환경 내부 변수들을 바탕으로 state 벡터를 생성한다.
        (기존 get_next_state()와 동일한 로직)
        """
        state = []

        # 1) LG 최근 지연시간 (정규화)
        state.append(self.normalize_latency(self.lg_last_latency))
        # 2) KT 최근 지연시간 (정규화)
        state.append(self.normalize_latency(self.kt_last_latency))
        # 3) LG 최근 N 스텝 평균 지연시간
        lg_mean = np.mean(self.lg_latency_history) if self.lg_latency_history else 0.0
        state.append(self.normalize_latency(lg_mean))
        # 4) KT 최근 N 스텝 평균 지연시간
        kt_mean = np.mean(self.kt_latency_history) if self.kt_latency_history else 0.0
        state.append(self.normalize_latency(kt_mean))
        # 5) LG 최근 N 스텝 분산
        lg_var = np.var(self.lg_latency_history) if len(self.lg_latency_history) > 1 else 0.0
        state.append(self.normalize_variance(lg_var))
        # 6) KT 최근 N 스텝 분산
        kt_var = np.var(self.kt_latency_history) if len(self.kt_latency_history) > 1 else 0.0
        state.append(self.normalize_variance(kt_var))
        # 7) LG 최근 N 스텝 패킷 손실률
        lg_loss_rate = np.mean(self.lg_packet_loss_history) if self.lg_packet_loss_history else 0.0
        state.append(lg_loss_rate)
        # 8) KT 최근 N 스텝 패킷 손실률
        kt_loss_rate = np.mean(self.kt_packet_loss_history) if self.kt_packet_loss_history else 0.0
        state.append(kt_loss_rate)
        # 9) 직전 프레임 인코딩 타입 (0: I-frame, 1: P-frame)
        state.append(self.last_encoding_type)
        # 10) 마지막 I-frame 이후 지난 프레임 수 (정규화)
        state.append(min(self.frames_since_last_iframe / self.max_frames_since_iframe, 1.0))
        # 11) 최근 전송 프레임 데이터 사이즈 (정규화)
        state.append(self.normalize_datasize(self.last_datasize))
        # 12) 최근 N 스텝 평균 데이터 사이즈 (정규화)
        recent_mean_datasize = np.mean(self.datasize_history) if self.datasize_history else 0.0
        state.append(self.normalize_datasize(recent_mean_datasize))
        # 13) 마지막 손실 이후 지난 프레임 수 (정규화)
        state.append(min(self.frames_since_last_loss / self.max_frames_since_loss, 1.0))

        return state

    def map_action(self, action):
        """
        만약 action이 튜플이면 벡터 방식으로 처리하고,
        그렇지 않으면 기존 스칼라 방식으로 처리합니다.
        """
        if self.config['action_type']=='vector':            
            path_index, frame_index = action
            path_choice = self.config['actions_vec']['path'][path_index]
            frame_type = self.config['actions_vec']['frame'][frame_index]
            return path_choice, frame_type
        else:
            return self.actions[action]

    def update_encoding_type(self, is_i_frame):
        """Update encoding type and frames since last I-frame."""
        self.last_encoding_type = 0 if is_i_frame else 1  # 0: I-frame, 1: P-frame
        if is_i_frame:
            self.frames_since_last_iframe = 0
        else:
            self.frames_since_last_iframe += 1

    def simulate_network_transmission(self, seq_num, path_choice):
        """Simulate network transmission and get latencies."""
        return self.get_network_latencies(seq_num, path_choice)

    def determine_frame_reception(self, latencies, path_choice):
        """Determine if the frame was received within latency threshold."""
        kt_received = False
        lg_received = False
        if path_choice in ['KT', 'Both']:
            kt_received = latencies['KT'] < self.latency_threshold
        if path_choice in ['LG', 'Both']:
            lg_received = latencies['LG'] < self.latency_threshold

        frame_received = kt_received or lg_received
        if frame_received:
            self.frames_since_last_loss += 1
        else:
            self.frames_since_last_loss = 0

        return frame_received

    def update_histories(self, latencies, path_choice):
        """Update latency and packet loss histories."""
        self.update_latency_histories(latencies, path_choice)
        self.update_packet_loss_histories(latencies, path_choice)

    def update_latency_histories(self, latencies, path_choice):
        """Update latency histories for KT and LG."""
        if path_choice in ['KT', 'Both']:
            if not np.isinf(latencies['KT']) and latencies['KT'] > 0:
                self.kt_last_latency = latencies['KT']
                self.kt_latency_history.append(self.kt_last_latency)
                if len(self.kt_latency_history) > self.history_length:
                    self.kt_latency_history.pop(0)

        if path_choice in ['LG', 'Both']:
            if not np.isinf(latencies['LG']) and latencies['LG'] > 0:
                self.lg_last_latency = latencies['LG']
                self.lg_latency_history.append(self.lg_last_latency)
                if len(self.lg_latency_history) > self.history_length:
                    self.lg_latency_history.pop(0)

    def update_packet_loss_histories(self, latencies, path_choice):
        """Update packet loss histories for KT and LG."""
        kt_received = latencies['KT'] < self.latency_threshold if path_choice in ['KT', 'Both'] else False
        lg_received = latencies['LG'] < self.latency_threshold if path_choice in ['LG', 'Both'] else False

        if path_choice in ['KT', 'Both']:
            self.kt_packet_loss_history.append(1 if kt_received else 0)
            if len(self.kt_packet_loss_history) > self.history_length:
                self.kt_packet_loss_history.pop(0)

        if path_choice in ['LG', 'Both']:
            self.lg_packet_loss_history.append(1 if lg_received else 0)
            if len(self.lg_packet_loss_history) > self.history_length:
                self.lg_packet_loss_history.pop(0)

    def calculate_reward(self, frame_received, encoded_data, frame_path, path_choice):
        """Calculate the reward based on SSIM and datasize."""
        ssim_value = self.compute_ssim(frame_received, encoded_data, frame_path)
        datasize = self.compute_datasize(encoded_data, path_choice)
        self.last_datasize = datasize
        self.last_ssim_value = ssim_value
        reward = self.compute_reward_function(ssim_value, datasize)
        return reward

    def compute_ssim(self, frame_received, encoded_data, frame_path):
        """Compute the SSIM value for the received frame."""
        if frame_received:
            decoded_frame = self.decoder.decode_frame(encoded_data, len(encoded_data), 640, 480)
            if decoded_frame is not None and decoded_frame.size > 0:
                ssim_value = self.calculate_ssim(frame_path, decoded_frame)
            else:
                ssim_value = 0.0
        else:
            ssim_value = 0.0
        return ssim_value

    def compute_datasize(self, encoded_data, path_choice):
        """Compute the datasize based on path choice."""
        datasize = len(encoded_data)
        if path_choice == 'Both':
            datasize *= 2
        return datasize

    def compute_reward_function(self, ssim_value, datasize):
        """Compute the reward using SSIM value and datasize."""
        if ssim_value > self.ssim_threshold:
            reward = math.log(ssim_value + (1 - self.ssim_threshold)) + \
                     self.gamma_reward * math.log(self.max_datasize / datasize)
        else:
            reward = math.log(ssim_value + (1 - self.ssim_threshold))
        return reward

    def get_network_latencies(self, seq_num, path_choice):
        """Retrieve network latencies for KT and LG."""
        latencies = {'KT': float('inf'), 'LG': float('inf')}
        if path_choice in ['KT', 'Both']:
            if seq_num in self.kt_log.index:
                latencies['KT'] = self.kt_log.loc[seq_num]['network_latency_ms']
        if path_choice in ['LG', 'Both']:
            if seq_num in self.lg_log.index:
                latencies['LG'] = self.lg_log.loc[seq_num]['network_latency_ms']
        return latencies

    def calculate_ssim(self, original_frame_path, decoded_frame):
        """Calculate SSIM between original and decoded frames (grayscale)."""
        seq_num = int(os.path.basename(original_frame_path).split('_')[0])
        if seq_num in self.frame_cache:
            original = self.frame_cache[seq_num]
        else:
            # original = np.array(Image.open(original_frame_path).convert('L'))
            original = cv2.imread(original_frame_path, cv2.IMREAD_GRAYSCALE)
            self.frame_cache[seq_num] = original
        decoded_gray = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2GRAY)
        return ssim(original, decoded_gray)

    def normalize_latency(self, latency):
        """Normalize latency between 0 and 1."""
        if (self.max_latency - self.min_latency) == 0:
            return 0.0
        clipped_latency = np.clip(latency, self.min_latency, self.max_latency)
        normalized = (clipped_latency - self.min_latency) / (self.max_latency - self.min_latency)
        return normalized

    def normalize_variance(self, variance):
        """Normalize variance between 0 and 1."""
        if (self.max_variance - self.min_variance) == 0:
            return 0.0
        clipped_variance = np.clip(variance, self.min_variance, self.max_variance)
        normalized = (clipped_variance - self.min_variance) / (self.max_variance - self.min_variance)
        return normalized

    def normalize_datasize(self, datasize):
        """Normalize datasize between 0 and 1."""
        if (self.max_datasize - self.min_datasize) == 0:
            return 0.0
        clipped_size = np.clip(datasize, self.min_datasize, self.max_datasize)
        normalized = (clipped_size - self.min_datasize) / (self.max_datasize - self.min_datasize)
        return normalized

    def get_frame_path(self, seq_num):
        """Return the frame path for the given sequence number."""
        return self.seq_num_to_frame_path.get(seq_num, None)
