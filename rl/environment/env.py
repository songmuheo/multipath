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
        # Initialize hyperparameters and configurations
        self.initialize_hyperparameters()
        # Initialize episode and step tracking
        self.initialize_episode_tracking()
        # Load network logs
        self.load_network_logs()
        # Initialize encoder and decoder
        self.encoder = Encoder(640, 480)
        self.decoder = Decoder()
        # Prepare frame sequences and mappings
        self.prepare_frame_sequences()
        # Initialize state variables
        self.initialize_state_variables()
        # Compute normalization parameters
        self.compute_normalization_params()
        # Initialize frame cache
        self.initialize_frame_cache()

    def initialize_hyperparameters(self):
        """Initialize hyperparameters from config."""
        self.latency_threshold = self.config['latency_threshold']
        self.ssim_threshold = self.config['ssim_threshold']
        self.gamma_reward = self.config['gamma_reward']
        self.history_length = self.config['history_length']
        self.actions = self.config['actions']
        self.state_num = self.config['state_num']

    def initialize_episode_tracking(self):
        """Initialize episode and step counters."""
        self.current_episode = 0
        self.current_step = 0
        self.done = False

    def load_network_logs(self):
        """Load network logs from CSV files and set 'sequence_number' as index."""
        self.kt_log = pd.read_csv(self.config['kt_log_dir']).set_index('sequence_number')
        self.lg_log = pd.read_csv(self.config['lg_log_dir']).set_index('sequence_number')

    def prepare_frame_sequences(self):
        """Prepare frame sequences and create mappings from sequence numbers to frame paths."""
        self.frames_dir = self.config['frame_dir']
        self.sequence_numbers, self.seq_num_to_frame_path = self.get_sequence_numbers()
        self.adjust_sequence_numbers()

    def initialize_frame_cache(self):
        """Initialize frame cache for original grayscale frames."""
        self.frame_cache = {}

    def initialize_state_variables(self):
        """Initialize state variables for the environment."""
        self.state = None
        self.kt_latency_history = []
        self.lg_latency_history = []
        self.kt_packet_loss_history = []
        self.lg_packet_loss_history = []
        self.kt_last_latency = 0
        self.lg_last_latency = 0
        self.last_encoding_type = 0  # 0: I-frame, 1: P-frame
        self.frames_since_last_iframe = 0
        self.last_datasize = 0
        self.last_ssim_value = 0
        self.frames_since_last_loss = 0

    def compute_normalization_params(self):
        """Compute normalization parameters for latency and variance."""
        self.latency_values = self.get_all_latency_values()
        self.min_latency = np.min(self.latency_values)
        self.max_latency = np.percentile(self.latency_values, 99)  # 99th percentile
        self.min_datasize = self.config['min_datasize']
        self.max_datasize = self.config['max_datasize']

        variance_values = self.compute_latency_variances()
        self.min_variance = np.min(variance_values) if len(variance_values) > 0 else 0
        self.max_variance = np.max(variance_values) if len(variance_values) > 0 else 1
        self.max_frames_since_iframe = self.config['max_frames_since_iframe']
        self.max_frames_since_loss = self.config['max_frames_since_loss']

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
        variance_values = np.array([np.var(latency_values[i:i+window_size]) for i in range(len(latency_values)-window_size+1)])
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
        mode = self.config['mode']
        total_sequences = len(self.sequence_numbers)
        split_index = int(total_sequences * self.config['training_data_split'])
        if mode == 'train':
            self.sequence_numbers = self.sequence_numbers[:split_index]
        elif mode == 'eval':
            self.sequence_numbers = self.sequence_numbers[split_index:]

    def reset(self):
        """Reset the environment to the initial state."""
        self.current_step = 0
        self.done = False
        self.encoder.reset()
        self.decoder.reset()
        self.initialize_state_variables()
        self.current_episode += 1

        """
        Decoder에 처음에 P-frame을 넣으면 에러가 발생하기 때문에, 에피소드 맨 처음에는 I-frame을 인코딩 및 디코딩 해줌
        """
        # 첫 번째 프레임을 I-frame으로 인코딩 및 디코딩하여 디코더에 참조 프레임 설정
        seq_num, frame_path = self.get_current_frame_path()
        is_i_frame = True  # 첫 번째 프레임은 강제로 I-frame으로 설정
        encoded_data = self.encoder.encode_frame(frame_path, is_i_frame)
        decoded_frame = self.decoder.decode_frame(encoded_data, len(encoded_data), 640, 480)

        # 현재 스텝을 1로 설정하여 다음 프레임부터 에이전트의 액션을 적용
        self.current_step = 1

        return self.get_initial_state()

    def get_initial_state(self):
        """Return the initial state."""
        return [0.0] * self.state_num

    def step(self, action):
        """Take an action in the environment."""
        # Map action to path choice and frame type
        path_choice, frame_type = self.map_action(action)
        seq_num, frame_path = self.get_current_frame_path()
        is_i_frame = (frame_type == 'I-frame')
        # Update encoding type and frames since last I-frame
        self.update_encoding_type(is_i_frame)
        # Encode frame
        encoded_data = self.encode_frame(frame_path, is_i_frame)
        # Simulate network transmission
        latencies = self.simulate_network_transmission(seq_num, path_choice)
        # Determine frame reception
        frame_received = self.determine_frame_reception(latencies, path_choice)
        # Update histories
        self.update_histories(latencies, path_choice)
        # Calculate reward
        reward = self.calculate_reward(frame_received, encoded_data, frame_path, path_choice)
        # Update state
        self.state = self.get_next_state()
        # Increment step and check if done
        self.increment_step()

        # print(f'\n\n{len(self.kt_packet_loss_history)}\n \
        #       {self.lg_packet_loss_history}\n')
        # Return state, reward, done, info
        info = {'seq_num': seq_num, 'ssim': self.last_ssim_value, 'datasize': self.last_datasize, 'frame_loss': frame_received}
        return self.state, reward, self.done, info

    def map_action(self, action):
        """Map action index to path choice and frame type."""
        return self.actions[action]

    def get_current_frame_path(self):
        """Get the current frame's sequence number and path."""
        seq_num = self.sequence_numbers[self.current_step]
        frame_path = self.get_frame_path(seq_num)
        return seq_num, frame_path

    def update_encoding_type(self, is_i_frame):
        """Update encoding type and frames since last I-frame."""
        self.last_encoding_type = 0 if is_i_frame else 1  # 0: I-frame, 1: P-frame
        if is_i_frame:
            self.frames_since_last_iframe = 0
        else:
            self.frames_since_last_iframe += 1

    def encode_frame(self, frame_path, is_i_frame):
        """Encode the frame using the encoder."""
        return self.encoder.encode_frame(frame_path, is_i_frame)

    def simulate_network_transmission(self, seq_num, path_choice):
        """Simulate network transmission and get latencies."""
        return self.get_network_latencies(seq_num, path_choice)

    def determine_frame_reception(self, latencies, path_choice):
        """Determine if the frame was received within latency threshold."""
        # Determine per-path frame reception
        kt_received = False
        lg_received = False
        if path_choice in ['KT', 'Both']:
            kt_received = latencies['KT'] < self.latency_threshold
        if path_choice in ['LG', 'Both']:
            lg_received = latencies['LG'] < self.latency_threshold
        frame_received = kt_received or lg_received
        # Update frames since last loss
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
        self.last_datasize = datasize  # State update
        self.last_ssim_value = ssim_value  # State update
        reward = self.compute_reward(ssim_value, datasize)
        return reward

    def compute_reward(self, ssim_value, datasize):
        """Compute the reward using SSIM value and datasize."""
        if ssim_value > self.ssim_threshold:
            reward = math.log(ssim_value + (1 - self.ssim_threshold)) + \
                     self.gamma_reward * math.log(self.max_datasize / datasize)
        else:
            reward = math.log(ssim_value + (1 - self.ssim_threshold))
        return reward

    def compute_ssim(self, frame_received, encoded_data, frame_path):
        """Compute the SSIM value for the received frame."""
        if frame_received:
            decoded_frame = self.decoder.decode_frame(encoded_data, len(encoded_data), 640, 480)
            if decoded_frame is not None and decoded_frame.size > 0:
                ssim_value = self.calculate_ssim(frame_path, decoded_frame)
            else:
                ssim_value = 0
        else:
            ssim_value = 0
        return ssim_value

    def compute_datasize(self, encoded_data, path_choice):
        """Compute the datasize based on path choice."""
        datasize = len(encoded_data)
        if path_choice == 'Both':
            datasize *= 2
        return datasize

    def increment_step(self):
        """Increment the current step and check if done."""
        self.current_step += 1
        if self.current_step >= len(self.sequence_numbers):
            self.done = True

    def get_frame_path(self, seq_num):
        """Return the frame path for the given sequence number."""
        return self.seq_num_to_frame_path.get(seq_num, None)

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
        """Calculate SSIM between original and decoded frames."""
        seq_num = int(os.path.basename(original_frame_path).split('_')[0])
        if seq_num in self.frame_cache:
            original = self.frame_cache[seq_num]
        else:
            original = np.array(Image.open(original_frame_path).convert('L'))
            self.frame_cache[seq_num] = original
        decoded_gray = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2GRAY)
        return ssim(original, decoded_gray)

    def get_next_state(self):
        """Construct the next state vector."""
        state = []
        # Variable 1: Most recent LG latency
        state.append(self.normalize_latency(self.lg_last_latency))
        # Variable 2: Most recent KT latency
        state.append(self.normalize_latency(self.kt_last_latency))
        # Variable 3: Average of last N LG latencies
        state.append(self.normalize_latency(np.mean(self.lg_latency_history)) if self.lg_latency_history else 0.0)
        # Variable 4: Average of last N KT latencies
        state.append(self.normalize_latency(np.mean(self.kt_latency_history)) if self.kt_latency_history else 0.0)
        # Variable 5: Variance of last N LG latencies
        state.append(self.normalize_variance(np.var(self.lg_latency_history)) if len(self.lg_latency_history) > 1 else 0.0)
        # Variable 6: Variance of last N KT latencies
        state.append(self.normalize_variance(np.var(self.kt_latency_history)) if len(self.kt_latency_history) > 1 else 0.0)
        # Variable 7: Packet loss rate of last N LG frames
        state.append(np.mean(self.lg_packet_loss_history) if self.lg_packet_loss_history else 0.0)
        # Variable 8: Packet loss rate of last N KT frames
        state.append(np.mean(self.kt_packet_loss_history) if self.kt_packet_loss_history else 0.0)
        # Variable 9: Encoding type of previous frame (0: I-frame, 1: P-frame)
        state.append(self.last_encoding_type)
        # Variable 10: Number of frames since last I-frame (normalized)
        state.append(min(self.frames_since_last_iframe / self.max_frames_since_iframe, 1.0))
        # Variable 11: Data size of recently sent frame (normalized)
        state.append(self.normalize_datasize(self.last_datasize))
        # Variable 12: SSIM value of recent frame at receiver
        state.append(self.last_ssim_value)
        # Variable 13: Count since last frame loss (normalized)
        state.append(min(self.frames_since_last_loss / self.max_frames_since_loss, 1.0))
        return state

    def normalize_latency(self, latency):
        """Normalize latency between 0 and 1."""
        min_latency = self.min_latency
        max_latency = self.max_latency
        if max_latency - min_latency == 0:
            return 0.0
        clipped_latency = np.clip(latency, min_latency, max_latency)
        normalized = (clipped_latency - min_latency) / (max_latency - min_latency)
        return normalized

    def normalize_variance(self, variance):
        """Normalize variance between 0 and 1."""
        min_variance = self.min_variance
        max_variance = self.max_variance
        if max_variance - min_variance == 0:
            return 0.0
        clipped_variance = np.clip(variance, min_variance, max_variance)
        normalized = (clipped_variance - min_variance) / (max_variance - min_variance)
        return normalized

    def normalize_datasize(self, datasize):
        """Normalize datasize between 0 and 1."""
        min_size = self.min_datasize
        max_size = self.max_datasize
        if max_size - min_size == 0:
            return 0.0
        clipped_size = np.clip(datasize, min_size, max_size)
        normalized = (clipped_size - min_size) / (max_size - min_size)
        return normalized
