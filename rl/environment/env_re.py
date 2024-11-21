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
        # Prepare frame sequences
        self.prepare_frame_sequences()
        # Initialize state variables
        self.initialize_state_variables()
        # Compute normalization parameters
        self.compute_normalization_params()

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
        """Load network logs from CSV files."""
        self.kt_log = pd.read_csv(self.config['kt_log_dir'])
        self.lg_log = pd.read_csv(self.config['lg_log_dir'])

    def prepare_frame_sequences(self):
        """Prepare frame sequences based on mode (train/eval)."""
        self.frames_dir = self.config['frame_dir']
        self.sequence_numbers = self.get_sequence_numbers()
        self.adjust_sequence_numbers()

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
        self.max_latency = np.percentile(self.latency_values, 95)  # 95th percentile

        self.min_datasize = self.config['min_datasize']
        self.max_datasize = self.config['max_datasize']

        variance_values = self.compute_latency_variances()
        self.min_variance = np.min(variance_values)
        self.max_variance = np.max(variance_values)
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
        variance_list = []

        for i in range(len(latency_values) - window_size + 1):
            window = latency_values[i:i+window_size]
            variance = np.var(window)
            variance_list.append(variance)

        variance_values = np.array(variance_list)
        return variance_values

    def get_sequence_numbers(self):
        """Extract sequence numbers from frame filenames."""
        files = os.listdir(self.frames_dir)
        seq_nums = [int(f.split('_')[0]) for f in files]
        return sorted(seq_nums)

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
        self.update_histories(latencies, frame_received, path_choice)
        # Calculate reward
        reward = self.calculate_reward(frame_received, encoded_data, frame_path, path_choice)
        # Update state
        self.state = self.get_next_state()
        # Increment step and check if done
        self.increment_step()
        # Return state, reward, done, info
        info = {'seq_num': seq_num, 'ssim': self.last_ssim_value, 'datasize': self.last_datasize}
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

    def update_histories(self, latencies, frame_received, path_choice):
        """Update latency and packet loss histories."""
        self.update_latency_histories(latencies, path_choice)
        self.update_packet_loss_histories(latencies, frame_received, path_choice)

    def update_latency_histories(self, latencies, path_choice):
        """Update latency histories for KT and LG."""
        if path_choice in ['KT', 'Both']:
            if latencies['KT'] != float('inf') and latencies['KT'] > 0:
                self.kt_last_latency = latencies['KT']
                self.kt_latency_history.append(self.kt_last_latency)
                if len(self.kt_latency_history) > self.history_length:
                    self.kt_latency_history.pop(0)
        if path_choice in ['LG', 'Both']:
            if latencies['LG'] != float('inf') and latencies['LG'] > 0:
                self.lg_last_latency = latencies['LG']
                self.lg_latency_history.append(self.lg_last_latency)
                if len(self.lg_latency_history) > self.history_length:
                    self.lg_latency_history.pop(0)

    def update_packet_loss_histories(self, latencies, frame_received, path_choice):
        """Update packet loss histories for KT and LG."""
        kt_received = latencies['KT'] < self.latency_threshold if path_choice in ['KT', 'Both'] else False
        lg_received = latencies['LG'] < self.latency_threshold if path_choice in ['LG', 'Both'] else False
        if path_choice in ['KT', 'Both']:
            self.kt_packet_loss_history.append(0 if kt_received else 1)
            if len(self.kt_packet_loss_history) > self.history_length:
                self.kt_packet_loss_history.pop(0)
        if path_choice in ['LG', 'Both']:
            self.lg_packet_loss_history.append(0 if lg_received else 1)
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
        for file in os.listdir(self.frames_dir):
            if file.startswith(f"{seq_num}_"):
                return os.path.join(self.frames_dir, file)
        return None

    def get_network_latencies(self, seq_num, path_choice):
        """Retrieve network latencies for KT and LG."""
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

    def calculate_ssim(self, original_frame_path, decoded_frame):
        """Calculate SSIM between original and decoded frames."""
        original = np.array(Image.open(original_frame_path).convert('L'))
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
        clipped_latency = np.clip(latency, min_latency, max_latency)
        normalized = (clipped_latency - min_latency) / (max_latency - min_latency)
        return normalized

    def normalize_variance(self, variance):
        """Normalize variance between 0 and 1."""
        min_variance = self.min_variance
        max_variance = self.max_variance
        clipped_variance = np.clip(variance, min_variance, max_variance)
        normalized = (clipped_variance - min_variance) / (max_variance - min_variance)
        return normalized

    def normalize_datasize(self, datasize):
        """Normalize datasize between 0 and 1."""
        min_size = self.min_datasize
        max_size = self.max_datasize
        clipped_size = np.clip(datasize, min_size, max_size)
        normalized = (clipped_size - min_size) / (max_size - min_size)
        return normalized