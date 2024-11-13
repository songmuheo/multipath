# utils/logger.py

import pandas as pd
import os
import yaml

class Logger:
    def __init__(self, results_dir, config):
        self.episode_logs = []
        self.step_logs = []
        self.results_dir = results_dir
        self.config = config
        self.logs_dir = os.path.join(self.results_dir, 'logs')

        # 설정 정보를 저장
        if self.config is not None:
            with open(os.path.join(self.logs_dir, 'config.yaml'), 'w') as f:
                yaml.dump(self.config, f)

        # 로그 디렉토리 설정
        self.episode_logs_dir = os.path.join(self.logs_dir, 'episode')
        self.step_logs_dir = os.path.join(self.logs_dir, 'step')
        os.makedirs(self.episode_logs_dir, exist_ok=True)
        os.makedirs(self.step_logs_dir, exist_ok=True)

    def log_step(self, data):
        self.step_logs.append(data)

    def log_episode(self, data):
        self.episode_logs.append(data)

    def save_logs(self, start_episode, end_episode):
        # 에피소드 로그 저장
        episode_log_df = pd.DataFrame(self.episode_logs)
        episode_log_filename = f'episode_logs_{start_episode}_{end_episode}.csv'
        episode_log_df.to_csv(os.path.join(self.episode_logs_dir, episode_log_filename), index=False)

        # 스텝 로그 저장
        step_log_df = pd.DataFrame(self.step_logs)
        step_log_filename = f'step_logs_{start_episode}_{end_episode}.csv'
        step_log_df.to_csv(os.path.join(self.step_logs_dir, step_log_filename), index=False)

        # 로그 리스트 초기화
        # 저장한 구간의 로그를 삭제하여 메모리 사용량을 관리합니다.
        self.episode_logs = []
        self.step_logs = []
