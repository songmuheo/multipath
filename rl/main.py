# main.py
from environment.env import StreamingEnvironment
from agents.agent import DQNAgent
from utils.directory_manager import create_timestamped_directory
from utils.config import load_config
from train.train import train
from train.evaluate import evaluate
import numpy as np
# import sys
# import os

def main():

    # 결과를 저장할 폴더 생성
    results_dir = create_timestamped_directory()
    print(f"Results will be saved in: {results_dir}")

    # load config
    config_path = 'configs/config.yaml'
    config = load_config(config_path)

    env = StreamingEnvironment(config)
    state_size = len(env.reset())
    action_size = len(config['actions'])  # Action 개수
    agent = DQNAgent(state_size, action_size, config)
    mode = config['mode']

    if mode == "train":
        train(agent, env, config, results_dir)
    elif mode == "eval":
        evaluate(agent, env, config, results_dir)
        
if __name__ == '__main__':
    main()
