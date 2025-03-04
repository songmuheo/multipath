# main.py

import importlib
from environment.env_delayed_feedback import StreamingEnvironment
from utils.directory_manager import create_timestamped_directory
from utils.config import load_config

def main():
    # load config
    config_path = 'configs/config.yaml'
    config = load_config(config_path)
    algorithm = config['algorithm']  # 'DQN' 또는 'PPO'

    env = StreamingEnvironment(config)
    state_size = len(env.reset())


    # config에 지정된 action_type에 따라 에이전트 분기
    action_type = config['action_type']
    if action_type == 'scalar':
        # 스칼라 방식: 기존 config['actions']는 dict형태(인덱스: [path, frame])임
        action_size = len(config['actions'])
        agent_module = importlib.import_module(f"agents.{algorithm.lower()}_agent")
    elif action_type == 'vector':
        # 벡터 방식: config['actions']는 { 'path': [...], 'frame': [...] } 구조임
        path_size = len(config['actions_vec']['path'])
        frame_size = len(config['actions_vec']['frame'])
        action_size = (path_size, frame_size)
        agent_module = importlib.import_module(f"agents.{algorithm.lower()}_agent_vector")
    else:
        raise ValueError("Unsupported action_type in config. Use 'scalar' or 'vector'.")

    # action_size = len(config['actions'])  # Action 개수

    # 에이전트 동적 임포트 및 인스턴스화
    # agent_module = importlib.import_module(f"agents.{algorithm.lower()}_agent")
    agent_class = getattr(agent_module, f"{algorithm}Agent")
    agent = agent_class(state_size, action_size, config) 

    mode = config['mode']

    if mode == "train":
        # 모델 로드 여부 확인
        model_path = config['load_model_path']
        if model_path is not None:
            agent.load_model(model_path)
            start_episode = agent.start_episode
            results_dir = "/".join(model_path.split("/")[:-2])
        else:
            # 결과를 저장할 폴더 생성
            results_dir = create_timestamped_directory(config)
            print(f"Results will be saved in: {results_dir}")
            start_episode = 0

        train_module = importlib.import_module(f"train.train_{algorithm.lower()}")
        train_function = getattr(train_module, "train")
        train_function(agent, env, config, results_dir, start_episode)
    elif mode == "eval":
        # 결과를 저장할 폴더 생성
        results_dir = create_timestamped_directory(config)
        print(f"Results will be saved in: {results_dir}")
        evaluate_module = importlib.import_module(f"train.evaluate_{algorithm.lower()}")
        evaluate_function = getattr(evaluate_module, "evaluate")
        evaluate_function(agent, env, config, results_dir)
        
if __name__ == '__main__':
    main()
