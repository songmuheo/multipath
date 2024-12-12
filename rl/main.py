# main.py
from environment.env import StreamingEnvironment
from utils.directory_manager import create_timestamped_directory
from utils.config import load_config
import importlib

def main():
    # 결과를 저장할 폴더 생성
    results_dir = create_timestamped_directory()
    print(f"Results will be saved in: {results_dir}")

    # load config
    config_path = 'configs/config.yaml'
    config = load_config(config_path)
    algorithm = config['algorithm']  # 'DQN' 또는 'PPO'

    env = StreamingEnvironment(config)
    state_size = len(env.reset())
    action_size = len(config['actions'])  # Action 개수

    # 에이전트 동적 임포트 및 인스턴스화
    agent_module = importlib.import_module(f"agents.{algorithm.lower()}_agent")
    agent_class = getattr(agent_module, f"{algorithm}Agent")
    agent = agent_class(state_size, action_size, config) 

    mode = config['mode']

    if mode == "train":
        # 모델 로드 여부 확인
        model_path = config['load_model_path']
        if model_path is not None:
            agent.load_model(model_path)
            start_episode = agent.start_episode
        else:
            start_episode = 0

        train_module = importlib.import_module(f"train.train_{algorithm.lower()}")
        train_function = getattr(train_module, "train")
        train_function(agent, env, config, results_dir, start_episode)
    elif mode == "eval":
        evaluate_module = importlib.import_module(f"train.evaluate_{algorithm.lower()}")
        evaluate_function = getattr(evaluate_module, "evaluate")
        evaluate_function(agent, env, config, results_dir)
        
if __name__ == '__main__':
    main()
