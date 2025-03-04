# evaluate/evaluate_ppo.py

import os
from tqdm import tqdm
import pandas as pd
import torch
from utils.logger import Logger

def evaluate(agent, env, config, results_dir):
    # 모델 로드
    agent.load_model(config['model_path'])
    agent.set_eval_mode()
    returns_ = []
    num_eval_episodes = config['num_eval_episodes']  # 평가 에피소드 수
    logger = Logger(results_dir, config)
    action_type = config.get('action_type', 'scalar')

    for episode in range(num_eval_episodes):
        state = env.reset()
        return_ = 0
        step = 0

        count_i_frame = 0
        count_p_frame = 0
        count_path_one_way = 0
        count_path_two_way = 0
        with torch.no_grad(): 
            with tqdm(total=len(env.sequence_numbers), desc=f"Evaluation Episode {episode+1}/{num_eval_episodes}", unit="step") as pbar:
                while not env.done:
                    # 에이전트로부터 액션 선택
                    action = agent.select_action(state, deterministic=True)
                    # 환경에서 한 스텝 진행
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    return_ += reward
                    step += 1

                    # 액션 해석: action_type에 따라 분기
                    if action_type == 'vector':
                        # action은 (path_action, frame_action)
                        if action[1] == 0:
                            count_i_frame += 1
                        else:
                            count_p_frame += 1
                        if action[0] == 2:  # 예: 2번이 'Both'
                            count_path_two_way += 1
                        else:
                            count_path_one_way += 1
                    else:
                        # scalar 방식 (예시: 홀수/짝수로 구분)
                        if action % 2 == 0:
                            count_i_frame += 1
                        else:
                            count_p_frame += 1
                        if action < 4:
                            count_path_one_way += 1
                        else:
                            count_path_two_way += 1
                    
                    # 스텝별 로그 기록
                    logger.log_step({
                        'Episode': episode + 1,
                        'Step': step,
                        'Frame Number': info['seq_num'],
                        'Action': action,
                        'Frame type': info['frame_type'],
                        'SSIM': info['ssim'],
                        'Data Size': info['datasize'],
                        'Frame loss': info['frame_loss'],
                        'Reward': reward,
                        'Total Reward': return_,
                    })

                    pbar.update(1)
            
        returns_.append(return_)
        print(f"Evaluation Episode {episode + 1}/{num_eval_episodes} | Return: {return_:.4f} | Steps: {step}")

        gop = (count_i_frame + count_p_frame) / count_i_frame
        avg_path = (count_path_one_way + (2 * count_path_two_way)) / (count_path_one_way + count_path_two_way)

        # 에피소드별 로그 기록
        logger.log_episode({
            'Episode': episode + 1,
            'Total Reward': return_,
            'Episode Length': step,
            'GoP': gop,
            'Average path': avg_path,
            'Count I-frame': count_i_frame,
            'Count P-frame': count_p_frame,
            'Count one way': count_path_one_way,
            'Count two way': count_path_two_way,
        })

    avg_reward = sum(returns_) / num_eval_episodes
    print(f"Average Reward over {num_eval_episodes} evaluation episodes: {avg_reward:.4f}")

    # 로그 데이터 저장
    logger.save_logs(1, 1)
