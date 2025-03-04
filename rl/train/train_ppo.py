# train/train_ppo.py

import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from utils.logger import Logger

def train(agent, env, config, results_dir, start_episode=0):
    agent.set_train_mode()  # 학습 모드 설정
    models_dir = os.path.join(results_dir, 'models')    # 모델을 저장할 directory
    num_episodes = config['num_episodes']
    moving_average_reward = 0
    rewards_window = []
    max_steps_per_episode = len(env.sequence_numbers)
    model_save_freq = config['model_save_freq']

    logger = Logger(results_dir, config)
    # 로그 저장 빈도 설정
    log_save_freq = config['log_save_freq']  # 100 에피소드마다 저장
    last_log_save_episode = start_episode  # 마지막으로 로그를 저장한 에피소드

    for episode in range(start_episode, num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        episode_loss = []
        agent.timestep = 0  # 에피소드 시작 시 timestep 초기화
        agent.memory = []  # 에피소드 시작 시 메모리 초기화
        with tqdm(total=max_steps_per_episode, desc=f"Episode {episode+1}/{num_episodes}", unit="step") as pbar:
            while not env.done and step < max_steps_per_episode:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                loss = agent.remember(reward, done)
                if loss is not None:
                    episode_loss.append(loss)
                state_dict = {f'State_{i}': val for i, val in enumerate(state)}
                state = next_state
                total_reward += reward
                step += 1

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
                    'Return': total_reward,
                    **state_dict
                })

                # 진행 상황 업데이트
                pbar.set_postfix({
                    'Step Reward': f"{reward:.4f}",
                    'Total Reward': f"{total_reward:.4f}",
                })
                pbar.update(1)

            # 에피소드가 끝난 후 남은 메모리로 업데이트 (필요한 경우)
            if agent.memory:
                loss = agent.update()
                agent.memory = []
                episode_loss.append(loss)
                
            # 스케줄러 업데이트 (에피소드 단위)
            current_lr = agent.get_current_lr()
            agent.update_scheduler()

        moving_average_reward = 0.9 * moving_average_reward + 0.1 * total_reward
        # 최근 100 에피소드의 평균 보상 계산
        rewards_window.append(total_reward)
        if len(rewards_window) > 100:
            rewards_window.pop(0)
        avg_reward = sum(rewards_window) / len(rewards_window)
        avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0.0

        logger.log_episode({
            'Episode': episode + 1,
            'Return': total_reward,
            'Average Reward (100 eps)': avg_reward,
            'Episode Length': step,
            'Average Loss': avg_loss,
            'Learning Rate': current_lr,
        })

        # 로그 저장
        if (episode + 1) % log_save_freq == 0:
            start_episode = last_log_save_episode + 1
            end_episode = episode + 1
            logger.save_logs(start_episode, end_episode)
            last_log_save_episode = episode + 1  # 마지막 저장 에피소드 업데이트

        # 모델 저장
        if (episode + 1) % model_save_freq == 0:
            model_path = os.path.join(models_dir, f'ppo_model_episode{episode + 1}.pth')
            agent.save_model(model_path, episode)

    # 최종 모델 저장
    final_model_path = os.path.join(models_dir, 'ppo_model_final.pth')
    agent.save_model(final_model_path, episode)
    print(f"Final model saved -> {final_model_path}")

    # 마지막으로 남은 로그 데이터 저장
    if last_log_save_episode < num_episodes:
        start_episode = last_log_save_episode + 1
        end_episode = num_episodes
        logger.save_logs(start_episode, end_episode)
