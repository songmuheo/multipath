# train/train.py

import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from utils.logger import Logger

def train(agent, env, config, results_dir):
    agent.set_train_mode()  # 학습 모드 설정
    models_dir = os.path.join(results_dir, 'models')    # 모델을 저장할 directory
    num_episodes = config['num_episodes']
    total_steps = 0  # episode가 바뀌어도 초기화 되지 않는 step number
    target_model_update_freq = config['target_model_update_freq']   # target network update 하는 주기(step 기준)
    moving_average_reward = 0
    rewards_window = []
    max_steps_per_episode = len(env.sequence_numbers)
    model_save_freq = config['model_save_freq']

    logger = Logger(results_dir, config)
    # 로그 저장 빈도 설정
    log_save_freq = config['log_save_freq']  # 100 에피소드마다 저장
    last_log_save_episode = 0  # 마지막으로 로그를 저장한 에피소드

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        episode_loss = []
        with tqdm(total=max_steps_per_episode, desc=f"Episode {episode+1}/{num_episodes}", unit="step") as pbar:
            while not env.done and step < max_steps_per_episode:
                action, max_q_value = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step += 1
                total_steps += 1

                loss = agent.replay()
                if loss is None:
                    loss = 0.0
                else:
                    episode_loss.append(loss)

                # target_network update
                if total_steps % target_model_update_freq == 0:
                    agent.update_target_model()

                # 스텝별 로그 기록
                logger.log_step({
                    'Episode': episode + 1,
                    'Step': step,
                    'Total Steps': total_steps,
                    'Frame Number': info['seq_num'],
                    'Action': action,
                    'Max Q Value': max_q_value if max_q_value is not None else 0.0,
                    'SSIM': info['ssim'],
                    'Data Size': info['data_size'],
                    'Reward': reward,
                    'Return': total_reward,
                    'Loss': loss,
                    'Epsilon': agent.epsilon
                })

                # 진행 상황 업데이트
                pbar.set_postfix({
                    'Step Reward': f"{reward:.4f}",
                    'Loss': f"{loss:.4f}",
                    'Total Reward': f"{total_reward:.4f}",
                    'Epsilon': f"{agent.epsilon:.4f}"
                })
                pbar.update(1)

        # epsilon 매 episode마다 선형적으로 감소
        agent.update_epsilon()
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
            'Epsilon': agent.epsilon
        })
    
        # 로그 저장
        if (episode + 1) % log_save_freq == 0:
            start_episode = last_log_save_episode + 1
            end_episode = episode + 1
            logger.save_logs(start_episode, end_episode)
            last_log_save_episode = episode + 1  # 마지막 저장 에피소드 업데이트

        # 모델 저장
        if episode % model_save_freq == 0:
            model_path = os.path.join(models_dir, f'dqn_model_episode{episode + 1}.pth')
            agent.save_model(model_path, episode)


    # 최종 모델 저장
    final_model_path = os.path.join(models_dir, 'dqn_model_final.pth')
    agent.save_model(final_model_path, episode)
    print(f"Final model saved -> {final_model_path}")

    # 마지막으로 남은 로그 데이터 저장
    if last_log_save_episode < num_episodes:
        start_episode = last_log_save_episode + 1
        end_episode = num_episodes
        logger.save_logs(start_episode, end_episode)