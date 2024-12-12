# train/train_rdqn.py

import os
from tqdm import tqdm
from utils.logger import Logger

def train(agent, env, config, results_dir):
    agent.set_train_mode()  # 학습 모드 설정
    models_dir = os.path.join(results_dir, 'models')    # 모델을 저장할 directory
    os.makedirs(models_dir, exist_ok=True)
    num_episodes = config['num_episodes']
    total_steps = 0  # episode가 바뀌어도 초기화 되지 않는 step number
    rewards_window = []
    max_steps_per_episode = 100#len(env.sequence_numbers)
    model_save_freq = config['model_save_freq']

    logger = Logger(results_dir, config)
    # 로그 저장 빈도 설정
    log_save_freq = config['log_save_freq']  # 예: 100 에피소드마다 저장
    last_log_save_episode = 0  # 마지막으로 로그를 저장한 에피소드

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        episode_loss = []
        hidden_state = None  # 에피소드 시작 시 은닉 상태 초기화
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)  # 에피소드 시작 시 epsilon 업데이트
        with tqdm(total=max_steps_per_episode, desc=f"Episode {episode+1}/{num_episodes}", unit="step") as pbar:
            while not env.done and step < max_steps_per_episode:
                action, max_q_value, new_hidden_state = agent.select_action(state, hidden_state)
                # print(f'\naction: {action}\nstate:{state}')
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                hidden_state = new_hidden_state
                total_reward += reward
                step += 1
                total_steps += 1

                loss = agent.replay()
                if loss is not None:
                    episode_loss.append(loss)
                else:
                    loss = 0.0  # replay가 수행되지 않을 경우

                # 스텝별 로그 기록
                logger.log_step({
                    'Episode': episode + 1,
                    'Step': step,
                    'Total Steps': total_steps,
                    'Frame Number': info.get('seq_num', 0),
                    'Action': action,
                    'Max Q Value': max_q_value if max_q_value is not None else 0.0,
                    'SSIM': info.get('ssim', 0.0),
                    'Data Size': info.get('datasize', 0),
                    'Reward': reward,
                    'Return': total_reward,
                    'Loss': f"{loss:.4f}",
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

        # 에피소드 종료 후, 마지막 경험을 done=True로 저장
        if not done:
            agent.remember(state, action, reward, next_state, True)

        # 최근 100 에피소드의 평균 보상 계산
        rewards_window.append(total_reward)
        if len(rewards_window) > 100:
            rewards_window.pop(0)
        avg_reward = sum(rewards_window) / len(rewards_window)
        avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0.0

        # 에피소드별 로그 기록
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
        if (episode + 1) % model_save_freq == 0:
            model_path = os.path.join(models_dir, f'rdqn_model_episode{episode + 1}.pth')
            agent.save_model(model_path, episode)

    # 최종 모델 저장
    final_model_path = os.path.join(models_dir, 'rdqn_model_final.pth')
    agent.save_model(final_model_path, episode)
    print(f"Final model saved -> {final_model_path}")

    # 마지막으로 남은 로그 데이터 저장
    if last_log_save_episode < num_episodes:
        start_episode = last_log_save_episode + 1
        end_episode = num_episodes
        logger.save_logs(start_episode, end_episode)
