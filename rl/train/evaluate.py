# evaluate/evaluate.py

import os
from tqdm import tqdm
from utils.logger import Logger

def evaluate(agent, env, config, results_dir):
    agent.load_model(config['model_path'])
    agent.set_eval_mode()
    total_rewards = []
    num_eval_episodes = config['num_eval_episodes']  # 평가 에피소드 수
    logger = Logger(results_dir, config)

    for episode in range(num_eval_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        with tqdm(total=len(env.sequence_numbers), desc=f"Evaluation Episode {episode+1}/{num_eval_episodes}", unit="step") as pbar:
            while not env.done:
                action, max_q_value = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                total_reward += reward
                step += 1

                # 스텝별 로그 기록
                logger.log_step({
                    'Episode': episode + 1,
                    'Step': step,
                    'Frame Number': info['seq_num'],
                    'Action': action,
                    'Max Q Value': max_q_value if max_q_value is not None else 0.0,
                    'SSIM': info['ssim'],
                    'Data Size': info['data_size'],
                    'Reward': reward,
                    'Total Reward': total_reward,
                })

                pbar.update(1)

        total_rewards.append(total_reward)
        print(f"Evaluation Episode {episode + 1}/{num_eval_episodes} | Total Reward: {total_reward:.4f} | Steps: {step}")
        env.save_episode_data(episode + 1)

        # 에피소드별 로그 기록
        logger.log_episode({
            'Episode': episode + 1,
            'Total Reward': total_reward,
            'Episode Length': step,
        })

    avg_reward = sum(total_rewards) / num_eval_episodes
    print(f"Average Reward over {num_eval_episodes} evaluation episodes: {avg_reward:.4f}")

    # 로그 데이터 저장
    logger.save_logs()