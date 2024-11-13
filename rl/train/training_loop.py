from tqdm import tqdm

def episode_loop(agent, env, config, episode, total_steps, logger):
    state = env.reset()
    total_reward = 0
    step = 0
    avg_loss = 0
    max_q_values = []

    with tqdm(total=len(env.sequence_numbers), desc=f"Episode {episode+1}/{config['num_episodes']}", unit="step") as pbar:
        while not env.done and step < len(env.sequence_numbers):
            action, max_q_value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()

            # Max Q-Value 및 Loss Logging
            if max_q_value is not None:
                max_q_values.append(max_q_value)
            if loss:
                avg_loss += loss

            # 상태, 보상 및 스텝 업데이트
            state = next_state
            total_reward += reward
            step += 1
            total_steps += 1

            logger.log_step({
                'Episode': episode + 1,
                'Step': step,
                'Total Steps': total_steps,
                'Frame Number': info['seq_num'],
                'SSIM': info['ssim'],
                'Data Size': info['data_size'],
                'Reward': reward,
                'Total Reward': total_reward,
                'Loss': loss if loss is not None else 0.0,
                'Epsilon': agent.epsilon
            })

            pbar.set_postfix({
                'Step Reward': f"{reward:.4f}",
                'Loss': f"{loss:.4f}" if loss else 'N/A',
                'Total Reward': f"{total_reward:.4f}",
                'Epsilon': f"{agent.epsilon:.4f}"
            })
            pbar.update(1)

    avg_loss /= step if step > 0 else 1
    return total_reward, step, avg_loss, max_q_values
