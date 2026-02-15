import numpy as np

class Trainer:
    def __init__(self, env, eval_env, agent, args):
        self.args = args

        self.agent = agent
        self.env_name = args.env_name
        self.env = env
        self.eval_env = eval_env

        self.start_steps = args.start_steps
        self.max_steps = args.max_steps
        self.batch_size = args.batch_size
        self.target_noise_scale = args.target_noise_scale

        self.eval_flag = args.eval_flag
        self.eval_episode = args.eval_episode
        self.eval_freq = args.eval_freq

        self.episode = 0
        self.episode_reward = 0
        self.total_steps = 0
        self.eval_num = 0
        self.finish_flag = False

        self.target_noise_scale = args.target_noise_scale
        self.policy_update_delay = args.policy_update_delay

    def evaluate(self):
        # Evaluate process
        self.eval_num += 1
        reward_list = []

        for epi in range(self.eval_episode):
            epi_reward = 0
            state, _ = self.eval_env.reset()

            done = False

            while not done:
                action = self.agent.get_action(state, use_checkpoint=self.args.use_checkpoints, add_noise=False)
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                epi_reward += reward
                state = next_state
            reward_list.append(epi_reward)

        print("Eval  |  total_step {}  |  episode {}  |  Average Reward {:.2f}  |  Max reward: {:.2f}  |  "
              "Min reward: {:.2f}".format(self.total_steps, self.episode, sum(reward_list)/len(reward_list),
                                               max(reward_list), min(reward_list), np.std(reward_list)))

    def run(self):
        # Train-process start.
        allow_train = False

        while not self.finish_flag:
            self.episode += 1
            done = False
            ep_total_reward, ep_timesteps = 0, 0

            state, _ = self.env.reset()
            # Episode start.
            while not done:
                self.total_steps += 1
                ep_timesteps += 1

                if allow_train:
                    action = self.agent.get_action(state, use_checkpoint=False, add_noise=True)
                else:
                    action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                ep_total_reward += reward

                done_mask = 0.0 if ep_timesteps == self.env._max_episode_steps else float(done)
                self.agent.buffer.push(state, action, reward, next_state, done_mask)

                state = next_state

                if allow_train and not self.args.use_checkpoints:
                    actor_loss, critic_loss, encoder_loss = self.agent.train()
                    # Print loss.
                    if self.args.show_loss:
                        print("Loss  |  Actor loss {:.3f}  |  Critic loss {:.3f}  |  Encoder loss {:.3f}"
                              .format(actor_loss, critic_loss, encoder_loss))

                if done:
                    if allow_train and self.args.use_checkpoints:
                        self.agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

                    if self.total_steps >= self.args.start_steps:
                        allow_train = True

                # Evaluation.
                if self.eval_flag and self.total_steps % self.eval_freq == 0:
                    self.evaluate()

                # Raise finish_flag.
                if self.total_steps == self.max_steps:
                    self.finish_flag = True










