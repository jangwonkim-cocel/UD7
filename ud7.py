import numpy as np
import torch
import torch.nn.functional as F
import copy
from replay_memory import LAP
from network import Policy, Encoder, EnsembleQNet
from utils import hard_update, LAP_huber


class UD7:
    def __init__(self, state_dim, action_dim, action_bound, device, args):
        self.args = args

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.device = device
        self.buffer = LAP(self.state_dim, self.action_dim, device, args.buffer_size, normalize_action=True,
                          max_action=action_bound[1], prioritized=True)
        self.batch_size = args.batch_size

        self.gamma = args.gamma
        self.act_noise_scale = args.act_noise_scale

        self.num_critics = args.num_critics

        self.actor = Policy(self.state_dim, self.action_dim, self.device, args.zs_dim, args.policy_hidden_dims).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.target_actor = Policy(self.state_dim, self.action_dim, self.device, args.zs_dim, args.policy_hidden_dims).to(self.device)

        self.critic = EnsembleQNet(self.num_critics,  self.state_dim, self.action_dim,
                                   self.device, args.zs_dim, args.critic_hidden_dims).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.target_critic = EnsembleQNet(self.num_critics, self.state_dim, self.action_dim,
                                   self.device, args.zs_dim, args.critic_hidden_dims).to(self.device)

        self.encoder = Encoder(state_dim, action_dim, self.device, args.zs_dim, args.encoder_hidden_dims).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.training_steps = 0

        self.max_action = action_bound[1]

        # Checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0

        hard_update(self.actor, self.target_actor)
        hard_update(self.critic, self.target_critic)

    def get_action(self, state, use_checkpoint=False, add_noise=True):
        with torch.no_grad():
            if add_noise:
                if use_checkpoint:
                    zs = self.checkpoint_encoder.zs(state)
                    action = self.checkpoint_actor(state, zs)
                    action = action + torch.randn_like(action) * self.act_noise_scale
                    action = np.clip(action.cpu().numpy()[0], -1, 1)
                else:
                    zs = self.fixed_encoder.zs(state)
                    action = self.actor(state, zs)
                    action = action + torch.randn_like(action) * self.act_noise_scale
                    action = np.clip(action.cpu().numpy()[0], -1, 1)
            else:
                if use_checkpoint:
                    zs = self.checkpoint_encoder.zs(state)
                    action = self.checkpoint_actor(state, zs).cpu().numpy()[0]
                else:
                    zs = self.fixed_encoder.zs(state)
                    action = self.actor(state, zs).cpu().numpy()[0]

            action = action * self.max_action
        return action

    def train(self):
        self.training_steps += 1

        # Sample from LAP
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Update Encoder
        with torch.no_grad():
            next_zs = self.encoder.zs(next_states)
        zs = self.encoder.zs(states)
        pred_zs = self.encoder.zsa(zs, actions)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # Update Critic
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_states)

            target_act_noise = (torch.randn_like(actions) * self.args.target_noise_scale).clamp(-self.args.target_noise_clip, self.args.target_noise_clip).to(self.device)

            if self.buffer.do_normalize_action is True:
                next_target_actions = (self.target_actor(next_states, fixed_target_zs) + target_act_noise).clamp(-1, 1)
            else:
                next_target_actions = (self.target_actor(next_states, fixed_target_zs) + target_act_noise).clamp(-self.max_action, self.max_action)

            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_target_actions)

            Q_target = self.target_critic(next_states, next_target_actions, fixed_target_zsa, fixed_target_zs)
            m = Q_target.mean(dim=1, keepdim=True)  # Sample mean
            b = Q_target.var(dim=1, unbiased=True, keepdim=True) # Sample variance
            Bias_Corrected_Q_target = m - 0.5641896 * torch.sqrt(b)  # bias-corrected target Q

            Q_target = rewards + (1 - dones) * self.gamma * Bias_Corrected_Q_target.clamp(self.min_target, self.max_target)

            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(states)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actions)

        Q = self.critic(states, actions, fixed_zsa, fixed_zs)

        td_loss = (Q - Q_target).abs()
        critic_loss = LAP_huber(td_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update LAP
        priority = td_loss.max(1)[0].clamp(min=self.args.min_priority).pow(self.args.alpha)
        self.buffer.update_priority(priority)

        # Update Actor
        if self.training_steps % self.args.policy_update_delay == 0:
            actor_actions = self.actor(states, fixed_zs)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_actions)
            Q = self.critic(states, actor_actions, fixed_zsa, fixed_zs)

            actor_loss = -Q.mean(dim=1, keepdim=True).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        else:
            actor_loss = torch.tensor(0.0)

        # Update Iteration
        if self.training_steps % self.args.target_update_rate == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

            self.buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

        return actor_loss.item(), critic_loss.item(), encoder_loss.item()

    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        # End evaluation of current policy early
        if self.min_return < self.best_min_return:
            self.train_and_reset()

        # Update checkpoint
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())

            self.train_and_reset()

    # Batch training
    def train_and_reset(self):
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.args.steps_before_checkpointing:
                self.best_min_return *= self.args.reset_weight
                self.max_eps_before_update = self.args.max_eps_when_checkpointing

            self.train()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8
