import argparse
import torch
from ud7 import UD7
from trainer import Trainer
from utils import set_seed, make_env


def get_parameters():
    parser = argparse.ArgumentParser()

    # Environment Setting
    parser.add_argument('--env-name', default='Humanoid-v4')
    parser.add_argument('--random-seed', default=-1, type=int)

    # UBOC
    parser.add_argument('--num_critics', default=5, type=int)

    # Checkpointing
    parser.add_argument('--use_checkpoints', default=True, type=bool)
    parser.add_argument('--max-eps-when-checkpointing', default=20, type=int)
    parser.add_argument('--steps-before-checkpointing', default=75e4, type=int)
    parser.add_argument('--reset-weight', default=0.9, type=float)

    # LAP
    parser.add_argument('--alpha', default=0.4, type=float)
    parser.add_argument('--min_priority', default=1, type=float)

    # Generic
    parser.add_argument('--target-update-rate', default=250, type=int)
    parser.add_argument('--start-steps', default=25e3, type=int)
    parser.add_argument('--max-steps', default=5000000, type=int)
    parser.add_argument('--zs-dim', default=256, type=int)
    parser.add_argument('--critic-hidden-dims', default=(256, 256))
    parser.add_argument('--policy-hidden-dims', default=(256, 256))
    parser.add_argument('--encoder-hidden-dims', default=(256, 256))
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--buffer-size', default=1000000, type=int)
    parser.add_argument('--policy-update-delay', default=2)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--actor-lr', default=0.0003, type=float)
    parser.add_argument('--critic-lr', default=0.0003, type=float)
    parser.add_argument('--encoder-lr', default=0.0003, type=float)

    # TD3
    parser.add_argument('--act-noise-scale', default=0.1, type=float)
    parser.add_argument('--target-noise-scale', default=0.2, type=float)
    parser.add_argument('--target-noise-clip', default=0.5, type=float)

    # Log & Evaluation
    parser.add_argument('--show-loss', default=False, type=bool)
    parser.add_argument('--eval_flag', default=True, type=bool)
    parser.add_argument('--eval-freq', default=5000, type=int)
    parser.add_argument('--eval-episode', default=10, type=int)

    param = parser.parse_args()

    return param


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_seed = set_seed(args.random_seed)
    env, eval_env = make_env(args.env_name, random_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = [env.action_space.low[0], env.action_space.high[0]]

    agent = UD7(state_dim, action_dim, action_bound, device, args)

    trainer = Trainer(env, eval_env, agent, args)
    trainer.run()


if __name__ == '__main__':
    args = get_parameters()
    main(args)
