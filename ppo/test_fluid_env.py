import os
import time
from collections import deque

import gym
import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import GlobalRolloutStorage
from PIL import Image


def main():
    args = get_args()
    num_scenes = args.num_processes
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    # device = torch.device("cuda:0" if args.cuda else "cpu")
    device = torch.device("cpu")
    W, H = (128, 128)
    observation_space = gym.spaces.Box(0, 20,
                                       (6,
                                        W,
                                        H), dtype=np.float32)
    action_space = gym.spaces.Discrete(8)
    spaces = (observation_space, action_space)
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, spaces=spaces, screen_size=W,
                         debug=True)

    actor_critic = Policy(
        observation_space.shape,
        action_space,
        base_kwargs={'recurrent': False})
    actor_critic.to(device)
    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)
    args.num_steps = 500

    rollouts = GlobalRolloutStorage(args.num_steps, args.num_processes,
                                    envs.observation_space.shape, envs.action_space,
                                    actor_critic.recurrent_hidden_state_size, 8)
    start = time.time()
    obs = envs.reset()
    print('reset:', time.time() - start)
    actions = np.zeros((num_scenes, 1))
    actions.fill(-1)
    obs, _, _, infos = envs.step(actions)
    rollouts.obs[0].copy_(obs)
    vector = np.zeros((num_scenes, 8))
    for i in range(num_scenes):
        vector[i] = infos[i]['vector']
    vector = torch.FloatTensor(vector)
    rollouts.extras[0].copy_(vector)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    old_map = torch.zeros((1, 2, args.map_size_h, args.map_size_v)).to(device)
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        t1 = time.time()
        for step in range(args.num_steps):
            action = torch.tensor([[6]])
            obs, reward, done, infos = envs.step(action)
            rgb = torch.stack([obs[0][0], obs[0][1], obs[0][2]], dim=-1) * 255
            img = Image.fromarray(rgb.cpu().numpy().astype(np.uint8), mode="RGB")
            img.save(f"/data/private/zqh/embodied/tdw/output/test_fluid/env/{str(step)}.jpg")
        print("Total time", time.time()-t1)
        exit(0)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.rec_states[-1],
                rollouts.masks[-1], rollouts.extras[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

    envs.close()


if __name__ == "__main__":
    main()
