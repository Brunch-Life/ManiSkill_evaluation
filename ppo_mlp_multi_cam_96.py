"""
finetune_mlp.py

eg:
    python finetune_mlp.py --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                          --dataset_name <DATASET_NAME> \
                          --run_root_dir <PATH/TO/LOGS/DIR> \
                          ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import gc
import time
import glob
import random
# import multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn as nn
# import torch.nn.functional as F
import tqdm
import wandb
import numpy as np
from torch.optim import AdamW
# from torch.utils.data import DataLoader, IterableDataset
# from scipy.spatial.transform import Rotation as R
# from transformers import AutoProcessor
# from PIL import Image
import tyro

import gymnasium as gym
# import imageio
from datetime import datetime

# import MLP related from SimplerEnv
from simpler_env.policies.MLP.MLP_train_multi_cam_96 import MLPPolicy

# import angle processing utilities
# from angle_utils import preprocess_action_for_training, robust_unwrap_angles

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class PPOConfig:
    # Directory Paths
    ckpt_path: str = "/mnt/public/chenyinuo/RL4VLA/ckpts/9.22_96/step_050000"       
    normalizer_path: str = "/mnt/public/chenyinuo/RL4VLA/ckpts/9.22_96/step_005000/normalize.npz"   
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    
    # base Parameters
    seed: int = 42                                                  # Random seed
    use_state: bool = False                                         # Whether to use state
    # MLP Model Parameters  
    mlp_embedding_size: int = 512                                   # MLP embedding size
    action_dim: int = 7                                             # Action dimension (3 pos + 3 euler + 1 gripper)
    freeze_backbone: bool = True
    torch_deterministic: bool = True
                                       
    # PPO things
    total_timesteps: int = 100000000
    """total timesteps of the experiments"""
    warmup_itr: int = 0
    """number of warmup iterations where we only optimize the value function"""
    save_interval: int = 10
    """save model interval (in iterations)"""
    learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 10
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    grad_acc_step: int = 16
    use_dense_reward: bool = False
    init_std_value: float = -0.1
    
    # Wandb Parameters
    wandb_project: str = "WM-ppo"                                  # Name of W&B project to log to
    
    # env parameters
    is_table_green: bool = False
    num_envs: int = 512
    env_id: str = "TabletopPickPlaceEnv-v1"
    shader: str = "default"  # default, rt
    episode_length: int = 200

    # fmt: on
    unnorm_key: Optional[str] = None
    
    @property
    def state_dim(self) -> int:
        """State dimension (3 pos + 4 quat + 1 gripper) = 8D if use_state, else 0"""
        return 8 if self.use_state else 0
    
class Normalizer():
    def __init__(self, path):
        self.action_stats = np.load(path)
        
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        "min max normalize action to [-1, 1]"
        return 2 * (action - self.action_stats["min"]) / (self.action_stats["max"] - self.action_stats["min"]) - 1.0

    def unnormalize_action(self, normed_action: np.ndarray) -> np.ndarray:
        "unnormalize action from [-1, 1] to original scale"
        return 0.5 * (normed_action + 1.0) * (self.action_stats["max"] - self.action_stats["min"]) + self.action_stats["min"]


def create_args_mock(mlp_embedding_size, alg_lr, action_dim=7, state_dim=0, use_state=False, init_std_value=-0.1):
    """create a mock args object for MLPPolicy"""
    class MockArgs:
        def __init__(self, mlp_embedding_size, alg_lr, action_dim, state_dim, use_state, init_std_value):
            self.mlp_embedding_size = mlp_embedding_size
            self.alg_lr = alg_lr
            self.action_dim = action_dim
            self.state_dim = state_dim
            self.use_state = use_state
            self.init_std_value = init_std_value

    return MockArgs(mlp_embedding_size, alg_lr, action_dim, state_dim, use_state, init_std_value)


def ppo_mlp(cfg: PPOConfig) -> None:
    print(f"PPO train MLP Model")
    cfg.batch_size = int(cfg.num_envs * cfg.num_steps)
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.num_iterations = cfg.total_timesteps // cfg.batch_size
    # assert cfg.num_steps % cfg.episode_length == 0, "num_steps must be multiple of episode_length"
    
    # [Validate] Ensure GPU Available & Set Device
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    
    # single GPU device setting
    device_id = 0
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device(device_id)
    print(f"Using single GPU training on device {device_id}")
    
    # init MLP policy
    mock_args = create_args_mock(cfg.mlp_embedding_size, cfg.learning_rate, cfg.action_dim, cfg.state_dim, cfg.use_state, cfg.init_std_value)
    mlp_policy = MLPPolicy(mock_args, device)
    
    # load pre-trained model (if provided)
    if cfg.ckpt_path is not None:
        print(f"Loading pre-trained MLP model from {cfg.ckpt_path}")
        mlp_policy.load(Path(cfg.ckpt_path))
    
    if cfg.normalizer_path is not None:
        print(f"Loading normalizer from {cfg.normalizer_path}")
        normalizer = Normalizer(Path(cfg.normalizer_path))
    else:
        raise ValueError("Please provide a normalizer path for finetuning!")

    # # freeze backbone (if needed)
    # if cfg.freeze_backbone:
    #     print("Freezing ResNet backbone")
    #     for key, param in mlp_policy.resnet_3rd.named_parameters():
    #         param.requires_grad = False
    #     for key, param in mlp_policy.resnet_wrist.named_parameters():
    #         param.requires_grad = False
    
    if cfg.warmup_itr > 0:
        for key, param in mlp_policy.named_parameters():
            print("freeze backbones and actors")
            if "critic" not in key:
                param.requires_grad = False
    # init optimizer
    optimizer = AdamW(mlp_policy.parameters(), lr=cfg.learning_rate)
    
    # init env
    env_kwargs = dict(
        num_envs=cfg.num_envs, 
        obs_mode="rgb+segmentation",
        control_mode="pd_ee_target_delta_pose",
        sim_backend="gpu",
        sim_config={
            "sim_freq": 1000,
            "control_freq": 25,
        },
        sensor_configs={"shader_pack": "default"},
        is_table_green = False,
        render_mode="rgb_array",
        robot_uids="panda_wristcam_96",
        reward_mode="normalized_dense"
    )
    env_kwargs["object_name"] = "green_bell_pepper"
    eval_env = gym.make("TabletopPickPlaceEnv96-v1", **env_kwargs)
    
    # init run dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"mlp_ppo_multi_cam_{timestamp}"
    # if not cfg.image_aug:
    #     exp_id += "-no_aug"
    run_dir = cfg.run_root_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    
    # init W&B logging
    name = f"{exp_id}"
    wandb.init(project=cfg.wandb_project, name=name)
    
    # init buffer
    image_shape = (3, 96, 96)
    action_shape = (7, )
    obs_front = torch.zeros((cfg.num_steps, cfg.num_envs) + image_shape)
    obs_wrist = torch.zeros((cfg.num_steps, cfg.num_envs) + image_shape)
    actions = torch.zeros((cfg.num_steps, cfg.num_envs) + action_shape)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs))
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs))
    dones = torch.zeros((cfg.num_steps, cfg.num_envs))
    values = torch.zeros((cfg.num_steps, cfg.num_envs))

    # Training Loop
    assert cfg.use_state == False, "Simulation evaluation currently only supports image-only input"
    print("Start training...")
    next_obs, _ = eval_env.reset()
    # render_list = []
    mlp_policy.eval()
    
    total_success_num = 0
    total_grasp_success_num = 0
    total_eval_num = 0
    
    global_step = 0
    start_time = time.time()
    next_obs, _ = eval_env.reset(seed=cfg.seed)
    next_obs_front = next_obs["sensor_data"]["3rd_view_camera"]["rgb"].permute(0, 3, 1, 2).to(device) / 255.0
    next_obs_wrist = next_obs["sensor_data"]["hand_camera"]["rgb"].permute(0, 3, 1, 2).to(device) / 255.0
    next_done = torch.zeros(cfg.num_envs).to(device)
    not_release_grad = True
    
    for iteration in range(1, cfg.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            lrnow = frac * cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        success_buffer = []
        success_tensor = np.zeros(cfg.num_envs, dtype=bool)
        for step in tqdm.tqdm(range(0, cfg.num_steps), desc=f"Rollout Episode {iteration}"):
        # for step in range(0, cfg.num_steps):
            global_step += cfg.num_envs
            obs_front[step] = next_obs_front.cpu()
            obs_wrist[step] = next_obs_wrist.cpu()
            dones[step] = next_done.cpu()

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = mlp_policy.get_action_and_value({'image_3rd': next_obs_front, 'image_wrist': next_obs_wrist}, action=None, deterministic=False)
            
            # store to buffer
            values[step] = value.flatten().cpu()
            actions[step] = action.cpu()
            logprobs[step] = logprob.cpu()

            # execute action
            action = normalizer.unnormalize_action(action.detach().cpu().numpy())
            next_obs, reward_ori, done, truncated, info = eval_env.step(action)
            
            # post process obs rewards and done.
            next_obs_front = next_obs["sensor_data"]["3rd_view_camera"]["rgb"].permute(0, 3, 1, 2).to(device) / 255.0
            next_obs_wrist = next_obs["sensor_data"]["hand_camera"]["rgb"].permute(0, 3, 1, 2).to(device) / 255.0
            success_tensor = np.logical_or(success_tensor, info["success"].cpu())
            if not cfg.use_dense_reward:
                grasped_tensor = info["is_src_obj_grasped"].cpu()
                max_reward_tensor = torch.ones_like(info["is_src_obj_grasped"]).cpu()
                reward_tensor = torch.min(max_reward_tensor, success_tensor.float() + grasped_tensor.float() * 0.1)
                reward = reward_tensor.numpy().copy().astype(np.float32)
            else:
                reward_tensor = torch.max(success_tensor.float(), reward_ori.cpu().float())
                reward = reward_tensor.cpu().numpy().copy().astype(np.float32)
            terminations = np.zeros(cfg.num_envs, dtype=bool)
            truncations = np.ones(cfg.num_envs, dtype=bool) if (step + 1) % cfg.episode_length == 0 else np.zeros(cfg.num_envs, dtype=bool)
            next_done = np.logical_or(terminations, truncations)
            
            # auto reset
            if next_done.any():
                assert next_done.all(), "All envs must be done at the same time!"
                next_obs, _ = eval_env.reset()
                next_obs_front = next_obs["sensor_data"]["3rd_view_camera"]["rgb"].permute(0, 3, 1, 2).to(device) / 255.0
                next_obs_wrist = next_obs["sensor_data"]["hand_camera"]["rgb"].permute(0, 3, 1, 2).to(device) / 255.0
                success_buffer.append(success_tensor.numpy().copy())
                success_tensor = np.zeros(cfg.num_envs, dtype=bool)
                # next_done = np.zeros(cfg.num_envs)
            
            rewards[step] = torch.tensor(reward).view(-1)
            next_done = torch.Tensor(next_done)

        # GAE 
        gc.collect()
        torch.cuda.empty_cache()
        # bootstrap value if not done
        with torch.no_grad():
            # next_value = mlp_policy.get_value({'image': next_obs}).reshape(1, -1).cpu()
            next_value = mlp_policy.get_value({'image_3rd': next_obs_front, "image_wrist": next_obs_wrist}).flatten().cpu()
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            
        # flatten the batch
        b_obs_front = obs_front.flatten(start_dim=0, end_dim=1)
        b_obs_wrist = obs_wrist.flatten(start_dim=0, end_dim=1)
        b_logprobs = logprobs.flatten()
        b_actions = actions.flatten(start_dim=0, end_dim=1)
        b_advantages = advantages.flatten()
        b_returns = returns.flatten()
        b_values = values.flatten()

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        kl_buffer = []
        ratio_buffer = []
        old_kl_buffer = []
        ratio_buffer = []
        vloss_buffer = []
        pg_loss_buffer = []
        entropy_buffer = []
        gc.collect()
        torch.cuda.empty_cache()
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            optimizer.zero_grad()
            epoch_loop_step = 0
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = mlp_policy.get_action_and_value({'image_3rd': b_obs_front[mb_inds].to(device), "image_wrist": b_obs_wrist[mb_inds].to(device)}, b_actions[mb_inds].to(device))
                newlogprob = newlogprob.view(-1)
                logratio = newlogprob - b_logprobs[mb_inds].to(device)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds].to(device)
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    batch_values = b_values[mb_inds].to(device)
                    v_loss_unclipped = (newvalue - batch_values) ** 2
                    v_clipped = batch_values + torch.clamp(
                        newvalue - batch_values,
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds].to(device)) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds].to(device)) ** 2).mean()
                    # print("vloss:", v_loss)

                entropy_loss = entropy.mean()
                if iteration < cfg.warmup_itr:
                    # warm up
                    loss = v_loss * cfg.vf_coef
                else:
                    loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef
                    if iteration == cfg.warmup_itr and not_release_grad:
                        not_release_grad = False
                        print("Warmup finished, optimizing policy and value function")
                        for key, param in mlp_policy.named_parameters():
                            if "resnet" not in key:
                                param.requires_grad = True
                    
                loss /= cfg.grad_acc_step

                # optimizer.zero_grad()
                loss.backward()
                epoch_loop_step += 1
                if epoch_loop_step % cfg.grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(mlp_policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    # for key, param in mlp_policy.named_parameters():
                    #     if "critic" in key:
                    #         print(key, ":", param)

                # add into buffer
                kl_buffer.append(approx_kl.item())
                old_kl_buffer.append(old_approx_kl.item())
                ratio_buffer.append(ratio.mean().item())
                vloss_buffer.append(v_loss.item())
                pg_loss_buffer.append(pg_loss.item())
                entropy_buffer.append(entropy_loss.item())
                ratio_buffer.append(ratio.mean().item())

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        wandb_log_dict = {}
        # print and add into wandb log dict
        success_array = np.concatenate(success_buffer, axis=0) if len(success_buffer) > 0 else np.zeros(cfg.num_envs)
        success_rate = success_array.mean()
        # print(f"---------------- Iteration {iteration} --------------")
        print(f"Iteration {iteration}  |  fps: {int(global_step / (time.time() - start_time))}  |  explained_var: {explained_var:.4f}  |  value_loss: {np.mean(np.array(vloss_buffer)):.4f}  | pg_loss: {np.mean(np.array(pg_loss_buffer)):.4f}  | ratio: {np.mean(np.array(ratio_buffer)):.4f} |  mean_return: {rewards.sum(dim=0).mean().item()/(cfg.num_steps // cfg.episode_length):.4f}  |  success_rate: {success_rate*100:.2f}%.")
        wandb_log_dict["charts/learning_rate"] = optimizer.param_groups[0]["lr"]
        wandb_log_dict["losses/value_loss"] = np.mean(vloss_buffer)
        wandb_log_dict["losses/policy_loss"] = np.mean(pg_loss_buffer)
        wandb_log_dict["losses/entropy"] = np.mean(entropy_buffer)
        wandb_log_dict["losses/old_approx_kl"] = np.mean(old_kl_buffer)
        wandb_log_dict["losses/approx_kl"] = np.mean(kl_buffer)
        wandb_log_dict["losses/clipfrac"] = np.mean(clipfracs)
        wandb_log_dict["losses/ratio"] = np.mean(ratio_buffer)
        wandb_log_dict["losses/explained_variance"] = explained_var
        wandb_log_dict["charts/SPS"] = int(global_step / (time.time() - start_time))
        wandb_log_dict["charts/mean_return"] = rewards.sum(dim=0).mean().item()/(cfg.num_steps // cfg.episode_length)
        wandb_log_dict["charts/success_rate"] = success_rate
        wandb.log(wandb_log_dict, step=global_step)
        
        if iteration % cfg.save_interval == 0:
            print(f"Saving model checkpoint for step {global_step}")
            save_path = run_dir / f"step_{global_step:06d}"
            
            # save model (single GPU)
            mlp_policy.save(save_path)
            
            # save optimizer state
            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'step': global_step,
            }, save_path / "optimizer.pt")
    
    
    
    # reference

    for idx in range(20):
        next_obs, _ = eval_env.reset()
        render_list = []
        mlp_policy.eval()
        
        with torch.no_grad():
            max_episode_step = cfg.episode_length
            success_tensor = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device_id)
            grasp_success_tensor = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device_id)
            for sim_step in tqdm.tqdm(range(max_episode_step), desc=f"Eval Episode {idx}"):
                image_front = next_obs["sensor_data"]["3rd_view_camera"]["rgb"].permute(0, 3, 1, 2).to(device_id) / 255.0
                image_wrist = next_obs["sensor_data"]["hand_camera"]["rgb"].permute(0, 3, 1, 2).to(device_id) / 255.0
                obs_dict = {
                    'image_3rd': image_front,
                    'image_wrist': image_wrist,
                }
                action, log_prob, entropy, value = mlp_policy.get_action_and_value(obs_dict, None, deterministic=True)
                action = normalizer.unnormalize_action(action.detach().cpu().numpy())
                next_obs, reward, done, truncated, info = eval_env.step(action)
                # img_tensor = next_obs["sensor_data"]["3rd_view_camera"]["rgb"]  # env.render()
                # render_list.append(img_tensor[0].cpu().numpy())  # (H, W, 3)
                
                # success_tensor = torch.logical_or(success_tensor, done)
                grasp_success_tensor = torch.logical_or(grasp_success_tensor, info["is_src_obj_grasped"].to(device_id))
                success_tensor = torch.logical_or(success_tensor, info["success"].to(device_id))
                
                # if done or truncated:
                #     break

            success_num = success_tensor.sum().item()
            grasp_success_num = grasp_success_tensor.sum().item()
            total_grasp_success_num += grasp_success_num
            total_success_num += success_num
            total_eval_num += cfg.num_envs
            grasp_success_rate = total_grasp_success_num / total_eval_num
            success_rate = total_success_num / total_eval_num
            print(f"  Episode {idx} Grasp Success Num: {grasp_success_num}/{cfg.num_envs}, Total Grasp Success Rate: {grasp_success_rate*100:.2f}%")
            print(f"  Episode {idx} Success Num: {success_num}/{cfg.num_envs}, Total Success Rate: {success_rate*100:.2f}%")
            
            # save video
            # video_path = f"./videos/step_{idx}.mp4"
            # os.makedirs(os.path.dirname(video_path), exist_ok=True)
            # imageio.mimwrite(video_path, np.array(render_list), fps=20, quality=8)

    # final save
    print("Saving final model...")
    final_save_path = run_dir / "final"
    mlp_policy.save(final_save_path)
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'step': iteration,
    }, final_save_path / "optimizer.pt")
    
    print("Training completed!")


if __name__ == "__main__":
    cfg = tyro.cli(PPOConfig)
    ppo_mlp(cfg)
