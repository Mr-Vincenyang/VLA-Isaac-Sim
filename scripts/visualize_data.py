#!/usr/bin/env python3
"""
VLA 演示数据可视化脚本
Visualize collected demonstration data from HDF5 files
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Optional
import os


def visualize_episode(
    episode_path: str,
    save_dir: Optional[str] = None,
    show_plot: bool = True,
    num_frames: int = 8
) -> None:
    """
    可视化单个 episode
    
    Args:
        episode_path: HDF5 文件路径
        save_dir: 保存图像的目录
        show_plot: 是否显示图像
        num_frames: 显示的帧数
    """
    with h5py.File(episode_path, 'r') as f:
        images = f['images'][:]
        actions = f['actions'][:]
        ee_positions = f['ee_positions'][:] if 'ee_positions' in f else None
        rewards = f['rewards'][:] if 'rewards' in f else None
        
        # 获取 instruction (可能是字符串或字节)
        instruction = ""
        if 'instruction' in f:
            inst = f['instruction'][()]
            if isinstance(inst, bytes):
                instruction = inst.decode('utf-8')
            else:
                instruction = str(inst)
    
    T = len(images)
    episode_name = Path(episode_path).stem
    
    # 创建图像
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'{episode_name}\nInstruction: "{instruction}"', fontsize=12)
    
    # 1. 显示关键帧
    frame_indices = np.linspace(0, T-1, num_frames, dtype=int)
    
    for i, idx in enumerate(frame_indices):
        ax = fig.add_subplot(3, num_frames, i + 1)
        ax.imshow(images[idx])
        ax.set_title(f't={idx}', fontsize=9)
        ax.axis('off')
    
    # 2. 绘制动作曲线
    ax_actions = fig.add_subplot(3, 1, 2)
    action_labels = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    
    for i in range(min(7, actions.shape[1])):
        ax_actions.plot(actions[:, i], label=action_labels[i], color=colors[i], alpha=0.8)
    
    ax_actions.set_xlabel('Time Step')
    ax_actions.set_ylabel('Action Value')
    ax_actions.set_title('Actions over Time')
    ax_actions.legend(loc='upper right', ncol=4, fontsize=8)
    ax_actions.grid(True, alpha=0.3)
    ax_actions.set_xlim(0, T)
    
    # 3. 绘制末端执行器轨迹或奖励
    if ee_positions is not None:
        ax_ee = fig.add_subplot(3, 1, 3)
        ax_ee.plot(ee_positions[:, 0], label='x', color='red')
        ax_ee.plot(ee_positions[:, 1], label='y', color='green')
        ax_ee.plot(ee_positions[:, 2], label='z', color='blue')
        ax_ee.set_xlabel('Time Step')
        ax_ee.set_ylabel('Position (m)')
        ax_ee.set_title('End-Effector Position')
        ax_ee.legend()
        ax_ee.grid(True, alpha=0.3)
        ax_ee.set_xlim(0, T)
    elif rewards is not None:
        ax_reward = fig.add_subplot(3, 1, 3)
        ax_reward.plot(rewards, color='purple')
        ax_reward.set_xlabel('Time Step')
        ax_reward.set_ylabel('Reward')
        ax_reward.set_title('Reward over Time')
        ax_reward.grid(True, alpha=0.3)
        ax_reward.set_xlim(0, T)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{episode_name}_viz.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_trajectory_animation(
    episode_path: str,
    output_path: str,
    fps: int = 10
) -> None:
    """
    创建轨迹动画 GIF
    """
    try:
        from PIL import Image
    except ImportError:
        print("请安装 Pillow: pip install Pillow")
        return
    
    with h5py.File(episode_path, 'r') as f:
        images = f['images'][:]
    
    # 转换为 PIL 图像
    pil_images = [Image.fromarray(img) for img in images]
    
    # 保存为 GIF
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=1000 // fps,
        loop=0
    )
    print(f'Saved animation: {output_path}')


def visualize_dataset_statistics(
    data_dir: str,
    save_path: Optional[str] = None,
    max_episodes: int = 100
) -> None:
    """
    可视化整个数据集的统计信息
    """
    data_dir = Path(data_dir)
    episode_files = sorted(data_dir.glob('episode_*.h5'))[:max_episodes]
    
    if not episode_files:
        print(f"No episodes found in {data_dir}")
        return
    
    print(f"Analyzing {len(episode_files)} episodes...")
    
    all_actions = []
    all_rewards = []
    episode_lengths = []
    
    for ep_file in episode_files:
        with h5py.File(ep_file, 'r') as f:
            all_actions.append(f['actions'][:])
            if 'rewards' in f:
                all_rewards.append(f['rewards'][:].sum())
            episode_lengths.append(len(f['actions']))
    
    all_actions = np.concatenate(all_actions, axis=0)
    
    # 创建统计图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Dataset Statistics ({len(episode_files)} episodes)', fontsize=14)
    
    # 1. 动作分布直方图
    ax1 = axes[0, 0]
    action_labels = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
    for i in range(min(7, all_actions.shape[1])):
        ax1.hist(all_actions[:, i], bins=50, alpha=0.5, label=action_labels[i])
    ax1.set_xlabel('Action Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Action Distribution')
    ax1.legend(fontsize=8)
    
    # 2. Episode 长度分布
    ax2 = axes[0, 1]
    ax2.hist(episode_lengths, bins=20, color='steelblue', edgecolor='white')
    ax2.set_xlabel('Episode Length')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Episode Lengths (mean={np.mean(episode_lengths):.1f})')
    ax2.axvline(np.mean(episode_lengths), color='red', linestyle='--', label='Mean')
    ax2.legend()
    
    # 3. 动作均值和标准差
    ax3 = axes[1, 0]
    action_means = all_actions.mean(axis=0)
    action_stds = all_actions.std(axis=0)
    x = np.arange(len(action_means))
    ax3.bar(x, action_means, yerr=action_stds, capsize=5, color='coral', edgecolor='white')
    ax3.set_xticks(x)
    ax3.set_xticklabels(action_labels[:len(action_means)], rotation=45)
    ax3.set_ylabel('Value')
    ax3.set_title('Action Mean ± Std')
    ax3.grid(True, alpha=0.3)
    
    # 4. 累积奖励分布
    ax4 = axes[1, 1]
    if all_rewards:
        ax4.hist(all_rewards, bins=20, color='green', edgecolor='white')
        ax4.set_xlabel('Cumulative Reward')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Episode Rewards (mean={np.mean(all_rewards):.2f})')
    else:
        ax4.text(0.5, 0.5, 'No reward data', ha='center', va='center', fontsize=14)
        ax4.set_title('Episode Rewards')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize VLA demonstration data')
    parser.add_argument('--data_dir', type=str, default='data/demos',
                       help='Directory containing HDF5 files')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode index to visualize')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save visualizations')
    parser.add_argument('--stats', action='store_true',
                       help='Show dataset statistics')
    parser.add_argument('--gif', action='store_true',
                       help='Create animation GIF')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display plots')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if args.stats:
        # 显示数据集统计
        save_path = os.path.join(args.save_dir, 'dataset_stats.png') if args.save_dir else None
        visualize_dataset_statistics(str(data_dir), save_path)
    else:
        # 显示单个 episode
        episode_files = sorted(data_dir.glob('episode_*.h5'))
        if not episode_files:
            print(f"No episodes found in {data_dir}")
            return
        
        if args.episode >= len(episode_files):
            print(f"Episode {args.episode} not found. Max index: {len(episode_files)-1}")
            return
        
        episode_path = str(episode_files[args.episode])
        print(f"Visualizing: {episode_path}")
        
        if args.gif:
            # 创建 GIF
            output_path = episode_path.replace('.h5', '.gif')
            if args.save_dir:
                output_path = os.path.join(args.save_dir, Path(output_path).name)
            create_trajectory_animation(episode_path, output_path)
        else:
            # 静态可视化
            visualize_episode(
                episode_path,
                save_dir=args.save_dir,
                show_plot=not args.no_show
            )


if __name__ == '__main__':
    main()
