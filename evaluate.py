import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

# Add base directory to sys.path to allow absolute imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from sim.microgrid_env import MicrogridEnv
from sim.fixed_controller import FixedController

def state_to_idx(state_arr):
    """Helper function to map the multi-discrete state array to a single index."""
    return (state_arr[0] * 12) + (state_arr[1] * 4) + (state_arr[2] * 2) + state_arr[3]

def run_simulation(env, policy_func):
    """
    Runs a 24-hour simulation using the provided policy function.
    Returns aggregated metrics and tracking histories.
    """
    obs, _ = env.reset()
    terminated = False
    
    total_cost = 0.0
    total_blackout = 0.0
    total_import = 0.0
    
    soc_history = []
    cost_history = []
    
    while not terminated:
        action = policy_func(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate metrics
        total_cost += info['cost']
        total_blackout += info['blackout']
        total_import += info['grid_import']
        
        # Record histories for plotting
        soc_history.append(info['soc'])
        cost_history.append(info['cost'])
        
        obs = next_obs
        
    return total_cost, total_blackout, total_import, soc_history, cost_history

def main():
    env = MicrogridEnv()
    experiments_dir = os.path.join(BASE_DIR, "experiments")
    
    # 1. Load the trained Q-table
    q_table_path = os.path.join(experiments_dir, "q_table.npy")
    try:
        q_table = np.load(q_table_path)
    except FileNotFoundError:
        print(f"Error: {q_table_path} not found. Please run train.py first.")
        return
        
    # Wrapper function to act as the RL policy
    def rl_policy(obs):
        state_idx = state_to_idx(obs)
        return np.argmax(q_table[state_idx])
        
    # 2. Run Simulations
    rl_cost, rl_blackout, rl_import, rl_soc, rl_cost_hist = run_simulation(env, rl_policy)
    
    fixed_ctrl = FixedController(env)
    fx_cost, fx_blackout, fx_import, fx_soc, fx_cost_hist = run_simulation(env, fixed_ctrl.act)
    
    # 3. Print Comparison Table
    print("\n" + "="*55)
    print(" 24-HOUR MICROGRID SIMULATION COMPARISON")
    print("="*55)
    print(f"{'Metric':<20} | {'RL Agent':<14} | {'Fixed Controller'}")
    print("-" * 55)
    print(f"{'Total Cost ($)':<20} | {rl_cost:<14.2f} | {fx_cost:.2f}")
    print(f"{'Blackouts (kWh)':<20} | {rl_blackout:<14.2f} | {fx_blackout:.2f}")
    print(f"{'Grid Import (kWh)':<20} | {rl_import:<14.2f} | {fx_import:.2f}")
    print("="*55 + "\n")
    
    # 4. Generate Plots
    
    # Plot A: Training Reward Curve
    csv_path = os.path.join(experiments_dir, "results_1.csv")
    episodes, rewards = [], []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row['episode']))
                rewards.append(float(row['total_reward']))
                
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards, label="Total Reward per Episode", color="#4C72B0", alpha=0.5)
        # Apply a simple moving average filter to smooth the noise if enough data points exist
        if len(rewards) > 50:
            ma = np.convolve(rewards, np.ones(50)/50, mode='valid')
            plt.plot(episodes[49:], ma, color="#C44E52", linewidth=2, label="Moving Average (50 ep)")
            
        plt.title("RL Agent Q-Learning Training Curve")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        curve_path = os.path.join(experiments_dir, "training_curve.png")
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curve plot to {curve_path}")
        plt.close()
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Skipping reward curve plot generation.")

    # Plot B: 24-Hour Step-by-Step State Comparison
    hours = range(24)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Subplot 1: Battery State of Charge (SOC)
    ax1.plot(hours, rl_soc, marker='o', label="RL Agent SOC", color="#55A868", linewidth=2)
    ax1.plot(hours, fx_soc, marker='x', label="Fixed Controller SOC", color="#DD8452", linewidth=2, linestyle='--')
    ax1.set_ylabel("Battery SOC (kWh)")
    ax1.set_title("24-Hour Microgrid Simulation: State of Charge (SOC) Tracking")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Subplot 2: Grid Cost per Hour
    ax2.plot(hours, rl_cost_hist, marker='o', label="RL Agent Cost", color="#C44E52", linewidth=2)
    ax2.plot(hours, fx_cost_hist, marker='x', label="Fixed Controller Cost", color="#8172B3", linewidth=2, linestyle='--')
    ax2.set_xlabel("Hour of Day (0-23)")
    ax2.set_ylabel("Grid Cost ($)")
    ax2.set_title("24-Hour Microgrid Simulation: Hourly Grid Cost")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    comp_path = os.path.join(experiments_dir, "comparison_plot.png")
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {comp_path}")
    plt.close()

if __name__ == "__main__":
    main()
