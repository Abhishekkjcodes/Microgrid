# Smart Energy Microgrid Optimizer

This project implements a Reinforcement Learning (RL) agent using a custom, from-scratch Q-learning algorithm to optimize energy management in a Smart Microgrid. The agent dynamically controls battery charging, discharging, and load shedding to minimize energy costs and maximize renewable energy utilization.

## Sustainable Development Goal 7 (SDG 7)

This project directly supports **SDG 7: Affordable and Clean Energy** by:
1. **Maximizing Renewable Usage:** The RL agent learns to intelligently store excess solar energy and deploy it during peak demand hours.
2. **Minimizing Grid Reliance:** Reducing peak-time grid imports via battery discharging decreases dependency on fossil-fuel-heavy grid generation.
3. **Enhancing Energy Efficiency:** Balancing load shedding and energy storage ensures a reliable electricity supply with maximum economic efficiency.

## Environment Details

A custom `gymnasium`-compliant environment (`sim/microgrid_env.py`) simulates a full 24-hour cycle. It relies entirely on internal synthetic data generation (solar sine waves, demand profiles, and ToU pricing).

- **State Space (Discrete):**
  - Battery SOC: 4 bins (0–25%, 25–50%, 50–75%, 75–100%)
  - Solar Output: 3 bins (Low, Med, High)
  - Current Load: 2 bins (Low, High)
  - Grid Price: 2 bins (Off-peak, Peak)
- **Action Space:**
  - `0`: Charge battery
  - `1`: Discharge battery
  - `2`: Import directly from grid
  - `3`: Shed load
- **Reward Function:**
  The agent minimizes the total operational cost. The reward is defined as `-(grid_import_units * current_price + blackout_penalty)`.

## Project Structure

```text
Microgrid/
├── sim/
│   ├── microgrid_env.py      # Custom gymnasium environment
│   └── fixed_controller.py   # Baseline heuristic controller for comparison
├── configs/
│   ├── qlearn_v1.yaml        # Config v1 (1000 episodes)
│   └── qlearn_v2.yaml        # Config v2 (2000 episodes)
├── experiments/              # Auto-generated outputs (CSVs, Q-tables, Plots)
├── train.py                  # Custom Q-Learning training loop
├── evaluate.py               # Evaluation logic, console tables, and matplotlib generation
└── README.md                 # Project documentation
```

## Installation & Setup

To run this project, you need Python installed on your system.
Install the required dependencies via pip:

```bash
pip install numpy pyyaml matplotlib gymnasium
```

## Usage Instructions

### 1. Train the Agent
Run the training script to learn the Q-table over the simulated environment episodes. This will save a `.csv` log of rewards and a `.npy` serialized Q-table into the `experiments/` folder.
*By default, it uses `configs/qlearn_v1.yaml`.*

```bash
python Microgrid/train.py
```


### 2. Evaluate and Compare
Run the evaluation script to pit the trained RL agent against the baseline fixed-rule controller. This script outputs a clear comparison table to the console and generates visualizations (a training curve and a 24-hour step-by-step comparison) in the `experiments/` folder.

```bash
python Microgrid/evaluate.py
```
## Reproducibility

To reproduce experiment 1 exactly:
```bash
python train.py configs/qlearn_v1.yaml
```
To reproduce experiment 2:
```bash
python train.py configs/qlearn_v2.yaml
```
Both use a fixed seed — anyone can clone this repo and get identical results.

## Monitoring Plan (Deployment Design)

If this system were deployed in a real microgrid, we would monitor the following:
- **Average grid import cost per hour** — alert if cost exceeds 20% above the training baseline, which could indicate the policy is failing to utilise solar correctly.
- **Battery SOC drift** — if battery stays below 20% for 3 or more consecutive hours, it signals the agent is not charging during solar surplus hours and policy retraining may be needed.
- **Renewable utilisation rate** — track what percentage of solar generation is being self-consumed vs wasted. Target above 80%.
- **Blackout frequency** — zero tolerance. Any unserved load event triggers an immediate alert.
- **Grid price tier alignment** — verify the agent is consistently choosing DISCHARGE during peak price hours (17–21) and CHARGE during off-peak solar hours (10–14). A drift here means the policy is degrading.
