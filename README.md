# Smart Energy Microgrid Optimizer

This project implements a Reinforcement Learning (RL) agent using a custom, from-scratch Q-learning algorithm to optimize energy management in a Smart Microgrid. The agent dynamically controls battery charging, discharging, and load shedding to minimize energy costs and maximize renewable energy utilization.

The project also incorporates basic MLOps concepts such as:
- experiment tracking,
- reproducibility,
- configuration management,
- Git-based version control,
- and dashboard-based visualization.

---

# Sustainable Development Goal 7 (SDG 7)

This project directly supports **SDG 7: Affordable and Clean Energy** by:

1. **Maximizing Renewable Usage:**  
   The RL agent learns to intelligently store excess solar energy and deploy it during peak demand hours.

2. **Minimizing Grid Reliance:**  
   Reducing peak-time grid imports via battery discharging decreases dependency on fossil-fuel-heavy grid generation.

3. **Enhancing Energy Efficiency:**  
   Balancing load shedding and energy storage ensures a reliable electricity supply with maximum economic efficiency.

---

# Environment Details

A custom `gymnasium`-compliant environment (`sim/microgrid_env.py`) simulates a full 24-hour cycle. It relies entirely on internal synthetic data generation including:
- solar sine-wave generation,
- demand profiles,
- and time-of-use (ToU) pricing.

## State Space (Discrete)

- Battery SOC: 4 bins
  - 0–25%
  - 25–50%
  - 50–75%
  - 75–100%

- Solar Output: 3 bins
  - Low
  - Medium
  - High

- Current Load: 2 bins
  - Low
  - High

- Grid Price: 2 bins
  - Off-Peak
  - Peak

Total discrete states:
48 state combinations.

---

## Action Space

| Action | Description |
|---|---|
| 0 | Charge Battery |
| 1 | Discharge Battery |
| 2 | Import Directly from Grid |
| 3 | Shed Non-Critical Load |

---

## Reward Function

The reward is defined as:

```python
reward = -(grid_import_cost + blackout_penalty)
```

The RL agent therefore learns to:
- minimize operational cost,
- reduce expensive grid imports,
- maximize renewable energy utilization,
- and avoid blackout conditions.

---

# Project Structure

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
├── evaluate.py               # Evaluation logic and visualization generation
├── app.py                    # Streamlit dashboard for visualization/demo
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

# Technologies and Libraries Used

- Python
- NumPy
- Gymnasium
- Matplotlib
- Pandas
- PyYAML
- Streamlit
- Pillow (PIL)
- Git & GitHub

---

# Installation & Setup

Clone the repository:

```bash
git clone <your-repository-link>
```

Move into the project folder:

```bash
cd Microgrid
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

# requirements.txt

Create a file named `requirements.txt` and add:

```text
numpy
matplotlib
pandas
gymnasium
pyyaml
streamlit
pillow
```

---

# Usage Instructions

## 1. Train the RL Agent

Train using Config Version 1:

```bash
python train.py configs/qlearn_v1.yaml
```

Train using Config Version 2:

```bash
python train.py configs/qlearn_v2.yaml
```

Training generates:
- Q-table
- Policy files
- CSV experiment logs
- Reward tracking data

---

## 2. Evaluate and Compare

Run the evaluation script to compare:
- RL Agent
vs
- Fixed Heuristic Controller

```bash
python evaluate.py
```

This generates:
- training curves,
- RL vs Fixed Controller comparison plots,
- and evaluation metrics
inside the `experiments/` folder.

---

## 3. Launch Streamlit Dashboard (Optional)

The Streamlit dashboard provides:
- experiment visualization,
- RL workflow explanation,
- graph visualization,
- and easier project demonstration.

Run:

```bash
streamlit run app.py
```

---

# Experiment Tracking

Each experiment stores:
- run-id,
- reward metrics,
- hyperparameters,
- and logs

inside CSV files in the `experiments/` folder.

---

# Reproducibility

Experiments are reproducible using YAML configuration files.

To reproduce experiment 1:

```bash
python train.py configs/qlearn_v1.yaml
```

To reproduce experiment 2:

```bash
python train.py configs/qlearn_v2.yaml
```

The project uses fixed seeds for reproducible results.

---

# Monitoring Plan (Deployment Design)

If this system were deployed in a real microgrid, we would monitor the following:

- **Average grid import cost per hour**  
  Alert if cost exceeds 20% above the training baseline.

- **Battery SOC drift**  
  If battery stays below 20% for 3 or more consecutive hours, retraining may be required.

- **Renewable utilisation rate**  
  Track percentage of solar energy being self-consumed vs wasted.

- **Blackout frequency**  
  Any unserved load event triggers an immediate alert.

- **Grid price tier alignment**  
  Verify the agent discharges during peak pricing periods and charges during off-peak periods.

---

# Results

The trained RL agent demonstrates:
- reduced operational cost,
- improved renewable energy utilization,
- and better peak-hour energy management

when compared to the fixed heuristic controller.

Training curves show gradual convergence of reward over episodes.

---

# Limitations

Current limitations include:
- synthetic rather than real-world datasets,
- simplified demand and pricing profiles,
- no battery degradation modeling,
- and scalability limitations of tabular Q-learning.

---

# Future Improvements

Potential future enhancements include:
- Deep Q-Networks (DQN),
- real-world smart-grid datasets,
- cloud deployment,
- battery degradation modeling,
- and real-time IoT integration.

---

# Conclusion

This project successfully demonstrates the application of Reinforcement Learning and MLOps concepts for smart microgrid energy optimization.

The RL agent learns adaptive energy-management strategies capable of outperforming fixed-rule systems under varying pricing and demand conditions while supporting sustainable and efficient energy utilization aligned with SDG 7: Affordable and Clean Energy.