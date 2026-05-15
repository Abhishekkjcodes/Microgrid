import streamlit as st
import subprocess
import os
import pandas as pd
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Microgrid Dashboard",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⚡ Smart Energy Microgrid Optimizer")
st.markdown("### Reinforcement Learning + MLOps Dashboard")

st.write(
    "This dashboard visualizes the RL-based smart microgrid project."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Controls")

config_choice = st.sidebar.selectbox(
    "Choose Config",
    (
        "configs/qlearn_v1.yaml",
        "configs/qlearn_v2.yaml"
    )
)

# --------------------------------------------------
# FLOWCHART SECTION
# --------------------------------------------------

st.header("🧠 Simulator + RL Workflow")

with st.expander("Show Complete RL Simulation Flowchart"):

    flowchart_path = "image.png"

    if os.path.exists(flowchart_path):

        flowchart = Image.open(flowchart_path)

        st.image(
            flowchart,
            caption="Complete RL + Microgrid Simulation Workflow",
            use_container_width=True
        )

    else:

        st.warning(
            "flowchart.png not found. Place the flowchart image in the project root folder."
        )

# --------------------------------------------------
# TRAIN BUTTON
# --------------------------------------------------

if st.sidebar.button("🚀 Train RL Agent"):

    with st.spinner("Training RL Agent..."):

        process = subprocess.run(
            [os.sys.executable, "train.py", config_choice],
            capture_output=True,
            text=True
        )

    if process.returncode == 0:

        st.success("Training completed successfully!")

        st.subheader("Training Logs")

        st.code(process.stdout)

    else:

        st.error("Training failed")

        st.code(process.stderr)

# --------------------------------------------------
# EVALUATION BUTTON
# --------------------------------------------------

if st.sidebar.button("📊 Run Evaluation"):

    with st.spinner("Running Evaluation..."):

        process = subprocess.run(
            [os.sys.executable, "evaluate.py"],
            capture_output=True,
            text=True
        )

    if process.returncode == 0:

        st.success("Evaluation completed successfully!")

        st.subheader("Evaluation Output")

        st.code(process.stdout)

    else:

        st.error("Evaluation failed")

        st.code(process.stderr)

# --------------------------------------------------
# TRAINING CURVE
# --------------------------------------------------

st.header("📈 Training Curve")

training_curve = os.path.join(
    "experiments",
    "training_curve.png"
)

if os.path.exists(training_curve):

    image = Image.open(training_curve)

    st.image(
        image,
        caption="Training Reward Curve",
        use_container_width=True
    )

else:

    st.warning("training_curve.png not found")

# --------------------------------------------------
# COMPARISON PLOT
# --------------------------------------------------

st.header("⚖ RL Agent vs Fixed Controller")

comparison_plot = os.path.join(
    "experiments",
    "comparison_plot.png"
)

if os.path.exists(comparison_plot):

    image = Image.open(comparison_plot)

    st.image(
        image,
        caption="RL vs Fixed Controller",
        use_container_width=True
    )

else:

    st.warning("comparison_plot.png not found")

# --------------------------------------------------
# EXPERIMENT CSV DISPLAY
# --------------------------------------------------

st.header("🧪 Experiment Tracking")

csv_options = []

exp_folder = "experiments"

if os.path.exists(exp_folder):

    for file in os.listdir(exp_folder):

        if file.endswith(".csv"):

            csv_options.append(file)

if len(csv_options) > 0:

    selected_csv = st.selectbox(
        "Choose CSV File",
        csv_options
    )

    csv_path = os.path.join(
        exp_folder,
        selected_csv
    )

    df = pd.read_csv(csv_path)

    st.dataframe(df)

    # Reward Graph

    if "episode" in df.columns and "avg_reward" in df.columns:

        st.subheader("📉 Average Reward Trend")

        chart_df = df[
            ["episode", "avg_reward"]
        ].set_index("episode")

        st.line_chart(chart_df)

else:

    st.warning("No CSV experiment files found")

# --------------------------------------------------
# RL DETAILS
# --------------------------------------------------

st.header("📚 RL Methodology")

col1, col2 = st.columns(2)

with col1:

    st.subheader("State Space")

    st.markdown(
        """
- Battery SOC
- Solar Output
- Load Demand
- Grid Price
        """
    )

    st.subheader("Action Space")

    st.markdown(
        """
- Charge Battery
- Discharge Battery
- Import from Grid
- Shed Load
        """
    )

with col2:

    st.subheader("Algorithm")

    st.write(
        "Q-Learning was chosen because the environment uses a discrete state and action space."
    )

    st.subheader("Reward Function")

    st.write(
        "Reward is defined as the negative operational cost."
    )

# --------------------------------------------------
# MONITORING PLAN
# --------------------------------------------------

st.header("🛠 Monitoring Plan")

st.markdown(
    """
If deployed in the real world, the following metrics would be monitored:

- Grid import cost
- Battery SOC drift
- Renewable energy utilization
- Blackout frequency
- Peak-hour policy behaviour
    """
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.write(
    "Smart Microgrid RL + MLOps Dashboard"
)