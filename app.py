import streamlit as st
import pandas as pd
from backend import DataPrepEnv, AutoAgent

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="DataPrepEnv 🚀",
    layout="wide",
    page_icon="🧠"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
/* Overall app styling */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.2);
}

.big-title {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    color: #ffffff;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    margin-bottom: 30px;
}

/* Metric cards */
.metric-card {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 20px;
    margin: 10px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #ffffff;
}

.metric-label {
    font-size: 16px;
    color: #e0e0e0;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Button styling */
.stButton>button {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

/* Subheader styling */
h2 {
    color: #ffffff;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}

/* Info and success messages */
.stAlert {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 20px;
    border: 2px dashed rgba(255, 255, 255, 0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🚀 AI Data Cleaning Dashboard</p>', unsafe_allow_html=True)

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.title("⚙️ Controls")

action = st.sidebar.selectbox("Choose Action", [
    "fill_missing_mean",
    "fill_missing_mode",
    "remove_rows",
    "fix_negative",
    "cap_outliers"
])

apply_btn = st.sidebar.button("▶️ Apply Action")
run_ai = st.sidebar.button("🤖 Run AI Agent")

# ----------------------------
# Upload File
# ----------------------------
uploaded_file = st.file_uploader("📂 Upload your messy CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Store environment
    if "env" not in st.session_state:
        st.session_state.env = DataPrepEnv(df)
        st.session_state.score_history = []

    env = st.session_state.env

    # ✅ FIX: Compute issues and score BEFORE using them in the metrics card
    issues = env.get_issues()
    score = env.get_score()

    # ----------------------------
    # Top Metrics
    # ----------------------------
    st.markdown("""
    <div style="display: flex; justify-content: space-around; margin: 20px 0;">
        <div class="metric-card">
            <div class="metric-value">📉 {}</div>
            <div class="metric-label">Missing Values</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">⚠️ {}</div>
            <div class="metric-label">Errors</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">📈 {}</div>
            <div class="metric-label">Score</div>
        </div>
    </div>
    """.format(int(issues["missing"]), int(issues["negative"] + issues["outliers"]), score), unsafe_allow_html=True)

    # Append current score to history
    st.session_state.score_history.append(score)

    # ----------------------------
    # Data + Issues
    # ----------------------------
    with st.container():
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Dataset")
            with st.expander("View Full Dataset", expanded=True):
                def highlight(val):
                    if pd.isna(val):
                        return "background-color: rgba(255, 0, 0, 0.3)"
                    return ""

                st.dataframe(env.df.style.applymap(highlight), use_container_width=True)

        with col2:
            st.subheader("⚠️ Data Issues")
            with st.expander("Data Quality Report", expanded=True):
                for issue, count in issues.items():
                    st.markdown(f"**{issue.capitalize()}:** {int(count)}")
                st.markdown("---")
                st.markdown(f"**Overall Score:** {score}/100")

    # ----------------------------
    # Apply Manual Action
    # ----------------------------
    if apply_btn:
        _, reward, old_score, new_score = env.step(action)

        st.success(f"Action Applied: {action}")
        st.info(f"Reward: {reward}")
        st.write(f"Score: {old_score} → {new_score}")

    # ----------------------------
    # AI Agent
    # ----------------------------
    if run_ai:
        agent = AutoAgent(env)

        with st.spinner("🤖 AI is cleaning data..."):
            steps = agent.run()

        st.success("Cleaning Completed!")

        for step in steps:
            st.markdown(f"""
            ✅ **{step['action']}**  
            Reward: `{step['reward']}`  
            Score: `{step['old_score']} → {step['new_score']}`
            """)

    # ----------------------------
    # Score Graph
    # ----------------------------
    with st.container():
        st.markdown("---")
        st.subheader("📊 Score Improvement Over Time")
        with st.expander("Score History Chart", expanded=True):
            st.line_chart(st.session_state.score_history, use_container_width=True)

    # ----------------------------
    # Final Data
    # ----------------------------
    with st.container():
        st.markdown("---")
        st.subheader("🧼 Cleaned Dataset")
        with st.expander("View Cleaned Data", expanded=True):
            st.dataframe(env.df, use_container_width=True)

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                "📥 Download Cleaned Data",
                env.df.to_csv(index=False),
                "cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1 style="color: white; font-size: 48px;">👋 Welcome to DataPrepEnv!</h1>
        <p style="color: #e0e0e0; font-size: 20px;">Upload your messy CSV file above to start cleaning your data with AI-powered tools.</p>
        <div style="margin: 30px 0;">
            <div style="display: inline-block; background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px; margin: 10px;">
                🤖 <strong>AI Agent:</strong> Automatically clean your data
            </div>
            <div style="display: inline-block; background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius, 15px; margin: 10px;">
                📊 <strong>Real-time Metrics:</strong> Track data quality improvements
            </div>
            <div style="display: inline-block; background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px; margin: 10px;">
                🧼 <strong>One-click Download:</strong> Get your cleaned dataset
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
