import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO, BytesIO

st.set_page_config(page_title="AI Data Cleaning", layout="wide")

st.markdown("""
    <style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    .reward-positive {
        color: #00ff00;
        font-weight: bold;
    }
    .reward-negative {
        color: #ff0000;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI Data Cleaning Environment")

# ============================================
# INTELLIGENT AGENT (No Gemini, No Flask)
# ============================================
class IntelligentAgent:
    def __init__(self, df):
        self.df = df.copy()
        self.actions_taken = []
        self.total_reward = 0
        
    def detect_issues(self, data=None):
        if data is None:
            data = self.df
        missing = data.isnull().sum().sum()
        numeric_df = data.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            negative = (numeric_df < 0).sum().sum()
            outliers = (numeric_df > 100).sum().sum()
        else:
            negative = 0
            outliers = 0
        return {"missing": missing, "negative": negative, "outliers": outliers}
    
    def calculate_score(self, data=None):
        if data is None:
            data = self.df
        issues = self.detect_issues(data)
        score = 100 - (issues["missing"] * 2 + issues["negative"] * 3 + issues["outliers"] * 2)
        return max(score, 0)
    
    def fill_missing_mean(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            return True
        return False
    
    def fill_missing_mode(self):
        for col in self.df.columns:
            if self.df[col].isnull().any():
                mode_val = self.df[col].mode()
                if len(mode_val) > 0:
                    self.df[col] = self.df[col].fillna(mode_val[0])
        return True
    
    def remove_rows(self):
        before = len(self.df)
        self.df = self.df.dropna()
        after = len(self.df)
        return before > after
    
    def fix_negative(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.df[numeric_cols] = self.df[numeric_cols].clip(lower=0)
            return True
        return False
    
    def cap_outliers(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.df[numeric_cols] = self.df[numeric_cols].clip(upper=100)
            return True
        return False
    
    def decide_next_action(self):
        """Intelligent decision making - priority based"""
        issues = self.detect_issues()
        
        # Priority 1: Fix negatives (highest penalty)
        if issues["negative"] > 0:
            return "fix_negative", 3  # High priority
            
        # Priority 2: Cap outliers
        if issues["outliers"] > 0:
            return "cap_outliers", 2  # Medium priority
            
        # Priority 3: Fill missing values
        if issues["missing"] > 0:
            # Choose best method based on data type
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            
            if len(numeric_cols) > 0 and self.df[numeric_cols].isnull().any().any():
                return "fill_missing_mean", 1
            elif len(categorical_cols) > 0:
                return "fill_missing_mode", 1
            else:
                return "remove_rows", 0
                
        return None, 0
    
    def run(self, max_steps=10):
        self.actions_taken = []
        self.total_reward = 0
        
        for step_num in range(max_steps):
            old_score = self.calculate_score()
            action, priority = self.decide_next_action()
            
            if action is None:
                break
                
            # Apply action
            if action == "fill_missing_mean":
                self.fill_missing_mean()
            elif action == "fill_missing_mode":
                self.fill_missing_mode()
            elif action == "remove_rows":
                self.remove_rows()
            elif action == "fix_negative":
                self.fix_negative()
            elif action == "cap_outliers":
                self.cap_outliers()
            
            new_score = self.calculate_score()
            reward = new_score - old_score
            self.total_reward += reward
            
            self.actions_taken.append({
                "step": step_num + 1,
                "action": action,
                "reward": reward,
                "old_score": old_score,
                "new_score": new_score,
                "priority": priority
            })
            
            # Stop if no improvement
            if reward <= 0:
                break
                
        return self.actions_taken, self.total_reward, self.df

# ============================================
# UI CODE
# ============================================

st.subheader("📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    if "original_df" not in st.session_state:
        st.session_state.original_df = df.copy()
        st.session_state.df = df.copy()
        st.session_state.cleaned = False
    
    df = st.session_state.df
    original_df = st.session_state.original_df
    
    def detect_issues(data):
        missing = data.isnull().sum().sum()
        numeric_df = data.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            negative = (numeric_df < 0).sum().sum()
            outliers = (numeric_df > 100).sum().sum()
        else:
            negative = 0
            outliers = 0
        return {"missing": int(missing), "negative": int(negative), "outliers": int(outliers)}
    
    def calculate_score(data):
        issues = detect_issues(data)
        score = 100 - (issues["missing"] * 2 + issues["negative"] * 3 + issues["outliers"] * 2)
        return max(score, 0)
    
    st.subheader("📊 Data Quality Overview")
    
    issues = detect_issues(df)
    current_score = calculate_score(df)
    original_score = calculate_score(original_df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Missing Values", issues["missing"])
    col2.metric("Negative Values", issues["negative"])
    col3.metric("Outliers (>100)", issues["outliers"])
    col4.metric("Quality Score", f"{current_score}/100", 
                delta=f"+{current_score - original_score}" if current_score != original_score else None)
    
    st.progress(current_score / 100)
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        chart_data = pd.DataFrame({
            "Issue Type": ["Missing", "Negative", "Outliers"],
            "Count": [issues["missing"], issues["negative"], issues["outliers"]]
        })
        st.bar_chart(chart_data.set_index("Issue Type"))
    
    with col2:
        score_comparison = pd.DataFrame({
            "Status": ["Original", "Current"],
            "Score": [original_score, current_score]
        })
        st.bar_chart(score_comparison.set_index("Status"))
    
    # Manual Actions
    st.subheader("⚙️ Manual Cleaning Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Fill Missing (Mean)"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            st.session_state.df = df
            st.success("✅ Missing values filled!")
            st.rerun()
    
    with col2:
        if st.button("🔧 Fix Negatives"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].clip(lower=0)
            st.session_state.df = df
            st.success("✅ Negative values fixed!")
            st.rerun()
    
    with col3:
        if st.button("📈 Cap Outliers"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].clip(upper=100)
            st.session_state.df = df
            st.success("✅ Outliers capped!")
            st.rerun()
    
    # AI Agent Button
    st.subheader("🤖 AI Auto Clean Agent")
    st.markdown("*Intelligent agent automatically fixes all issues in priority order*")
    
    if st.button("🚀 Run AI Agent", type="primary"):
        with st.spinner("🤖 AI Agent is cleaning your data..."):
            agent = IntelligentAgent(st.session_state.df)
            actions, total_reward, cleaned_df = agent.run()
            
            st.session_state.df = cleaned_df
            st.session_state.cleaned = True
            
            # Display results
            st.success(f"✅ AI Agent completed {len(actions)} actions! Total Reward: +{total_reward}")
            
            st.subheader("📋 Actions Performed")
            for action in actions:
                reward_class = "reward-positive" if action['reward'] > 0 else "reward-negative"
                st.markdown(f"""
                <div style='padding: 10px; margin: 5px 0; background-color: #2d2d2d; border-radius: 5px;'>
                    <b>Step {action['step']}:</b> {action['action']}<br>
                    <b>Reward:</b> <span class='{reward_class}'>+{action['reward']}</span><br>
                    <b>Score:</b> {action['old_score']} → {action['new_score']}
                </div>
                """, unsafe_allow_html=True)
            
            st.rerun()
    
    # Reset Button
    if st.button("🔄 Reset to Original"):
        st.session_state.df = st.session_state.original_df.copy()
        st.session_state.cleaned = False
        st.success("✅ Reset complete!")
        st.rerun()
    
    # Display Data
    st.subheader("📈 Current Dataset")
    st.dataframe(st.session_state.df, use_container_width=True, height=400)
    
    # Download Section
    st.subheader("💾 Download Cleaned Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = st.session_state.df.to_csv(index=False)
        st.download_button("📥 Download as CSV", csv, "cleaned_data.csv", "text/csv", use_container_width=True)
    
    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.df.to_excel(writer, index=False, sheet_name='Cleaned Data')
        st.download_button("📥 Download as Excel", output.getvalue(), "cleaned_data.xlsx", 
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                          use_container_width=True)
    
    # Report
    if st.session_state.cleaned:
        st.subheader("📄 Cleaning Report")
        final_issues = detect_issues(st.session_state.df)
        original_issues = detect_issues(st.session_state.original_df)
        
        report = f"""
        ### Summary Report
        
        **Issues Fixed:**
        - Missing: {original_issues['missing']} → {final_issues['missing']}
        - Negative: {original_issues['negative']} → {final_issues['negative']}
        - Outliers: {original_issues['outliers']} → {final_issues['outliers']}
        
        **Score Improvement:** {original_score} → {current_score} (+{current_score - original_score})
        """
        st.markdown(report)
        st.download_button("📄 Download Report", report, "cleaning_report.txt")

else:
    st.info("👆 Please upload a CSV or Excel file to start")

with st.sidebar:
    st.markdown("## 📖 How to Use")
    st.markdown("""
    1. **Upload** your CSV/Excel file
    2. **View** data quality metrics
    3. **Manual Clean** - Use individual buttons
    4. **AI Auto Clean** - Let agent fix everything
    5. **Download** cleaned data
    
    ### 🤖 AI Agent Strategy
    1. Fix negative values (highest penalty)
    2. Cap outliers (>100)
    3. Fill missing values
    
    ### 🎯 Score System
    - Missing: -2 points each
    - Negatives: -3 points each
    - Outliers: -2 points each
    """)
