import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from io import StringIO, BytesIO

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Data Cleaning with Gemini", layout="wide")

# -------------------------------
# STYLING
# -------------------------------
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

# -------------------------------
# TITLE
# -------------------------------
st.title("🧠 AI Data Cleaning Environment with Gemini AI")

# -------------------------------
# BACKEND CONFIGURATION
# -------------------------------
BACKEND_URL = "http://localhost:5000/clean"

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.subheader("📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    
    # Read file based on type
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Store in session state
    if "original_df" not in st.session_state:
        st.session_state.original_df = df.copy()
        st.session_state.df = df.copy()
        st.session_state.cleaning_history = []
        st.session_state.total_reward = 0
        st.session_state.gemini_used = False
    
    df = st.session_state.df
    original_df = st.session_state.original_df
    
    # -------------------------------
    # ISSUE DETECTION FUNCTION
    # -------------------------------
    def detect_issues(data):
        missing = data.isnull().sum().sum()
        numeric_df = data.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            negative = (numeric_df < 0).sum().sum()
            outliers = (numeric_df > 100).sum().sum()
        else:
            negative = 0
            outliers = 0
        return {
            "missing": int(missing),
            "negative": int(negative),
            "outliers": int(outliers)
        }
    
    # -------------------------------
    # SCORE CALCULATION
    # -------------------------------
    def calculate_score(data):
        issues = detect_issues(data)
        score = 100 - (issues["missing"] * 2 + issues["negative"] * 3 + issues["outliers"] * 2)
        return max(score, 0)
    
    # -------------------------------
    # METRICS DISPLAY
    # -------------------------------
    st.subheader("📊 Data Quality Overview")
    
    issues = detect_issues(df)
    current_score = calculate_score(df)
    original_score = calculate_score(original_df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Missing Values", issues["missing"])
    col2.metric("Negative Values", issues["negative"])
    col3.metric("Outliers (>100)", issues["outliers"])
    col4.metric("Quality Score", f"{current_score}/100", delta=f"+{current_score - original_score}" if current_score != original_score else None)
    
    # -------------------------------
    # PROGRESS BAR
    # -------------------------------
    st.progress(current_score / 100)
    
    # -------------------------------
    # CHARTS
    # -------------------------------
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
    
    # -------------------------------
    # MANUAL ACTIONS
    # -------------------------------
    st.subheader("⚙️ Manual Cleaning Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Fill Missing Values (Mean)"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            st.session_state.df = df
            st.success("✅ Missing values filled with mean!")
            st.rerun()
    
    with col2:
        if st.button("🔧 Fix Negative Values (Set to 0)"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].clip(lower=0)
            st.session_state.df = df
            st.success("✅ Negative values fixed!")
            st.rerun()
    
    with col3:
        if st.button("📈 Cap Outliers (Set to 100)"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].clip(upper=100)
            st.session_state.df = df
            st.success("✅ Outliers capped!")
            st.rerun()
    
    # -------------------------------
    # GEMINI AI AUTO CLEAN
    # -------------------------------
    st.subheader("🤖 Gemini AI Auto Clean")
    st.markdown("*Let Gemini AI intelligently decide the best cleaning actions*")
    
    if st.button("🚀 Run Gemini AI Agent", type="primary"):
        with st.spinner("🤖 Gemini AI is analyzing and cleaning your data..."):
            try:
                # Save current dataframe to CSV for sending
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                files = {'file': ('data.csv', csv_buffer.getvalue(), 'text/csv')}
                
                # Send to backend
                response = requests.post(BACKEND_URL, files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['success']:
                        # Update dataframe
                        cleaned_df = pd.DataFrame(result['cleaned_data'])
                        st.session_state.df = cleaned_df
                        st.session_state.gemini_used = True
                        
                        # Display actions taken
                        st.success(f"✅ Gemini AI completed {len(result['actions'])} actions!")
                        
                        # Show detailed actions
                        st.subheader("📋 Actions Performed by Gemini AI")
                        for action in result['actions']:
                            reward_class = "reward-positive" if action['reward'] > 0 else "reward-negative"
                            st.markdown(f"""
                            <div style='padding: 10px; margin: 5px 0; background-color: #2d2d2d; border-radius: 5px;'>
                                <b>Action:</b> {action['action']}<br>
                                <b>Reward:</b> <span class='{reward_class}'>+{action['reward']}</span><br>
                                <b>Score Change:</b> {action['old_score']} → {action['new_score']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Calculate total reward
                        total_reward = sum([a['reward'] for a in result['actions']])
                        st.metric("Total Reward", f"+{total_reward}", delta=f"Score: {result['final_score']}")
                        
                        st.rerun()
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
                else:
                    st.error(f"Backend error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to Gemini backend. Make sure Flask server is running on http://localhost:5000")
                st.info("💡 Run this command in another terminal: python flask_backend.py")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # -------------------------------
    # RESET BUTTON
    # -------------------------------
    if st.button("🔄 Reset to Original Data"):
        st.session_state.df = st.session_state.original_df.copy()
        st.session_state.cleaning_history = []
        st.session_state.total_reward = 0
        st.session_state.gemini_used = False
        st.success("✅ Reset to original data!")
        st.rerun()
    
    # -------------------------------
    # DISPLAY DATA
    # -------------------------------
    st.subheader("📈 Current Dataset")
    
    # Show data preview
    st.dataframe(st.session_state.df, use_container_width=True, height=400)
    
    # Show statistics
    with st.expander("📊 Dataset Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Shape:**", st.session_state.df.shape)
            st.write("**Columns:**", list(st.session_state.df.columns))
        with col2:
            st.write("**Data Types:**")
            st.write(st.session_state.df.dtypes)
    
    # -------------------------------
    # DOWNLOAD CLEANED DATA
    # -------------------------------
    st.subheader("💾 Download Cleaned Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Download
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel Download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.df.to_excel(writer, index=False, sheet_name='Cleaned Data')
        excel_data = output.getvalue()
        st.download_button(
            label="📥 Download as Excel",
            data=excel_data,
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # -------------------------------
    # CLEANING REPORT
    # -------------------------------
    if st.session_state.gemini_used:
        st.subheader("📄 Cleaning Report")
        
        final_issues = detect_issues(st.session_state.df)
        
        report = f"""
        ### Summary Report
        
        **Original Issues:**
        - Missing Values: {detect_issues(st.session_state.original_df)['missing']}
        - Negative Values: {detect_issues(st.session_state.original_df)['negative']}
        - Outliers: {detect_issues(st.session_state.original_df)['outliers']}
        
        **Final Issues:**
        - Missing Values: {final_issues['missing']}
        - Negative Values: {final_issues['negative']}
        - Outliers: {final_issues['outliers']}
        
        **Improvement:**
        - Score Improvement: +{current_score - original_score} points
        - Total Issues Fixed: {(detect_issues(st.session_state.original_df)['missing'] + detect_issues(st.session_state.original_df)['negative'] + detect_issues(st.session_state.original_df)['outliers']) - (final_issues['missing'] + final_issues['negative'] + final_issues['outliers'])}
        """
        
        st.markdown(report)
        
        # Download report
        st.download_button(
            label="📄 Download Cleaning Report",
            data=report,
            file_name="cleaning_report.txt",
            mime="text/plain"
        )

else:
    st.info("👆 Please upload a CSV or Excel file to start")

# -------------------------------
# SIDEBAR - INSTRUCTIONS
# -------------------------------
with st.sidebar:
    st.markdown("## 📖 How to Use")
    st.markdown("""
    1. **Upload** your CSV/Excel file
    2. **View** data quality metrics
    3. **Choose** manual cleaning or AI auto-clean
    4. **Download** cleaned data
    
    ### 🤖 Gemini AI Agent
    - Intelligently analyzes data issues
    - Chooses best cleaning actions
    - Shows rewards and score improvements
    - Explains what was done
    
    ### 🎯 Score System
    - Missing values: -2 points each
    - Negative values: -3 points each  
    - Outliers (>100): -2 points each
    - Higher score = cleaner data
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Backend Required")
    st.code("""
    # Run Flask backend first:
    python flask_backend.py
    """)
