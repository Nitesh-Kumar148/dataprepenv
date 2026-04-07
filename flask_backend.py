from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)

# REPLACE WITH YOUR ACTUAL API KEY
genai.configure(api_key="AIzaSyCbtuubmQO00E-1hfMcmZXBbkyKci6hKn0")
model = genai.GenerativeModel('gemini-pro')

class DataPrepEnv:
    def __init__(self, df):
        self.df = df.copy()

    def get_issues(self):
        missing = self.df.isnull().sum().sum()
        numeric_df = self.df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            negative = (numeric_df < 0).sum().sum()
            outliers = (numeric_df > 100).sum().sum()
        else:
            negative = 0
            outliers = 0
        return {"missing": missing, "negative": negative, "outliers": outliers}

    def get_score(self):
        issues = self.get_issues()
        return -(issues["missing"] * 2 + issues["negative"] * 3 + issues["outliers"] * 2)

    def fill_missing_mean(self):
        numeric_df = self.df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            self.df[numeric_df.columns] = numeric_df.fillna(numeric_df.mean())

    def fill_missing_mode(self):
        for col in self.df.columns:
            mode_series = self.df[col].mode()
            if not mode_series.empty:
                self.df[col] = self.df[col].fillna(mode_series[0])

    def remove_rows(self):
        self.df = self.df.dropna()

    def fix_negative(self):
        for col in self.df.select_dtypes(include=[np.number]).columns:
            self.df[col] = np.where(self.df[col] < 0, 0, self.df[col])

    def cap_outliers(self):
        for col in self.df.select_dtypes(include=[np.number]).columns:
            self.df[col] = np.where(self.df[col] > 100, 100, self.df[col])

    def step(self, action):
        old_score = self.get_score()
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
        new_score = self.get_score()
        return self.df, new_score - old_score, old_score, new_score

class GeminiAgent:
    def __init__(self, env):
        self.env = env
        self.actions = ["fill_missing_mean", "fill_missing_mode", "remove_rows", "fix_negative", "cap_outliers"]

    def get_action(self):
        issues = self.env.get_issues()
        
        if issues['negative'] > 0:
            return "fix_negative"
        if issues['outliers'] > 0:
            return "cap_outliers"
        if issues['missing'] > 0:
            return "fill_missing_mean"
            
        try:
            prompt = f"Missing:{issues['missing']} Negative:{issues['negative']} Outliers:{issues['outliers']} Score:{self.env.get_score()}. Choose: fill_missing_mean, fill_missing_mode, remove_rows, fix_negative, cap_outliers. Reply only action name:"
            response = model.generate_content(prompt, timeout=3)
            action = response.text.strip().strip('"').strip("'")
            if action in self.actions:
                return action
        except:
            pass
        return None

    def run(self):
        steps = []
        for _ in range(5):
            old_score = self.env.get_score()
            action = self.get_action()
            if not action:
                break
            _, reward, _, new_score = self.env.step(action)
            steps.append({"action": action, "reward": reward, "old_score": old_score, "new_score": new_score})
            if reward <= 0:
                break
        return steps

@app.route('/clean', methods=['POST'])
def clean_uploaded_file():
    try:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Only CSV and Excel files allowed"}), 400
        
        env = DataPrepEnv(df)
        agent = GeminiAgent(env)
        steps = agent.run()
        
        cleaned_json = env.df.replace({np.nan: None}).to_dict(orient='records')
        
        return jsonify({
            "success": True,
            "actions": steps,
            "cleaned_data": cleaned_json,
            "final_score": env.get_score()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
