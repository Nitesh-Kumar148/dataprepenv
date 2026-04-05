import pandas as pd
import numpy as np

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
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            self.df[col] = np.where(self.df[col] < 0, 0, self.df[col])

    def cap_outliers(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
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
        reward = new_score - old_score

        return self.df, reward, old_score, new_score


class AutoAgent:
    def __init__(self, env):
        self.env = env
        self.actions = [
            "fill_missing_mean",
            "fill_missing_mode",
            "remove_rows",
            "fix_negative",
            "cap_outliers"
        ]

    def take_best_action(self):
        best_reward = -float('inf')
        best_action = None

        # Try all actions on a copy
        for action in self.actions:
            env_copy = DataPrepEnv(self.env.df.copy())
            _, reward, _, _ = env_copy.step(action)

            if reward > best_reward:
                best_reward = reward
                best_action = action

        # Stop if no improvement
        if best_reward <= 0:
            return None, 0

        # Apply best action to real environment
        _, actual_reward, _, _ = self.env.step(best_action)
        return best_action, actual_reward

    def run(self):
        steps = []

        while True:
            old_score = self.env.get_score()

            best_action, reward = self.take_best_action()
            if not best_action:
                break

            new_score = self.env.get_score()

            # ✅ FIX: return structured dict instead of string
            steps.append({
                "action": best_action,
                "reward": reward,
                "old_score": old_score,
                "new_score": new_score
            })

        return steps