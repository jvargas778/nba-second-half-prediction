#!/usr/bin/env python3
"""
NBA Second Half Total Points Model Test Script

This script:
1. Loads the most recent XGBoost model trained with xg2.py
2. Evaluates performance metrics (MAE, RMSE, R², MBE)
3. Analyzes error distribution (how often predictions are within 1, 2, 3 points, etc.)
4. Creates visualizations of the results
5. Analyzes betting win rate for over/under predictions
"""

import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
MODELS_DIR = os.path.join(BASE_DIR, "modeling", "models")
TEST_DATA_PATH = os.path.join(BASE_DIR, "modeling", "testingdata", "testing.csv")
TARGET = "TOTAL_2H_PTS"

# Features list from xg2.py
FEATURES = [
    "HOME_PTS_QTR1", "HOME_PTS_QTR2", 
    "HOME_1H_FGM", "HOME_1H_FGA", "HOME_1H_FG_PCT", 
    "HOME_1H_FG3M", "HOME_1H_FG3A", "HOME_1H_FG3_PCT", 
    "HOME_1H_FTM", "HOME_1H_FTA", "HOME_1H_FT_PCT", 
    "HOME_1H_OREB", "HOME_1H_DREB", "HOME_1H_REB", 
    "HOME_1H_AST", "HOME_1H_STL", "HOME_1H_BLK", 
    "HOME_1H_TO", "HOME_1H_PF", "HOME_1H_PTS",  "HOME_TEAM_TRAVEL_DIST", 
    "HOME_ALTITUDE", "HOME_PREV_ALTITUDE", "HOME_ATL_DIFF", 
    "HOME_2H_L5_H", "HOME_2H_L5_A", "HOME_1H_L5_H", 
    "HOME_1H_L5_A", "HOME_PACE_L5_H", "HOME_PACE_L5_A", 
    "HOME_L7_GP", "HOME_B2B", "HOME_HB2B",
    "AWAY_B2B", "AWAY_AB2B", "AWAY_PTS_QTR1", 
    "AWAY_PTS_QTR2", "AWAY_1H_FGM", "AWAY_1H_FGA", 
    "AWAY_1H_FG_PCT", "AWAY_1H_FG3M", "AWAY_1H_FG3A", 
    "AWAY_1H_FG3_PCT", "AWAY_1H_FTM", "AWAY_1H_FTA", 
    "AWAY_1H_FT_PCT", "AWAY_1H_OREB", "AWAY_1H_DREB", 
    "AWAY_1H_REB", "AWAY_1H_AST", "AWAY_1H_STL", 
    "AWAY_1H_BLK", "AWAY_1H_TO", "AWAY_1H_PF", 
    "AWAY_1H_PTS", "AWAY_TEAM_TRAVEL_DIST", 
    "AWAY_ALTITUDE", "AWAY_PREV_ALTITUDE", "AWAY_ATL_DIFF", 
    "AWAY_2H_L5_H", "AWAY_2H_L5_A", "AWAY_1H_L5_H", 
    "AWAY_1H_L5_A", "AWAY_PACE_L5_H", "AWAY_PACE_L5_A", 
    "AWAY_L7_GP","HOME_L7_ORTG","HOME_L7_DRTG","AWAY_L7_ORTG","AWAY_L7_DRTG","HOME_1H_L7_ORTG",
    "HOME_1H_L7_DRTG","AWAY_1H_L7_ORTG","AWAY_1H_L7_DRTG","ROTOWIRE_TOTAL","ROTOWIRE_HOME_SPREAD",
    "AVG_AWAY_2H_MARGIN","AVG_1H_TOTAL","AVG_2H_TOTAL", "PERCENTAGE_DIFF","PREDICTED_1H_TOTAL","PREDICTED_2H_TOTAL"]


def find_latest_model():
    """
    Find the most recently created model in the models directory
    
    Returns:
        str: Path to the latest model file
    """
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.json"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {MODELS_DIR}")
    
    # Sort by creation time (newest first)
    latest_model = max(model_files, key=os.path.getctime)
    
    return latest_model


def load_model(model_path):
    """
    Load the XGBoost model from the specified path
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        xgb.Booster: Loaded XGBoost model
    """
    print(f"Loading model from {model_path}...")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def load_test_data():
    """
    Load the test data and prepare it for prediction
    
    Returns:
        tuple: X_test (features), y_test (target), df (full DataFrame)
    """
    print(f"Loading test data from {TEST_DATA_PATH}...")
    df = pd.read_csv(TEST_DATA_PATH)
    print(f"Loaded {len(df)} test records")
    
    # Get target
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in test data")
    
    y_test = df[TARGET].copy()
    
    # Check which features are available in the dataframe
    available_features = [f for f in FEATURES if f in df.columns]
    missing_features = set(FEATURES) - set(available_features)
    
    if missing_features:
        print(f"Warning: {len(missing_features)} features not found in test dataset:")
        print(", ".join(missing_features))
    
    print(f"Using {len(available_features)} features for testing")
    
    # Get only the features needed for the model
    X_test = df[available_features].copy()
    
    # Exclude any non-numeric columns
    object_columns = X_test.select_dtypes(include=['object']).columns
    if len(object_columns) > 0:
        print(f"Dropping non-numeric columns: {list(object_columns)}")
        X_test = X_test.drop(columns=object_columns)
    
    # Handle missing values
    missing_count = X_test.isnull().sum().sum()
    if missing_count > 0:
        print(f"Filling {missing_count} missing feature values with median values")
        X_test = X_test.fillna(X_test.median())
    
    # Replace infinities with NaN, then fill with median
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.fillna(X_test.median())
    
    return X_test, y_test, df