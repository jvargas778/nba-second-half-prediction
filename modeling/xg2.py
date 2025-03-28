"""
Simple XGBoost Model for NBA Second Half Total Points Prediction - Date-Based Split Version

This script:
1. Loads the full dataset
2. Drops any rows with NaN values in any column
3. Splits it into training (before 10/01/2023) and testing (after 10/01/2023) sets
4. Saves these splits to CSV files
5. Trains an XGBoost model on the training data
6. Saves the trained model
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import datetime

# Input file and target variable
#INPUT_FILE = '../gamelog/oddsdatabase/fulldata.csv'
#INPUT_FILE = '../gamelog/oddsdatabase/rotowire_merged_data.csv'
INPUT_FILE = '../gamelog/oddsdatabase/proj_added.csv'
TARGET = "TOTAL_2H_PTS"

# Date for train/test split (MM/DD/YYYY)
SPLIT_DATE = "10/01/2023"

# Output directories
MODEL_DIR = 'models'
TRAINING_DIR = 'trainingdata'
TESTING_DIR = 'testingdata'

# Feature list from xg.py
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



def ensure_output_dirs():
    """
    Ensure that output directories exist
    """
    for directory in [MODEL_DIR, TRAINING_DIR, TESTING_DIR]:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)

def save_model(model):
    """
    Save the trained model
    
    Args:
        model (xgb.XGBRegressor): Trained XGBoost model
    """
    print("\nSaving model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{MODEL_DIR}/xgboost_2h_total_{timestamp}.json"
    
    model.save_model(model_filename)
    print(f"Model saved to {model_filename}")

def load_and_split_data():
    """
    Load the data, drop rows with any NaN values, split it into training and testing sets, and save the splits
    Training set: Games before 10/01/2023
    Testing set: Games on or after 10/01/2023
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names
    """
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records")
    
    # Check which features are available in the dataframe
    available_features = [f for f in FEATURES if f in df.columns]
    missing_features = set(FEATURES) - set(available_features)
    
    if missing_features:
        print(f"Warning: {len(missing_features)} features not found in dataset:")
        print(", ".join(missing_features))
    
    print(f"Using {len(available_features)} features for modeling")
    
    # Get target column
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset")
    
    # Check if GAME_DATE column exists
    if 'GAME_DATE' not in df.columns:
        raise ValueError("GAME_DATE column not found in the dataset. Cannot perform date-based split.")
    
    # Convert GAME_DATE to datetime for proper comparison
    print("Converting GAME_DATE to datetime format...")
    
    # Determine the date format and convert
    try:
        # Try standard MM/DD/YYYY format
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%m/%d/%Y', errors='raise')
        print("Converted GAME_DATE using format MM/DD/YYYY")
    except ValueError:
        try:
            # Try YYYY-MM-DD format
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%Y-%m-%d', errors='raise')
            print("Converted GAME_DATE using format YYYY-MM-DD")
        except ValueError:
            # Try automatic parsing as a last resort
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
            print("Converted GAME_DATE using automatic format detection")
            
            # Check if any dates were not converted properly
            if df['GAME_DATE'].isna().any():
                print(f"Warning: {df['GAME_DATE'].isna().sum()} dates could not be parsed")
    
    # Convert split date to datetime
    split_date = pd.to_datetime(SPLIT_DATE, format='%m/%d/%Y')
    print(f"Using {split_date.strftime('%Y-%m-%d')} as the train/test split date")
    
    # Create a subset with only the needed columns (features + target)
    needed_columns = available_features + [TARGET] + ['GAME_DATE']
    df_subset = df[needed_columns].copy()
    
    # Check for NaN values in the entire dataset
    initial_rows = len(df_subset)
    nan_count = df_subset.isnull().sum().sum()
    
    if nan_count > 0:
        print(f"Found {nan_count} NaN values across {initial_rows} rows")
        
        # Drop any rows with NaN values
        df_clean = df_subset.dropna()
        rows_dropped = initial_rows - len(df_clean)
        print(f"Dropped {rows_dropped} rows with NaN values ({(rows_dropped/initial_rows)*100:.2f}% of data)")
        
        # Replace the subset with the clean version
        df_subset = df_clean
    else:
        print("No NaN values found in dataset")
    
    # Split based on date
    train_df = df_subset[df_subset['GAME_DATE'] < split_date]
    test_df = df_subset[df_subset['GAME_DATE'] >= split_date]
    
    print(f"Split dataset based on date {split_date.strftime('%Y-%m-%d')}:")
    print(f"  Training set (before {split_date.strftime('%Y-%m-%d')}): {len(train_df)} games")
    print(f"  Testing set (on or after {split_date.strftime('%Y-%m-%d')}): {len(test_df)} games")
    
    # Check if any of the sets is empty
    if len(train_df) == 0:
        raise ValueError("Training set is empty. Please check your date split criteria.")
    
    if len(test_df) == 0:
        raise ValueError("Testing set is empty. Please check your date split criteria.")
    
    # Get features and target from the split dataset
    X_train = train_df[available_features].copy()
    y_train = train_df[TARGET].copy()
    X_test = test_df[available_features].copy()
    y_test = test_df[TARGET].copy()
    
    # Replace infinities with NaN, then drop those rows too
    X_train_with_inf = X_train.replace([np.inf, -np.inf], np.nan)
    inf_rows = X_train_with_inf.isnull().any(axis=1).sum()
    
    if inf_rows > 0:
        print(f"Found {inf_rows} rows with infinite values in training data")
        # Identify rows with inf values
        inf_mask = X_train_with_inf.isnull().any(axis=1)
        # Keep only rows without inf values
        X_train = X_train[~inf_mask].copy()
        y_train = y_train[~inf_mask].copy()
        print(f"Dropped {inf_rows} rows with infinite values from training data")
    
    X_test_with_inf = X_test.replace([np.inf, -np.inf], np.nan)
    inf_rows = X_test_with_inf.isnull().any(axis=1).sum()
    
    if inf_rows > 0:
        print(f"Found {inf_rows} rows with infinite values in testing data")
        # Identify rows with inf values
        inf_mask = X_test_with_inf.isnull().any(axis=1)
        # Keep only rows without inf values
        X_test = X_test[~inf_mask].copy()
        y_test = y_test[~inf_mask].copy()
        print(f"Dropped {inf_rows} rows with infinite values from testing data")
    
    # Handle any non-numeric columns in training data
    for col in X_train.select_dtypes(include=['object']).columns:
        print(f"Converting categorical column {col} in training data to numeric")
        X_train[col] = pd.factorize(X_train[col])[0]
    
    # Handle any non-numeric columns in testing data
    for col in X_test.select_dtypes(include=['object']).columns:
        print(f"Converting categorical column {col} in testing data to numeric")
        X_test[col] = pd.factorize(X_test[col])[0]
    
    # Final verification - ensure no NaN values remain
    train_nan_count = X_train.isnull().sum().sum() + y_train.isnull().sum()
    test_nan_count = X_test.isnull().sum().sum() + y_test.isnull().sum()
    
    if train_nan_count > 0 or test_nan_count > 0:
        raise ValueError(f"Found NaN values after cleaning: {train_nan_count} in training, {test_nan_count} in testing")
    
    # Create full train/test dataframes with all columns for saving
    # Add back the split date column to the original dataframe for reference
    df['split_date'] = pd.to_datetime(SPLIT_DATE, format='%m/%d/%Y')
    df['is_training'] = df['GAME_DATE'] < df['split_date']
    
    # Filter the original dataframe based on the date
    train_df_full = df[df['GAME_DATE'] < split_date].copy()
    test_df_full = df[df['GAME_DATE'] >= split_date].copy()
    
    # Save training and testing data to CSV
    train_output_path = f"{TRAINING_DIR}/training.csv"
    test_output_path = f"{TESTING_DIR}/testing.csv"
    
    print(f"Saving training data ({len(train_df_full)} rows) to {train_output_path}")
    train_df_full.to_csv(train_output_path, index=False)
    
    print(f"Saving testing data ({len(test_df_full)} rows) to {test_output_path}")
    test_df_full.to_csv(test_output_path, index=False)
    
    print(f"Final training set: {X_train.shape[0]} samples")
    print(f"Final test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, available_features

def train_model(X_train, y_train):
    """
    Train an XGBoost model
    
    Args:
        X_train, y_train: Training data
        
    Returns:
        xgb.XGBRegressor: Trained model
    """
    print("Training XGBoost model...")
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 7,
        'learning_rate': 0.03,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.5,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'verbose': 100
    }
    
    # Create validation set for early stopping (20% of training data)
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42
    )
    
    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    print(f"Model trained with {model.best_iteration} boosting rounds")
    return model

def main():
    """
    Main function to run the NBA Second Half Total Points Prediction Model
    """
    print("Starting NBA Second Half Total Points Prediction Model - Date-Based Split Version...")
    print(f"Using {SPLIT_DATE} as the train/test split date")
    
    # Ensure output directories exist
    ensure_output_dirs()
    
    # Load and split data
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Save the model
    save_model(model)
    
    print("\nData split and model training complete!")

if __name__ == "__main__":
    main()