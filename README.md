# NBA Second Half Prediction Model

This project implements a machine learning model to predict the total points in the second half of NBA games, with a focus on providing betting insights for over/under bets.

## Overview

The system works by:

1. **Training a model** on historical NBA game data, including first-half statistics and team-related features
2. **Predicting second half total points** for NBA games
3. **Analyzing betting performance** by comparing predictions against the halftime total line
4. **Visualizing results** to identify profitable betting strategies

## Project Structure

- `modeling/`: Code for training and creating prediction models
  - `xg2.py`: Main training script using XGBoost with date-based split
  - `models/`: Trained model files
  - `trainingdata/`: Training dataset
  - `testingdata/`: Testing dataset
- `model_testing/`: Scripts for evaluating model performance
  - `pregame_test.py`: Comprehensive script for evaluating model performance and betting strategy
  - `plots/`: Visualizations of model performance and betting analysis
  - `results/`: CSV files containing detailed betting analysis results

## Model Features

The model uses 82 features related to NBA games, including:
- First half statistics (points, rebounds, assists, etc.)
- Team travel distance
- Altitude of game venue
- Back-to-back game indicators
- Team offensive/defensive ratings
- Historical performance metrics

## Key Findings

Our analysis shows:
- **Overall Win Rate**: 60.60% on betting predictions (break-even is 52.4%)
- **ROI**: 15.69% assuming -110 odds
- **Best Performance**: 91.76% win rate on bets with 10+ point margins
- **Monthly Performance**: Profitable in all months, with October showing highest ROI
- **Prediction Types**: Over predictions (64.66%) outperform under predictions (57.86%)

## Visualizations

The project includes several useful visualizations:
- Win rate by prediction margin
- Monthly profit analysis
- Cumulative profit over time
- Error distribution
- Margin range analysis

## Requirements

```
numpy
pandas
xgboost
scikit-learn
matplotlib
seaborn
```

## Usage

### Training the Model

```
cd modeling
python xg2.py
```

This will:
- Load data from `../gamelog/oddsdatabase/proj_added.csv`
- Split data based on date (games before/after Oct 1, 2023)
- Train an XGBoost model
- Save the model to the `models/` directory

### Testing and Analyzing

```
cd model_testing
python pregame_test.py
```

This will:
- Load the most recent model
- Calculate performance metrics (MAE, RMSE, R²)
- Analyze betting performance
- Generate visualizations
- Save detailed results to CSV files

## Betting Strategy

Based on the results, we recommend:
1. Focus on bets with larger margins between the predicted total and halftime line
2. Consider a tiered betting approach based on confidence levels
3. Monitor monthly performance as some months show stronger results than others

## Sample Results

- 0-2 points margin: 54.46% win rate, 3.97% ROI
- 2-5 points margin: 54.73% win rate, 4.48% ROI
- 5-7 points margin: 62.96% win rate, 20.20% ROI
- 7-10 points margin: 63.91% win rate, 22.00% ROI
- 10+ points margin: 91.76% win rate, 75.19% ROI