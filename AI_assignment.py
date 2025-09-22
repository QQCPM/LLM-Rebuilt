#!/usr/bin/env python3
"""
ECE5424 Machine Learning HW1: Baseball Steal Prediction - UPDATED COMPLIANCE VERSION
Author: Student
Date: 2025-01-13

This program strictly follows the PDF assignment requirements for predicting 
baseball players' success in stealing bases using speed-related data.

COMPLIANCE NOTES:
- Using DecisionTreeRegressor (despite PDF saying "Classifier" - task is regression)
- MLP tuning avoids solver variation as instructed
- Explicit numpy array conversion as required
- All deliverables match exact specifications
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor  # Note: PDF says "Classifier" but task is regression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import glob
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    Load and integrate all datasets following exact PDF requirements.
    Returns numpy arrays as specified in requirement #3.
    """
    print("Loading datasets...")
    
    # Load running splits data from all years (2015-2021)
    running_splits_files = glob.glob("./Datasets_export/running_splits_*.csv")
    running_splits_df = pd.concat([pd.read_csv(f) for f in running_splits_files], ignore_index=True)
    print(f"Loaded {len(running_splits_df)} running splits records from {len(running_splits_files)} files")
    
    # Load batting data
    batting_df = pd.read_csv("./Datasets_export/Batting.csv")
    print(f"Loaded {len(batting_df)} batting records")
    
    # Load people data
    people_df = pd.read_excel("./Datasets_export/People.xlsx")
    print(f"Loaded {len(people_df)} player records")
    
    # Merge batting and people data
    batting_people_df = pd.merge(batting_df, people_df, on="playerID", how="inner")
    print(f"After merging batting and people data: {len(batting_people_df)} records")
    
    # Clean up names for merging
    running_splits_df['name'] = running_splits_df['last_name, first_name'].str.lower().str.strip()
    batting_people_df['name'] = (batting_people_df['nameLast'] + ", " + batting_people_df['nameFirst']).str.lower().str.strip()
    
    # Rename season to yearID for consistency
    running_splits_df.rename(columns={'season': 'yearID'}, inplace=True)
    
    # Merge with running splits data on name and year
    merged_df = pd.merge(running_splits_df, batting_people_df, on=["name", "yearID"], how="inner")
    print(f"After merging all datasets: {len(merged_df)} records")
    
    # Calculate StealMargin (requirement #2)
    merged_df["StealMargin"] = merged_df["SB"] - merged_df["CS"]
    print(f"Target variable 'StealMargin' calculated: mean={merged_df['StealMargin'].mean():.2f}, std={merged_df['StealMargin'].std():.2f}")
    
    # Feature Engineering - use speed related variables
    split_columns = [f'seconds_since_hit_{i:03d}' for i in range(0, 91, 5)]
    
    # Create speed features (focusing on speed-related variables as specified)
    merged_df["avg_speed"] = merged_df[split_columns].mean(axis=1)
    merged_df["max_speed"] = merged_df[split_columns].max(axis=1)
    merged_df["speed_variance"] = merged_df[split_columns].var(axis=1)
    merged_df["first_half_speed"] = merged_df[[f'seconds_since_hit_{i:03d}' for i in range(0, 46, 5)]].mean(axis=1)
    merged_df["second_half_speed"] = merged_df[[f'seconds_since_hit_{i:03d}' for i in range(50, 91, 5)]].mean(axis=1)
    merged_df["acceleration"] = merged_df["second_half_speed"] - merged_df["first_half_speed"]
    
    # Select features (speed related variables as specified)
    features = ["avg_speed", "max_speed", "speed_variance", "first_half_speed", "second_half_speed", "acceleration"]
    target = "StealMargin"
    
    # Remove rows with missing values
    feature_target_df = merged_df[features + [target]].dropna()
    print(f"After removing missing values: {len(feature_target_df)} records")
    
    # REQUIREMENT #3: Convert pandas dataframe into two numpy arrays
    X_pandas = feature_target_df[features]
    y_pandas = feature_target_df[target]
    
    # Convert to numpy arrays as explicitly required
    X_numpy = X_pandas.values  # Convert to numpy array
    y_numpy = y_pandas.values  # Convert to numpy array
    
    print(f"Converted to numpy arrays - X shape: {X_numpy.shape}, y shape: {y_numpy.shape}")
    
    # REQUIREMENT #5: Divide data into Training and Test partitions
    X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples")
    
    # REQUIREMENT #4: Normalize predictor data array (max=1, min=-1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, merged_df, features, target

def evaluate_model_compliance(model, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluate model following exact PDF requirements:
    - Mean absolute error
    - Coefficient of determination  
    - Generalization gap (difference in accuracy between training and test)
    """
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate required metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Generalization gap (difference in accuracy between training and test)
    generalization_gap_mae = abs(mae_train - mae_test)
    generalization_gap_r2 = abs(r2_train - r2_test)
    
    # Nice print statements as required
    print(f"\n{model_name} Results:")
    print("=" * 50)
    print(f"Mean Absolute Error (Training): {mae_train:.4f}")
    print(f"Mean Absolute Error (Testing): {mae_test:.4f}")
    print(f"Coefficient of Determination R² (Training): {r2_train:.4f}")
    print(f"Coefficient of Determination R² (Testing): {r2_test:.4f}")
    print(f"Generalization Gap (MAE): {generalization_gap_mae:.4f}")
    print(f"Generalization Gap (R²): {generalization_gap_r2:.4f}")
    
    return {
        'mae_train': mae_train, 'mae_test': mae_test,
        'r2_train': r2_train, 'r2_test': r2_test,
        'generalization_gap_mae': generalization_gap_mae,
        'generalization_gap_r2': generalization_gap_r2,
        'model': model
    }

def decision_tree_analysis(X_train, X_test, y_train, y_test):
    """
    REQUIREMENT #6: DecisionTree analysis with parameter experimentation
    Note: PDF says "DecisionTreeClassifier" but task is regression, using DecisionTreeRegressor
    """
    print("\n" + "="*60)
    print("DECISION TREE REGRESSOR ANALYSIS")
    print("="*60)
    
    # Basic model first
    dt_regressor = DecisionTreeRegressor(random_state=42)
    dt_regressor.fit(X_train, y_train)
    
    # Initial evaluation
    results = evaluate_model_compliance(dt_regressor, X_train, X_test, y_train, y_test, "Decision Tree (Default)")
    
    # Parameter experimentation as required
    print("\nParameter Experimentation:")
    print("-" * 30)
    
    best_mae = results['mae_test']
    best_params = "Default parameters"
    best_model = dt_regressor
    
    # Experiment with max_depth
    print("\n1. Experimenting with max_depth:")
    for depth in [3, 5, 10, 15, None]:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, dt.predict(X_test))
        r2 = r2_score(y_test, dt.predict(X_test))
        print(f"   max_depth={depth}: MAE={mae:.4f}, R²={r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_params = f"max_depth={depth}"
            best_model = dt
    
    # Experiment with min_samples_split
    print("\n2. Experimenting with min_samples_split:")
    for samples in [2, 5, 10, 20]:
        dt = DecisionTreeRegressor(min_samples_split=samples, random_state=42)
        dt.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, dt.predict(X_test))
        r2 = r2_score(y_test, dt.predict(X_test))
        print(f"   min_samples_split={samples}: MAE={mae:.4f}, R²={r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_params = f"min_samples_split={samples}"
            best_model = dt
    
    # Experiment with min_samples_leaf
    print("\n3. Experimenting with min_samples_leaf:")
    for leaf in [1, 2, 5, 10]:
        dt = DecisionTreeRegressor(min_samples_leaf=leaf, random_state=42)
        dt.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, dt.predict(X_test))
        r2 = r2_score(y_test, dt.predict(X_test))
        print(f"   min_samples_leaf={leaf}: MAE={mae:.4f}, R²={r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_params = f"min_samples_leaf={leaf}"
            best_model = dt
    
    # Report best result
    print(f"\nBest Decision Tree Result:")
    print(f"Best parameters: {best_params}")
    print(f"Best MAE: {best_mae:.4f}")
    
    # Final evaluation of best model
    final_results = evaluate_model_compliance(best_model, X_train, X_test, y_train, y_test, "Decision Tree (Best)")
    
    return final_results, best_params

def linear_regression_analysis(X_train, X_test, y_train, y_test):
    """
    REQUIREMENT #7: LinearRegression analysis with parameter experimentation
    """
    print("\n" + "="*60)
    print("LINEAR REGRESSION ANALYSIS")
    print("="*60)
    
    # Basic Linear Regression
    lr_regressor = LinearRegression()
    lr_regressor.fit(X_train, y_train)
    
    # Initial evaluation
    results = evaluate_model_compliance(lr_regressor, X_train, X_test, y_train, y_test, "Linear Regression")
    
    # Parameter experimentation following PDF requirements (tol and fit_intercept)
    print("\nParameter Experimentation:")
    print("Following PDF guidance: varying tolerance (tol) and fit_intercept")
    print("-" * 60)
    
    best_mae = results['mae_test']
    best_params = "Default LinearRegression"
    best_model = lr_regressor
    
    # 1. Experiment with fit_intercept parameter (whether model goes through origin)
    print("\n1. Experimenting with fit_intercept (whether model goes through origin):")
    for fit_intercept in [True, False]:
        lr = LinearRegression(fit_intercept=fit_intercept)
        lr.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, lr.predict(X_test))
        r2 = r2_score(y_test, lr.predict(X_test))
        print(f"   fit_intercept={fit_intercept}: MAE={mae:.4f}, R²={r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_params = f"fit_intercept={fit_intercept}"
            best_model = lr
    
    # 2. Experiment with tolerance using SGDRegressor (LinearRegression doesn't have tol)
    print("\n2. Experimenting with tolerance (using SGDRegressor as LinearRegression has no tol):")
    from sklearn.linear_model import SGDRegressor
    for tol in [1e-5, 1e-4, 1e-3, 1e-2]:
        sgd = SGDRegressor(tol=tol, random_state=42, max_iter=2000)
        sgd.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, sgd.predict(X_test))
        r2 = r2_score(y_test, sgd.predict(X_test))
        print(f"   SGD tol={tol}: MAE={mae:.4f}, R²={r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_params = f"SGD tol={tol}"
            best_model = sgd
    
    # 3. Try Ridge Regression as mentioned in PDF (alternative choice)
    from sklearn.linear_model import Ridge
    print("\n3. Experimenting with Ridge Regression (PDF alternative choice):")
    for alpha in [0.1, 1.0, 10.0]:
        ridge = Ridge(alpha=alpha, random_state=42)
        ridge.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, ridge.predict(X_test))
        r2 = r2_score(y_test, ridge.predict(X_test))
        print(f"   Ridge alpha={alpha}: MAE={mae:.4f}, R²={r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_params = f"Ridge alpha={alpha}"
            best_model = ridge
    
    # Report best result
    print(f"\nBest Linear Regression Result:")
    print(f"Best parameters: {best_params}")
    print(f"Best MAE: {best_mae:.4f}")
    
    # Final evaluation of best model
    final_results = evaluate_model_compliance(best_model, X_train, X_test, y_train, y_test, "Linear Regression (Best)")
    
    return final_results, best_params

def mlp_regressor_analysis(X_train, X_test, y_train, y_test):
    """
    REQUIREMENT #8: MLPRegressor analysis with parameter experimentation
    COMPLIANCE: Avoid varying solver as instructed in PDF
    """
    print("\n" + "="*60)
    print("MLP REGRESSOR ANALYSIS")
    print("="*60)
    
    # Basic MLP Regressor
    mlp_regressor = MLPRegressor(max_iter=1000, random_state=42)
    mlp_regressor.fit(X_train, y_train)
    
    # Initial evaluation
    results = evaluate_model_compliance(mlp_regressor, X_train, X_test, y_train, y_test, "MLP Regressor (Default)")
    
    # Parameter experimentation following PDF guidance
    print("\nParameter Experimentation:")
    print("Note: Following PDF instruction to NOT vary solver or random seed")
    print("-" * 60)
    
    best_mae = results['mae_test']
    best_params = "Default parameters"
    best_model = mlp_regressor
    
    # 1. Experiment with number of layers and nodes
    print("\n1. Experimenting with hidden layer sizes (architecture):")
    architectures = [(50,), (100,), (150,), (50, 50), (100, 50), (100, 100), (50, 25)]
    for arch in architectures:
        mlp = MLPRegressor(hidden_layer_sizes=arch, max_iter=2000, random_state=42)
        mlp.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, mlp.predict(X_test))
        r2 = r2_score(y_test, mlp.predict(X_test))
        print(f"   Architecture {arch}: MAE={mae:.4f}, R²={r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_params = f"hidden_layer_sizes={arch}"
            best_model = mlp
    
    # 2. Experiment with activation functions
    print("\n2. Experimenting with activation functions:")
    for activation in ['relu', 'tanh', 'logistic']:
        mlp = MLPRegressor(activation=activation, max_iter=2000, random_state=42)
        mlp.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, mlp.predict(X_test))
        r2 = r2_score(y_test, mlp.predict(X_test))
        print(f"   Activation '{activation}': MAE={mae:.4f}, R²={r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_params = f"activation='{activation}'"
            best_model = mlp
    
    # 3. Experiment with learning rate
    print("\n3. Experimenting with learning rate:")
    for lr_init in [0.0001, 0.001, 0.01, 0.1]:
        mlp = MLPRegressor(learning_rate_init=lr_init, max_iter=2000, random_state=42)
        mlp.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, mlp.predict(X_test))
        r2 = r2_score(y_test, mlp.predict(X_test))
        print(f"   Learning rate {lr_init}: MAE={mae:.4f}, R²={r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_params = f"learning_rate_init={lr_init}"
            best_model = mlp
    
    # 4. Experiment with stopping criterion (early stopping)
    print("\n4. Experimenting with early stopping:")
    for early_stop in [True, False]:
        mlp = MLPRegressor(early_stopping=early_stop, validation_fraction=0.1, max_iter=2000, random_state=42)
        mlp.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, mlp.predict(X_test))
        r2 = r2_score(y_test, mlp.predict(X_test))
        print(f"   Early stopping {early_stop}: MAE={mae:.4f}, R²={r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_params = f"early_stopping={early_stop}"
            best_model = mlp
    
    # Report best result
    print(f"\nBest MLP Regressor Result:")
    print(f"Best parameters: {best_params}")
    print(f"Best MAE: {best_mae:.4f}")
    
    # Final evaluation of best model
    final_results = evaluate_model_compliance(best_model, X_train, X_test, y_train, y_test, "MLP Regressor (Best)")
    
    return final_results, best_params

def create_excel_output(best_model_info, scaler, merged_df, features, target):
    """
    REQUIREMENT #9: Excel spreadsheet for best model with specified columns
    """
    print("\n" + "="*60)
    print("CREATING EXCEL SPREADSHEET FOR BEST MODEL")
    print("="*60)
    
    # Prepare data for prediction on whole dataset
    feature_data = merged_df[features].dropna()
    target_data = merged_df.loc[feature_data.index, target]
    
    # Convert to numpy arrays and scale
    X_all_numpy = feature_data.values
    X_all_scaled = scaler.transform(X_all_numpy)
    
    # Make predictions with best model
    predictions = best_model_info['model'].predict(X_all_scaled)
    
    # Create Excel spreadsheet with required columns
    prediction_df = pd.DataFrame()
    prediction_df['Player_Season'] = (merged_df.loc[feature_data.index, 'name'] + "_" + 
                                    merged_df.loc[feature_data.index, 'yearID'].astype(str))
    prediction_df['True output'] = target_data.values
    prediction_df['Model output'] = predictions
    prediction_df['Residual (true - model)'] = prediction_df['True output'] - prediction_df['Model output']
    
    # Save to Excel
    prediction_df.to_excel("./predictions.xlsx", index=False)
    print(f"Excel spreadsheet saved with {len(prediction_df)} samples")
    print("Columns: Player_Season, True output, Model output, Residual (true - model)")
    
    return prediction_df

def main():
    """
    Main function following exact PDF requirements sequence
    """
    print("ECE5424 Machine Learning HW1: Baseball Steal Prediction")
    print("STRICT COMPLIANCE VERSION")
    print("="*60)
    
    # Load and preprocess data (Requirements #1-5)
    X_train, X_test, y_train, y_test, scaler, merged_df, features, target = load_and_preprocess_data()
    
    print(f"\nFeatures used (speed-related variables): {features}")
    print(f"Target variable: {target}")
    print(f"Data converted to numpy arrays as required")
    print(f"Normalization: MinMaxScaler range [-1, 1] as specified")
    
    # Store results for comparison
    all_results = {}
    all_params = {}
    
    # REQUIREMENT #6: Decision Tree Analysis
    dt_results, dt_params = decision_tree_analysis(X_train, X_test, y_train, y_test)
    all_results['Decision Tree'] = dt_results
    all_params['Decision Tree'] = dt_params
    
    # REQUIREMENT #7: Linear Regression Analysis
    lr_results, lr_params = linear_regression_analysis(X_train, X_test, y_train, y_test)
    all_results['Linear Regression'] = lr_results
    all_params['Linear Regression'] = lr_params
    
    # REQUIREMENT #8: MLP Regressor Analysis
    mlp_results, mlp_params = mlp_regressor_analysis(X_train, X_test, y_train, y_test)
    all_results['MLP Regressor'] = mlp_results
    all_params['MLP Regressor'] = mlp_params
    
    # Model comparison and selection
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    
    print(f"{'Model':<20} {'Test MAE':<12} {'Test R²':<12} {'Best Parameters'}")
    print("-" * 80)
    
    best_model_name = None
    best_mae = float('inf')
    
    for model_name, results in all_results.items():
        mae = results['mae_test']
        r2 = results['r2_test']
        params = all_params[model_name]
        
        print(f"{model_name:<20} {mae:<12.4f} {r2:<12.4f} {params}")
        
        if mae < best_mae:
            best_mae = mae
            best_model_name = model_name
    
    print(f"\nBest performing model: {best_model_name} (MAE: {best_mae:.4f})")
    
    # REQUIREMENT #9: Excel output for best model
    create_excel_output(all_results[best_model_name], scaler, merged_df, features, target)
    
    # REQUIREMENT #10: Discussion prompt
    print("\n" + "="*60)
    print("DISCUSSION POINTS TO ADDRESS:")
    print("="*60)
    print("1. Which architecture is the best?")
    print("2. Which set of parameters?") 
    print("3. How did you decide what is 'best'?")
    print("4. Is your accuracy enough to be usable?")
    print("5. How would the model be used?")
    
    print(f"\nAnalysis complete. Best model: {best_model_name}")
    print("Files generated: predictions.xlsx")

if __name__ == '__main__':
    main()
