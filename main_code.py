import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from scipy.optimize import nnls
import joblib
import math
import os

# custom modules
from computing_statistics import calculate_stats
from generating_plot import analyze_data


# load data
def load_dataset(file_path):
    """Load the dataset from the given file path."""
    dataset = pd.read_csv(file_path, low_memory=False, comment='#', sep='\t')
    return dataset

apply_function = {
#    "lum": "log",
#    "Radius": "log",
#    "Teff": "log",
#    "Xc": "log",
#    "Xs": "log",
#    "Zs": "log",
#    "Age_adim": "del"
}

def preprocess_transformations(dataset, apply_function):
    """
    Apply transformations to columns in the dataset based on the apply_function dictionary.

    Parameters:
        dataset (pd.DataFrame): The DataFrame to apply transformations to.
        apply_function (dict): A dictionary mapping column names to transformation names.

    Returns:
        pd.DataFrame: The modified DataFrame after applying transformations.
    """
    for column_name, transform_name in apply_function.items():
        if column_name in dataset.columns:
            if transform_name == "del":
                # Drop the column from the DataFrame
                dataset.drop(columns=column_name, inplace=True)
            elif transform_name == "log":
                # Apply log transformation
                dataset[column_name] = np.log10(dataset[column_name])
            # Add more transformation options if needed
    return dataset

def preprocess_filters(dataset):
    """
    Preprocess the dataset by applying various filtering.

    Parameters:
        dataset (pd.DataFrame): The DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    #dataset['Age'] = dataset['Age'] / 1000  # convert age to Gyr

    # Apply conditional filtering
    dataset = dataset[(dataset['age'] >= 0) & (dataset['age'] <= 14)]
    
    # Apply other filters if needed

    return dataset

# Load your dataset into a DataFrame
file_path = '/home/kamju/projects/ml_project/ml_grid/grid_DYDZ/uniform_grid_file.txt'
print("Loading dataset...")
dataset = load_dataset(file_path)

print("Dataset loaded successfully!")
# Apply preprocessing transformations
print("Applying preprocessing transformations...")
dataset = preprocess_transformations(dataset, apply_function)

# Apply preprocessing filters
print("Applying preprocessing filters...")
data = preprocess_filters(dataset)

print(data['L'])

# Split the data into features and target variables
X = data[['Teff', '[Fe/H]', 'L']]
y = data[['mass', 'radius', 'age']]

# Split the data into training and test sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data using RobustScaler
print("Normalizing data...")
scaler = RobustScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')
print("Scaler saved.")

# Define base models
base_models = [
#    ("KNN", KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
#    ("MLP", MLPRegressor(hidden_layer_sizes=(100,),activation='relu', solver='sgd',verbose=False,shuffle=True,batch_size='auto',max_iter=1000, random_state=42,tol=1e-4)),
#    ("XGBoost", XGBRegressor( n_estimators=1000,random_state=42, n_jobs=-1)),
#    ("RF", RandomForestRegressor( random_state=42, n_jobs=-1)),
#    ("CatBoost", CatBoostRegressor(iterations=1000,random_state=42, silent=True)),
    ("XT", ExtraTreesRegressor(random_state=42, n_jobs=-1)),
]

# Cross-validation setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize placeholders for results
meta_model_coefficients_dict = {target: [] for target in y.columns}

# Function to train and evaluate base models for a specific target variable
def train_base_models(X_train_norm, y_train, y_test, target_name):
    print(f"\nTraining models for target: {target_name}")
    rmse_cv_results = {name: [] for name, _ in base_models}
    rmse_test_results = {name: [] for name, _ in base_models}
    stacked_rmse_cv_results = []
    stacked_rmse_test_results = []
    coefficients_results = []

    for train_index, val_index in kf.split(X_train_norm):
        meta_features = np.zeros((X_train_norm.shape[0], len(base_models)))
        print("Cross-validation split...")
        for i, (name, model) in enumerate(base_models):
            print(f"Training {name} model...")
            model.fit(X_train_norm[train_index], y_train.iloc[train_index])
            y_pred = model.predict(X_train_norm[val_index])
            meta_features[val_index, i] = y_pred
            rmse_cv = np.sqrt(mean_squared_error(y_train.iloc[val_index], y_pred))
            rmse_cv_results[name].append(rmse_cv)
            print(f"RMSE for {name} on validation set: {rmse_cv:.4f}")
        meta_model_coefficients, _ = nnls(meta_features, y_train)
        meta_model_coefficients /= np.sum(meta_model_coefficients)
        coefficients_results.append(meta_model_coefficients)
        final_predictions_cv = meta_features @ meta_model_coefficients
        stacked_rmse_cv = np.sqrt(mean_squared_error(y_train, final_predictions_cv))
        stacked_rmse_cv_results.append(stacked_rmse_cv)
        print(f"Stacked RMSE on validation set: {stacked_rmse_cv:.4f}")

    meta_features_test = np.zeros((X_test_normalized.shape[0], len(base_models)))
    for i, (name, model) in enumerate(base_models):
        print(f"Retraining {name} model on entire training set...")
        model.fit(X_train_norm, y_train)
        joblib.dump(model, f'{name}_model_{target_name}.joblib')
        y_pred_test = model.predict(X_test_normalized)
        meta_features_test[:, i] = y_pred_test
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        rmse_test_results[name].append(rmse_test)
        print(f"RMSE for {name} on test set: {rmse_test:.4f}")

    final_predictions_test = meta_features_test @ meta_model_coefficients
    stacked_rmse_test = np.sqrt(mean_squared_error(y_test, final_predictions_test))
    stacked_rmse_test_results.append(stacked_rmse_test)
    print(f"Stacked RMSE on test set: {stacked_rmse_test:.4f}")

    return meta_model_coefficients, coefficients_results

# Train and save models for each target variable
for target in y.columns:
    meta_model_coefficients_dict[target], _ = train_base_models(X_train_normalized, y_train[target], y_test[target], target)

# Save meta-model coefficients
for target in y.columns:
    np.save(f'meta_model_coefficients_{target}.npy', meta_model_coefficients_dict[target])
    print(f"Meta-model coefficients for {target} saved.")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% loading and predicting for real observations %%%%%%%%%%%%%%%%%%%%%%%%

#file_path = '/home/kamju/projects/ml_project/sun_results/sun_constraints.tsv'
#file_path = '/home/kamju/projects/hot_jupiters_project/data_files/observables_HARPS_data.csv'
file_path  = '/home/kamju/projects/jupiters_project/data_files/HARPS_ml_constraints.csv'
 #binaries_TFL_observables.tsv

#BINARIES_w_revised_lum_observable_3D, non_BINARIES_lum_observable_3D

#Load the data into a pandas DataFrame
df = pd.read_csv(file_path, delimiter='\t', skiprows=0)
print(df.head())
# Extract object names
obj_names = df.iloc[:, 0].values

# Extract actual values (assuming actual values are in odd columns starting from the second column)
actual_values = df.iloc[:, 1::2].values.astype(float)

# Extract standard errors (assuming standard errors are in even columns starting from the third column)
std_errors = df.iloc[:, 2::2].values.astype(float)
# Create a boolean mask identifying rows with any NaN values
nan_mask = df.isnull().any(axis=1)

# Use the mask to filter the DataFrame and list the rows with NaN values
rows_with_nan = df[nan_mask]
#print("Rows with NaN values:")
#print(rows_with_nan)
#print(actual_values)
#print(real_data)
#print(obj_names)
#print(actual_values)
#print(std_errors)
# Number of noisy samples per data point

num_objects = len(obj_names)
num_samples = 10000
#print(f"Generating {num_samples} noisy samples per data point...")

np.random.seed(42)
noisy_data = np.empty((num_samples, actual_values.shape[1], num_objects), dtype=float)  # an empty array to store the data

# Generate 10,000 random values from each observed value and error pair 
for obj_index in range(num_objects):
    for i, (x, y) in enumerate(zip(actual_values[obj_index], std_errors[obj_index])):
       # lower_limit = x - y  # + because the lower uncertainties are negatives
       # upper_limit = x + y
        values = x + y*np.random.normal(0, 1, 10000)
        noisy_data[:, i, obj_index] = values  # Assign generated values to columns
#print(noisy_data,"Noisy data generation complete.")


features = ['Teff', '[Fe/H]', 'L']

noisy_data_dfs = []

for obj_index in range(num_objects):
    obj_name = obj_names[obj_index]
    data_obj = noisy_data[:, :, obj_index]
    
    # Convert the synthetic data to a DataFrame
    data_df = pd.DataFrame(data_obj, columns=features)
    #print(data_df)
        
    # Save predictions
    #file_path = f'/home/kamju/projects/ml_project/sun_results/{obj_name}_saved_preds.txt'
   # joblib.dump(preds_for_obj, file_path)
    noisy_data_dfs.append(data_df)
    

#print(noisy_data_dfs,"Loading scaler...")
scaler = joblib.load('scaler.joblib')

# Define base models
base_models = [
#    ("KNN", KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
#    ("MLP", MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=5000, random_state=42)),
#    ("XGBoost", XGBRegressor(n_estimators=700, random_state=42, n_jobs=-1)),
#    ("RF", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
#    ("CatBoost", CatBoostRegressor(n_estimators=4000, random_state=42, silent=True)),
    ("XT", ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
]

# Load base models and meta-model coefficients for real data predictions
#print("Loading base models and meta-model coefficients...")
base_models_dict = {
    target: [joblib.load(f'{name}_model_{target}.joblib') for name, _ in base_models]
    for target in ['mass', 'radius', 'age']
}
meta_model_coefficients_dict = {
    target: np.load(f'meta_model_coefficients_{target}.npy')
    for target in ['mass', 'radius', 'age']
}


# Function to generate predictions for noisy data
def predict_with_uncertainty(base_models, meta_model_coefficients, X_real_norm):
    meta_features_real = np.zeros((X_real_norm.shape[0], len(base_models)))
    for i, model in enumerate(base_models):
        meta_features_real[:, i] = model.predict(X_real_norm)
    final_predictions_real = meta_features_real @ meta_model_coefficients
    return final_predictions_real

# Function to generate predictions for noisy data
def predict_noisy_data(base_models, meta_model_coefficients, noisy_data_dfs, scaler, obj_names):
    for obj_name, data_df in zip(obj_names, noisy_data_dfs):
        X_noisy = data_df[['Teff', '[Fe/H]', 'L']]
        X_noisy_normalized = scaler.transform(X_noisy)
        
        predictions_dict = {}
        for target in ['mass', 'radius', 'age']:
            noisy_predictions = predict_with_uncertainty(base_models[target], meta_model_coefficients[target], X_noisy_normalized)
            predictions_dict[target] = noisy_predictions
        
        # Convert predictions_dict to DataFrame
        predictions_df = pd.DataFrame({
            'mass': predictions_dict['mass'],
            'radius': predictions_dict['radius'],
            'age': predictions_dict['age']
        })
        
        # Save predictions to a single file for each object
        file_path = f'/home/kamju/projects/jupiters_project/SWcat/test_results/{obj_name}_saved_preds.txt'
        joblib.dump(predictions_df, file_path)
        

predict_noisy_data(base_models_dict, meta_model_coefficients_dict, noisy_data_dfs, scaler, obj_names)
        
print("predicting noisy data")
def calculate_stats(predictions):
    median = np.median(predictions)
    q1 = np.percentile(predictions, 16)
    q3 = np.percentile(predictions, 84)
    upper_bound = q3 - median
    lower_bound = -1 * (median - q1)
    return median, upper_bound, lower_bound

# Create instances to store results
object_names_list = []
median_mass_list = []
ub_mass_list = []
lb_mass_list = []
median_radius_list = []
ub_radius_list = []
lb_radius_list = []
median_age_list = []
ub_age_list = []
lb_age_list = []
# Iterate over each saved file
for obj_index in range(num_objects):
    obj_name 	= obj_names[obj_index]
    file_path 	= '/home/kamju/projects/jupiters_project/SWcat/test_results/{}_saved_preds.txt'.format(obj_name)
    
    # Load predictions from the saved file
    preds_for_obj 	= joblib.load(file_path)
    print(preds_for_obj)
    # Calculate median and upper and lower bounds
    median_mass, ub_mass, lb_mass = calculate_stats(preds_for_obj['mass']) #  
    median_radius, ub_radius,lb_radius = calculate_stats(preds_for_obj['radius'])  # [:, 1]
    median_age, ub_age, lb_age = calculate_stats(preds_for_obj['age'])   # [:, 2]

    # Append the results to the lists
    object_names_list.append(obj_name)
    median_age_list.append('{:.3f}'.format(median_age))
    ub_age_list.append('{:.3f}'.format(ub_age))
    lb_age_list.append('{:.3f}'.format(lb_age))

    median_radius_list.append('{:.3f}'.format(median_radius))
    ub_radius_list.append('{:.3f}'.format(ub_radius))
    lb_radius_list.append('{:.3f}'.format(lb_radius))

    median_mass_list.append('{:.3f}'.format(median_mass))
    ub_mass_list.append('{:.3f}'.format(ub_mass))
    lb_mass_list.append('{:.3f}'.format(lb_mass))


# Create a DataFrame from the lists
result_df = pd.DataFrame({
    'object_name': object_names_list,
    'mass': median_mass_list,
    'ub_mass': ub_mass_list,
    'lb_mass': lb_mass_list,
    

    'radius': median_radius_list,
    'ub_radius': ub_radius_list,
    'lb_radius': lb_radius_list,

    'age': median_age_list,
    'ub_age': ub_age_list,
    'lb_age': lb_age_list
   
     })
# Save the results to a new file
result_file_path = '/home/kamju/projects/jupiters_project/SWcat/test_results/SWcat_results.txt'
result_df.to_csv(result_file_path, index=False, sep='\t')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% visualising distributions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def create_results_directory():
    results_dir = '/home/kamju/projects/jupiters_project/SWcat/test_results/histograms/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

if __name__ == '__main__': # do not delete this line
    directory_path = '/home/kamju/projects/jupiters_project/SWcat/test_results/'
    results_directory = create_results_directory()
    analyze_data(directory_path, results_directory)
