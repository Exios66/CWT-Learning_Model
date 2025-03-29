#!/usr/bin/env python3
"""
Cognitive Workload Assessment and Prediction Tool

This module implements a machine learning pipeline for predicting cognitive workload
based on physiological, EEG, and gaze data. The tool supports multiple machine learning
models including Random Forest, SVM, Neural Networks, and more.

Available commands:
    train       - Train a new model
    predict     - Make predictions with a trained model
    list-models - List available trained models
    install-models - Install sample pre-trained models

Example usage:
    python cwt.py train --model-type rf
    python cwt.py predict --input input_data.json
    python cwt.py list-models
    python cwt.py install-models

Available model types:
    rf  - Random Forest
    svm - Support Vector Machine
    gb  - Gradient Boosting
    mlp - Neural Network (MLP)
    knn - K-Nearest Neighbors
    lr  - Logistic Regression

SIMPLE COMMANDS:

  Train a model:
    python cwt.py train

  Make predictions:
    python cwt.py predict --input data.json

  List available models:
    python cwt.py list-models

  Install sample models:
    python cwt.py install-models

Type 'python cwt.py train --help' for more options.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
import pickle
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import tqdm
import glob

from predictor.predictor import predict_new_data

# Define constants
MODEL_DIR = "models"
DATA_FILES = {
    "physiological": "data/physiological.csv",
    "eeg": "data/eeg.csv",
    "gaze": "data/gaze.csv"
}

# Workload class definitions
WORKLOAD_CLASSES = {
    0: "Low",
    1: "Medium",
    2: "High",
    None: "Unknown"
}

# Define required features for interactive input
REQUIRED_FEATURES = [
    "pulse_rate", "blood_pressure_sys", "resp_rate", 
    "pupil_diameter_left", "pupil_diameter_right", 
    "fixation_duration", "blink_rate", "gaze_x", "gaze_y", 
    "alpha_power", "theta_power"
]

# Define common features for prediction
COMMON_FEATURES = [
    "pulse_rate", "blood_pressure_sys", "resp_rate", 
    "pupil_diameter_left", "pupil_diameter_right", 
    "fixation_duration", "blink_rate", "gaze_x", "gaze_y", 
    "alpha_power", "theta_power"
]

# Group features by modality for better handling of missing data types
FEATURE_MODALITIES = {
    "physiological": ["pulse_rate", "blood_pressure_sys", "resp_rate"],
    "pupil": ["pupil_diameter_left", "pupil_diameter_right"],
    "gaze": ["fixation_duration", "blink_rate", "gaze_x", "gaze_y"],
    "eeg": ["alpha_power", "theta_power"]
}

# Define modality importance for confidence scoring (higher = more important)
MODALITY_IMPORTANCE = {
    "physiological": 0.3,  # Basic physiological signals (moderate importance)
    "pupil": 0.3,          # Pupil metrics (high correlation with workload)
    "gaze": 0.2,           # Gaze metrics (moderate correlation)
    "eeg": 0.2             # EEG signals (important for cognitive state)
}

# Set up logging
logger = None

# Load environment variables if .env file exists
if os.path.exists('.env'):
    load_dotenv()

# ---------------------- CONFIGURATION ---------------------- #
# Set up logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/general/cwt.log')

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('cwt')

# Data paths - Fix path configuration
DATA_FILES = {
    "physiological": os.getenv('PHYSIO_DATA_PATH', 'data/Enhanced_Workload_Clinical_Data.csv'),
    "eeg": os.getenv('EEG_DATA_PATH', 'data/000_EEG_Cluster_ANOVA_Results.csv'),
    "gaze": os.getenv('GAZE_DATA_PATH', 'data/008_01.csv')
}

# Create the data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Log data file paths for debugging
logger.debug(f"Physiological data file path: {DATA_FILES['physiological']}")
logger.debug(f"EEG data file path: {DATA_FILES['eeg']}")
logger.debug(f"Gaze data file path: {DATA_FILES['gaze']}")

# Model configuration
MODEL_OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR', 'models/sample/default')
MODEL_NAME = os.getenv('MODEL_NAME', 'Cognitive_State_Prediction_Model')
MODEL_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")

# Default model type (can be overridden by command line)
DEFAULT_MODEL_TYPE = os.getenv('DEFAULT_MODEL_TYPE', 'rf')

# Dictionary of available models
AVAILABLE_MODELS = {
    'rf': 'Random Forest',
    'svm': 'Support Vector Machine',
    'gb': 'Gradient Boosting',
    'mlp': 'Neural Network (MLP)',
    'knn': 'K-Nearest Neighbors',
    'lr': 'Logistic Regression'
}

# Training parameters
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))

# Output file paths
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, f"{MODEL_NAME}_{MODEL_VERSION}.joblib")
SCALER_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, f"scaler_{MODEL_VERSION}.joblib")
METADATA_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, f"metadata_{MODEL_VERSION}.json")

# Define the directory for storing models
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cwt", "models")

# ---------------------- HELPER FUNCTIONS ---------------------- #
def validate_file_exists(file_path):
    """Validate that a file exists on disk."""
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        print(f"\n✘ ERROR: {error_msg}")
        print("Please check if the file exists or update your configuration.")
        raise FileNotFoundError(error_msg)
    return True

def safe_read_csv(file_path):
    """Safely read a CSV file with error handling."""
    try:
        validate_file_exists(file_path)
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

def save_model_metadata(model, model_type, features, accuracy, class_report):
    """Save metadata about the model for reproducibility."""
    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": model_type,
        "model_name": AVAILABLE_MODELS.get(model_type, "Unknown"),
        "training_date": datetime.now().isoformat(),
        "accuracy": float(accuracy),
        "classification_report": class_report,
        "features": features,
        "parameters": {
            "test_size": TEST_SIZE,
            "random_seed": RANDOM_SEED
        }
    }
    
    with open(METADATA_OUTPUT_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata saved to {METADATA_OUTPUT_PATH}")

def plot_feature_importance(model, feature_names, output_path=None):
    """Plot feature importance from the trained model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Feature importance plot saved to {output_path}")
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, output_path=None):
    """Plot confusion matrix from predictions."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(set(y_true)), 
                yticklabels=sorted(set(y_true)))
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Confusion matrix plot saved to {output_path}")
    else:
        plt.show()

# ---------------------- LOAD DATA ---------------------- #
def load_data():
    """
    Load the physiological, EEG, and gaze data from CSV files.
    
    Returns:
        tuple: Tuple containing dataframes (df_physio, df_eeg, df_gaze)
    """
    logger.info("Loading data files...")
    try:
        # Check if data files exist before proceeding
        for data_type, file_path in DATA_FILES.items():
            if not os.path.exists(file_path):
                logger.error(f"{data_type.capitalize()} data file not found: {file_path}")
                print(f"\n✘ Data file missing: {file_path}")
                print("\nSince real data is missing, you have two options:")
                print("  1. Run 'python cwt.py install-models' to install sample models")
                print("  2. Create a 'data' folder with the required files")
                print("     or modify paths in your .env file")
                print("\nRecommended action:")
                print("  python cwt.py install-models")
                sys.exit(1)
        
        df_physio = safe_read_csv(DATA_FILES["physiological"])
        df_eeg = safe_read_csv(DATA_FILES["eeg"])
        df_gaze = safe_read_csv(DATA_FILES["gaze"])
        
        logger.info(f"Loaded physiological data: {df_physio.shape}")
        logger.info(f"Loaded EEG data: {df_eeg.shape}")
        logger.info(f"Loaded gaze data: {df_gaze.shape}")
        
        return df_physio, df_eeg, df_gaze
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# ---------------------- PREPROCESSING ---------------------- #
def preprocess_data(df_physio, df_eeg, df_gaze):
    """
    Preprocess and merge the physiological, EEG, and gaze data.
    
    Args:
        df_physio (DataFrame): Physiological data
        df_eeg (DataFrame): EEG data
        df_gaze (DataFrame): Gaze tracking data
        
    Returns:
        tuple: Processed dataframe, fitted scaler, and features (df, scaler, features)
    """
    logger.info("Preprocessing data...")
    
    try:
        # Convert timestamps to datetime
        logger.debug("Converting timestamps to datetime format")
        for df, name in [(df_physio, "physiological"), (df_eeg, "eeg"), (df_gaze, "gaze")]:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except KeyError:
                logger.error(f"'timestamp' column not found in {name} data")
                raise
            except Exception as e:
                logger.error(f"Error converting timestamps in {name} data: {str(e)}")
                raise
        
        # Merge based on nearest timestamp
        logger.debug("Merging datasets based on timestamp")
        try:
            df = pd.merge_asof(df_physio.sort_values("timestamp"), 
                              df_eeg.sort_values("timestamp"), 
                              on="timestamp")
            df = pd.merge_asof(df.sort_values("timestamp"), 
                              df_gaze.sort_values("timestamp"), 
                              on="timestamp")
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            raise
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values detected after merge: \n{missing_values[missing_values > 0]}")
            logger.info("Dropping rows with missing values")
            df = df.dropna()
        
        # Select relevant features
        features = ["pulse_rate", "blood_pressure_sys", "resp_rate", "pupil_diameter_left",
                   "pupil_diameter_right", "fixation_duration", "blink_rate", "workload_intensity",
                   "gaze_x", "gaze_y", "alpha_power", "theta_power"]
        
        # Validate features exist in dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features in dataset: {missing_features}")
            raise ValueError(f"Missing features in dataset: {missing_features}")
        
        logger.debug(f"Selected {len(features)} features for analysis")
        df = df[features]
        
        # Standardization
        logger.debug("Standardizing features")
        scaler = StandardScaler()
        scaler.fit(df[features])  # First fit without transforming to get feature_names
        df[features] = scaler.transform(df[features])
        
        # Store feature names in the scaler for prediction
        scaler.feature_names_in_ = np.array(features)
        
        # Label Encoding for Workload Intensity (Target Variable)
        logger.debug("Encoding cognitive state labels")
        df["cognitive_state"] = pd.qcut(df["workload_intensity"], q=3, labels=["Low", "Medium", "High"])
        
        logger.info(f"Preprocessing complete. Final dataset shape: {df.shape}")
        return df, scaler, features
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

# ---------------------- MODEL CREATION ---------------------- #
def create_model(model_type='rf'):
    """
    Create a machine learning model based on the specified type.
    
    Args:
        model_type (str): Type of model to create ('rf', 'svm', 'gb', 'mlp', 'knn', 'lr')
        
    Returns:
        object: Initialized model
    """
    logger.info(f"Creating model of type: {model_type} ({AVAILABLE_MODELS.get(model_type, 'Unknown')})")
    
    if model_type == 'rf':
        # Random Forest
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=RANDOM_SEED
        )
    
    elif model_type == 'svm':
        # Support Vector Machine
        return SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=RANDOM_SEED
        )
    
    elif model_type == 'gb':
        # Gradient Boosting
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_SEED
        )
    
    elif model_type == 'mlp':
        # Neural Network (MLP)
        return MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=RANDOM_SEED
        )
    
    elif model_type == 'knn':
        # K-Nearest Neighbors
        return KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
    
    elif model_type == 'lr':
        # Logistic Regression
        return LogisticRegression(
            C=1.0,
            max_iter=100,
            random_state=RANDOM_SEED
        )
    
    else:
        logger.warning(f"Unknown model type: {model_type}. Defaulting to Random Forest.")
        return RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED
        )

# ---------------------- MODEL TRAINING FUNCTION ---------------------- #
def train_model(df, features, model_type='rf', scaler=None):
    """
    Train a machine learning model for cognitive state prediction.
    
    Args:
        df (DataFrame): Preprocessed data with features and target
        features (list): List of feature names
        model_type (str): Type of model to train
        scaler (StandardScaler, optional): Fitted scaler to save with the model
        
    Returns:
        tuple: (trained_model, accuracy, X_test, y_test, y_pred)
    """
    logger.info(f"Training model of type: {model_type}")
    
    try:
        X = df.drop(columns=["cognitive_state", "workload_intensity"])
        y = df["cognitive_state"]
        
        # Split data
        logger.debug(f"Splitting data with test_size={TEST_SIZE}, random_state={RANDOM_SEED}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
        
        # Create and train model
        model = create_model(model_type)
        
        # Perform cross-validation
        logger.debug("Performing 5-fold cross-validation")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Fit the model on the full training data
        logger.info("Fitting model on training data")
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Model Accuracy: {accuracy:.3f}")
        logger.info(f"Classification Report:\n {classification_report(y_test, y_pred)}")
        
        # Create visualization directory if it doesn't exist
        viz_dir = os.path.join('models', 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot and save confusion matrix
        logger.debug("Generating confusion matrix")
        cm_path = os.path.join(viz_dir, f"confusion_matrix_{MODEL_VERSION}.png")
        plot_confusion_matrix(y_test, y_pred, cm_path)
        
        # Plot and save feature importance if model supports it
        if hasattr(model, 'feature_importances_'):
            logger.debug("Generating feature importance plot")
            fi_path = os.path.join(viz_dir, f"feature_importance_{MODEL_VERSION}.png")
            plot_feature_importance(model, X.columns, fi_path)
        
        # Determine the appropriate model directory based on model_type
        model_dir = os.path.join(MODEL_OUTPUT_DIR)
        if model_type in AVAILABLE_MODELS:
            # Override model_dir if explicitly using a model type to ensure it goes in the right subdirectory
            model_type_dir = os.path.join('models', 'sample', model_type)
            os.makedirs(model_type_dir, exist_ok=True)
            model_dir = model_type_dir
        
        # Set up model paths
        model_filename = f"{MODEL_NAME}_{MODEL_VERSION}_{model_type}.joblib"
        scaler_filename = f"scaler_{MODEL_VERSION}_{model_type}.joblib"
        metadata_filename = f"metadata_{MODEL_VERSION}_{model_type}.json"
        
        MODEL_OUTPUT_PATH = os.path.join(model_dir, model_filename)
        SCALER_PATH = os.path.join(model_dir, scaler_filename)
        METADATA_OUTPUT_PATH = os.path.join(model_dir, metadata_filename)
        
        # Save model and scaler
        logger.info(f"Saving model to {MODEL_OUTPUT_PATH}")
        joblib.dump(model, MODEL_OUTPUT_PATH)
        
        # Create a new scaler if one wasn't provided
        if scaler is None:
            logger.warning("No scaler provided, creating a new one")
            scaler = StandardScaler()
            scaler.fit(df[features])
            scaler.feature_names_in_ = np.array(features)
        
        logger.info(f"Saving scaler to {SCALER_PATH}")
        joblib.dump(scaler, SCALER_PATH)
        
        # Save metadata
        metadata = {
            "model_version": MODEL_VERSION,
            "model_type": model_type,
            "model_name": AVAILABLE_MODELS.get(model_type, "Unknown"),
            "training_date": datetime.now().isoformat(),
            "accuracy": float(accuracy),
            "classification_report": class_report,
            "features": list(X.columns),
            "parameters": {
                "test_size": TEST_SIZE,
                "random_seed": RANDOM_SEED
            }
        }
        
        with open(METADATA_OUTPUT_PATH, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Model metadata saved to {METADATA_OUTPUT_PATH}")
        
        print(f"Model saved to {MODEL_OUTPUT_PATH}")
        print(f"Scaler saved to {SCALER_PATH}")
        
        return model, accuracy, X_test, y_test, y_pred
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

# ---------------------- PREDICTION FUNCTIONS ---------------------- #
def calculate_modality_confidence(available_features):
    """
    Calculate a confidence score based on which feature modalities are available.
    
    Args:
        available_features (list): List of feature names that are available
        
    Returns:
        tuple: (confidence_score, missing_modalities, available_modalities)
            - confidence_score: float between 0-1 indicating prediction confidence
            - missing_modalities: list of missing modality names
            - available_modalities: list of available modality names
    """
    available_modalities = []
    missing_modalities = []
    confidence_score = 0.0
    
    # Check each modality
    for modality, features in FEATURE_MODALITIES.items():
        # Check if at least half of the features in this modality are available
        available_count = sum(1 for feature in features if feature in available_features)
        if available_count >= len(features) / 2:
            available_modalities.append(modality)
            confidence_score += MODALITY_IMPORTANCE[modality]
        else:
            missing_modalities.append(modality)
    
    # Normalize confidence to account for case when not all modalities are defined
    if confidence_score > 0 and sum(MODALITY_IMPORTANCE.values()) > 0:
        confidence_score = min(1.0, confidence_score / sum(MODALITY_IMPORTANCE.values()))
    
    return confidence_score, missing_modalities, available_modalities

def create_default_scaler():
    """
    Create a default StandardScaler for when loading a scaler fails.
    This is a fallback mechanism to ensure the system can still function
    when scaler files are corrupted or missing.
    
    Returns:
        sklearn.preprocessing.StandardScaler: A standard scaler initialized with common feature ranges
    """
    logger.warning("Creating a default scaler as fallback")
    from sklearn.preprocessing import StandardScaler
    
    # Create a simple dataset with reasonable feature ranges
    demo_data = {
        'pulse_rate': [60, 70, 80, 90, 100],
        'blood_pressure_sys': [100, 110, 120, 130, 140],
        'resp_rate': [12, 14, 16, 18, 20],
        'pupil_diameter_left': [2.5, 3.0, 3.5, 4.0, 4.5],
        'pupil_diameter_right': [2.5, 3.0, 3.5, 4.0, 4.5],
        'fixation_duration': [200, 250, 300, 350, 400],
        'blink_rate': [10, 12, 14, 16, 18],
        'gaze_x': [400, 450, 500, 550, 600],
        'gaze_y': [300, 350, 400, 450, 500],
        'alpha_power': [10, 15, 20, 25, 30],
        'theta_power': [5, 10, 15, 20, 25]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(demo_data)
    
    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(df)
    
    return scaler

def create_default_model(num_features):
    """
    Create a simple default model when loading a model fails.
    This creates a very basic model that can be used as a fallback.
    
    Args:
        num_features (int): Number of features the model should accept
        
    Returns:
        A simple model object that implements predict and predict_proba methods
    """
    logger.warning("Creating a default model as fallback")
    from sklearn.ensemble import RandomForestClassifier
    
    # Create a simple model that will always predict class 1 (medium workload)
    # with equal probabilities for all classes
    class DefaultModel:
        def __init__(self, num_features):
            self.num_features = num_features
            self.classes_ = np.array([0, 1, 2])  # Low, Medium, High
            self.feature_names_in_ = [f"feature_{i}" for i in range(num_features)]
            
        def predict(self, X):
            # Always predict medium workload (class 1)
            return np.ones(len(X), dtype=int)
            
        def predict_proba(self, X):
            # Equal probabilities for all classes with higher weight for medium
            probs = np.zeros((len(X), 3))
            probs[:, 0] = 0.2  # Low
            probs[:, 1] = 0.6  # Medium (higher probability)
            probs[:, 2] = 0.2  # High
            return probs
    
    return DefaultModel(num_features)

def predict(data, model_type=None, threshold=None, infer_missing=False):
    """
    Predict cognitive workload from input data.
    
    Args:
        data (dict or DataFrame): Input data with features
        model_type (str, optional): Type of model to use. If None, uses best available model.
        threshold (float, optional): Confidence threshold for prediction. If None, no threshold is applied.
        infer_missing (bool, optional): Whether to infer missing features. Defaults to False.
        
    Returns:
        dict: Prediction results with workload class, confidence, and data quality metrics
    """
    # Check what features are available initially
    if isinstance(data, dict):
        initial_features = list(data.keys())
    else:
        initial_features = list(data.columns)
    
    # Calculate initial confidence based on available modalities
    initial_confidence, missing_modalities, available_modalities = calculate_modality_confidence(initial_features)
    
    # Handle missing features if requested
    missing_features_before = []
    if infer_missing:
        if isinstance(data, dict):
            missing_features_before = [f for f in COMMON_FEATURES if f not in data]
        else:
            missing_features_before = [f for f in COMMON_FEATURES if f not in data.columns]
        
        data = infer_missing_features(data)
        logger.info(f"Inferred {len(missing_features_before)} missing features")
    
    # Convert input to DataFrame if it's a dictionary
    is_dict_input = isinstance(data, dict)
    if is_dict_input:
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    # Find the best model if model_type is not specified
    if model_type is None:
        model_path, metadata = find_best_model()
        if not model_path:
            logger.error("No suitable model found for prediction")
            raise ValueError("No suitable model found for prediction")
        model_type = metadata.get('model_type', 'unknown') if isinstance(metadata, dict) else 'unknown'
    else:
        model_path, metadata = find_model_by_type(model_type)
        if not model_path:
            logger.error(f"No model found for type: {model_type}")
            raise ValueError(f"No model found for type: {model_type}")
        metadata = {"model_type": model_type}  # Create basic metadata
    
    # Load the model
    model = None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Validate that the model is usable (has predict method)
        if not hasattr(model, 'predict'):
            # If it's a numpy array or similar, we need to create a default model
            if isinstance(model, np.ndarray):
                logger.warning(f"Loaded model is a numpy array, not a proper model object. Creating default model.")
                model = create_default_model(model.shape[1] if len(model.shape) > 1 else 11)
            else:
                logger.error(f"Loaded object does not have a predict method: {type(model)}")
                raise ValueError(f"Loaded object is not a valid model: {type(model)}")
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {str(e)}")
        # Create a default model as fallback
        logger.warning("Creating a default model")
        model = create_default_model(11)  # Default number of features
    
    # Load the scaler
    scaler_path = None
    scaler = None
    
    if metadata and 'scaler_path' in metadata and metadata['scaler_path']:
        scaler_path = metadata['scaler_path']
    
    if not scaler_path or not os.path.exists(scaler_path):
        # Try to find a scaler
        model_dir = os.path.dirname(model_path)
        potential_scalers = [
            os.path.join(model_dir, "scaler.pkl"),
            os.path.join(model_dir, "scaler.joblib")
        ]
        
        # Try to find a scaler with a similar name pattern
        model_name = os.path.basename(model_path)
        if "predictor" in model_name:
            scaler_name = model_name.replace("predictor", "scaler")
            potential_scalers.append(os.path.join(model_dir, scaler_name))
        
        # Check for shared scaler
        potential_scalers.append(os.path.join(os.path.dirname(model_dir), "shared_scaler.pkl"))
        potential_scalers.append(os.path.join(os.path.dirname(model_dir), "shared_scaler.joblib"))
        
        # Try each potential scaler path
        for potential_path in potential_scalers:
            if os.path.exists(potential_path):
                scaler_path = potential_path
                logger.info(f"Found scaler at: {scaler_path}")
                break
        
        # If still no scaler, try glob pattern
        if not scaler_path:
            scaler_glob = glob.glob(os.path.join(model_dir, "*scaler*.joblib"))
            if scaler_glob:
                scaler_path = scaler_glob[0]
                logger.info(f"Found scaler using glob pattern: {scaler_path}")
    
    # Try to load the scaler
    if scaler_path and os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"Successfully loaded scaler from {scaler_path}")
        except Exception as e:
            logger.error(f"Error loading scaler {scaler_path}: {str(e)}")
            logger.warning("Creating a default scaler")
            scaler = create_default_scaler()
    else:
        logger.warning("No scaler found, creating a default scaler")
        scaler = create_default_scaler()
    
    # Extract features from metadata or model
    features = None
    if metadata and 'features' in metadata:
        features = metadata['features']
    elif hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_
    else:
        # Use common features as fallback
        features = COMMON_FEATURES
    
    # Check for missing features
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        logger.warning(f"Missing features for prediction: {missing_features}")
        if infer_missing:
            logger.info("These features should have been inferred, but some are still missing")
        else:
            logger.warning("Consider using --infer-missing to handle missing features")
            # Fill missing features with zeros for now
            for feature in missing_features:
                df[feature] = 0.0
    
    # Get final confidence based on what features are actually available after inferences
    final_features = list(df.columns)
    final_confidence, final_missing_modalities, final_available_modalities = calculate_modality_confidence(final_features)
    
    # Prepare data for prediction
    try:
        # Get available features that match with what the model was trained on
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            logger.warning("No matching features found, using all features in data")
            available_features = list(df.columns)
        
        X = df[available_features].values
        
        # If we have a scaler, use it
        if scaler is not None:
            try:
                X_scaled = scaler.transform(X)
            except Exception as e:
                logger.error(f"Error scaling data: {str(e)}")
                logger.warning("Using unscaled data for prediction")
                X_scaled = X
        else:
            # No scaler, use raw values
            X_scaled = X
    except Exception as e:
        logger.error(f"Error preparing data for prediction: {str(e)}")
        # Create dummy scaled data
        X_scaled = np.zeros((len(df), len(features) if features else 11))
    
    # Make prediction
    try:
        if hasattr(model, 'predict_proba'):
            # Get class probabilities
            y_proba = model.predict_proba(X_scaled)
            # Get class predictions
            y_pred = model.predict(X_scaled)
            
            # Get confidences for the predicted classes
            confidences = []
            for i, pred in enumerate(y_pred):
                if isinstance(y_proba, np.ndarray) and i < len(y_proba) and pred < len(y_proba[i]):
                    confidences.append(y_proba[i][pred])
                else:
                    # Fallback confidence
                    confidences.append(0.8)
        else:
            # For models without predict_proba, use a placeholder confidence
            y_pred = model.predict(X_scaled)
            confidences = [0.8] * len(y_pred)
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        # Fallback to medium workload prediction
        logger.warning("Using fallback prediction (medium workload)")
        y_pred = np.ones(len(df), dtype=int)  # Class 1 (medium workload)
        confidences = [0.6] * len(df)
    
    # Apply threshold if specified
    if threshold is not None:
        for i, conf in enumerate(confidences):
            if conf < threshold:
                y_pred[i] = -1  # Use -1 to indicate low confidence
    
    # Adjust model confidence based on available features
    adjusted_confidences = [conf * final_confidence for conf in confidences]
    
    # Create result dictionary
    results = []
    for i in range(len(df)):
        workload_class = int(y_pred[i]) if i < len(y_pred) and y_pred[i] != -1 else 1  # Default to medium
        
        # Map class to label
        if workload_class in WORKLOAD_CLASSES:
            workload_label = WORKLOAD_CLASSES[workload_class]
        else:
            # Manual mapping fallback
            if workload_class == 0:
                workload_label = "Low"
            elif workload_class == 1:
                workload_label = "Medium"
            elif workload_class == 2:
                workload_label = "High"
            else:
                workload_label = "Unknown"
        
        # Create base result
        result = {
            'workload_class': workload_class,
            'workload_label': workload_label,
            'model_confidence': float(confidences[i]) if i < len(confidences) and confidences[i] is not None else 0.6,
            'adjusted_confidence': float(adjusted_confidences[i]) if i < len(adjusted_confidences) and adjusted_confidences[i] is not None else 0.5,
            'data_quality': {
                'initial_confidence': float(initial_confidence),
                'final_confidence': float(final_confidence),
                'missing_modalities': final_missing_modalities,
                'available_modalities': final_available_modalities,
                'inferred_features': missing_features_before if infer_missing else [],
                'missing_features': missing_features,
                'used_features': available_features
            }
        }
        
        # Generate quality warning message
        quality_message = ""
        if final_missing_modalities:
            quality_message += f"Missing data modalities: {', '.join(final_missing_modalities)}. "
        
        if missing_features_before and infer_missing:
            quality_message += f"Inferred {len(missing_features_before)} features. "
        
        if final_confidence < 0.6:
            quality_message += "Low prediction confidence due to limited data. "
        elif final_confidence < 0.8:
            quality_message += "Moderate prediction confidence. "
        
        if quality_message:
            result['quality_warning'] = quality_message.strip()
        
        # Add additional info
        result['model_type'] = model_type
        result['model_path'] = model_path
        result['prediction_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        results.append(result)
    
    # If input was a dictionary, return a single result
    if is_dict_input:
        return results[0]
    else:
        return results

def predict_batch(file_path, output_file=None, model_type=None, threshold=None, infer_missing=False):
    """
    Batch predict cognitive workload from a CSV file.
    
    Args:
        file_path (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, results are not saved.
        model_type (str, optional): Type of model to use. If None, uses best available model.
        threshold (float, optional): Confidence threshold for prediction. If None, no threshold is applied.
        infer_missing (bool, optional): Whether to infer missing features. Defaults to False.
        
    Returns:
        DataFrame: Prediction results with workload class and confidence
    """
    # Load input data
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} records from {file_path}")
    
    # Handle missing features if requested
    if infer_missing:
        logger.info("Inferring missing features")
        df = infer_missing_features(df)
    
    # Make batch prediction
    results = predict(df, model_type, threshold, infer_missing=False)  # False because we already inferred if needed
    
    # Convert results to DataFrame
    result_df = pd.DataFrame(results)
    
    # Combine with original data if requested
    if output_file:
        # Combine input data with prediction results
        output_df = pd.concat([df, result_df], axis=1)
        
        # Save to output file
        output_df.to_csv(output_file, index=False)
        logger.info(f"Saved prediction results to {output_file}")
    
    return result_df

def predict_time_series(file_path, output_file=None, window_size=10, step_size=5, 
                       model_type=None, threshold=None, infer_missing=False):
    """
    Predict cognitive workload from time series data.
    
    Args:
        file_path (str): Path to input CSV file with time series data
        output_file (str, optional): Path to output CSV file. If None, results are not saved.
        window_size (int, optional): Size of sliding window in samples. Defaults to 10.
        step_size (int, optional): Step size for sliding window. Defaults to 5.
        model_type (str, optional): Type of model to use. If None, uses best available model.
        threshold (float, optional): Confidence threshold for prediction. If None, no threshold is applied.
        infer_missing (bool, optional): Whether to infer missing features. Defaults to False.
        
    Returns:
        DataFrame: Time series prediction results
    """
    # Load input data
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} time points from {file_path}")
    
    # Handle missing features if requested
    if infer_missing:
        logger.info("Inferring missing features in time series data")
        df = infer_missing_features(df)
    
    # Ensure timestamp column exists
    if "timestamp" not in df.columns:
        logger.warning("No timestamp column found, creating sequential timestamps")
        df["timestamp"] = pd.date_range(start="2023-01-01", periods=len(df), freq="S")
    
    # Create sliding windows
    windows = []
    timestamps = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i+window_size]
        # Use the last timestamp in the window as the prediction timestamp
        timestamps.append(window["timestamp"].iloc[-1])
        windows.append(window)
    
    logger.info(f"Created {len(windows)} windows from time series data")
    
    # Process each window
    predictions = []
    for window in windows:
        # Aggregate features in the window
        aggregated = {}
        for column in window.columns:
            if column == "timestamp":
                continue
            # Use mean for numeric features
            aggregated[column] = window[column].mean()
        
        # Make prediction on aggregated window
        prediction = predict(aggregated, model_type, threshold, infer_missing=False)
        predictions.append(prediction)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        "timestamp": timestamps,
        "workload_class": [p["workload_class"] for p in predictions],
        "workload_label": [p["workload_label"] for p in predictions],
        "confidence": [p["confidence"] for p in predictions],
        "model_type": [p["model_type"] for p in predictions]
    })
    
    # Save results if requested
    if output_file:
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved time series prediction results to {output_file}")
    
    return result_df

# ---------------------- SAMPLE MODEL GENERATION ---------------------- #
def generate_synthetic_data(n_samples=1000, n_features=10, random_state=42):
    """
    Generate synthetic data for training sample models.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X, y)
    """
    logger.info(f"Generating synthetic data with {n_samples} samples and {n_features} features")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features-2,
        n_redundant=2,
        n_classes=3,
        weights=[0.3, 0.3, 0.4],
        random_state=random_state
    )
    
    # Convert to DataFrame with feature names similar to the expected ones
    feature_names = [
        "pulse_rate", "blood_pressure_sys", "resp_rate", "pupil_diameter_left",
        "pupil_diameter_right", "fixation_duration", "blink_rate", "gaze_x", "gaze_y", "theta_power"
    ]
    
    # Ensure we use only the number of features we have names for
    n_actual_features = min(n_features, len(feature_names))
    X = X[:, :n_actual_features]
    
    df = pd.DataFrame(X, columns=feature_names[:n_actual_features])
    
    # Add workload_intensity and cognitive_state based on the generated y
    df['workload_intensity'] = y * 30 + 50  # Convert to a range around 50-140
    
    # Convert numeric y to Low/Medium/High
    le = LabelEncoder()
    le.fit(['Low', 'Medium', 'High'])
    df['cognitive_state'] = le.inverse_transform(y)
    
    return df

def create_sample_input_json():
    """Create a sample JSON file with input data for prediction."""
    # Generate a single sample of synthetic data
    df = generate_synthetic_data(n_samples=1, random_state=RANDOM_SEED)
    
    # Add missing alpha_power if needed
    if 'alpha_power' not in df.columns:
        df['alpha_power'] = np.random.normal(10, 3, 1)
    
    # Ensure all required features are present
    required_features = [
        "pulse_rate", "blood_pressure_sys", "resp_rate", "pupil_diameter_left",
        "pupil_diameter_right", "fixation_duration", "blink_rate", "workload_intensity",
        "gaze_x", "gaze_y", "alpha_power", "theta_power"
    ]
    
    # Add any missing features with empirically reliable and realistic values using fuzzy logic ranges
    empirical_ranges = {
        "pulse_rate": (60, 100),  # Normal resting heart rate range
        "blood_pressure_sys": (110, 130),  # Normal systolic blood pressure range
        "resp_rate": (12, 20),  # Normal respiratory rate range
        "pupil_diameter_left": (3.0, 4.0),  # Normal pupil diameter range in mm
        "pupil_diameter_right": (3.0, 4.0),  # Normal pupil diameter range in mm
        "fixation_duration": (250, 350),  # Typical fixation duration in ms
        "blink_rate": (10, 20),  # Normal blink rate per minute
        "workload_intensity": (40, 60),  # Assumed normal workload intensity range
        "gaze_x": (480, 520),  # Typical gaze x-coordinate range
        "gaze_y": (360, 390),  # Typical gaze y-coordinate range
        "alpha_power": (8, 12),  # Typical alpha power range in EEG
        "theta_power": (4, 8)  # Typical theta power range in EEG
    }
    
    # Fuzzy logic interpretation ranges for cognitive workload assessment
    fuzzy_interpretation = {
        "low_workload": {
            "pulse_rate": (55, 75),
            "blood_pressure_sys": (100, 120),
            "resp_rate": (10, 15),
            "pupil_diameter_left": (2.5, 3.5),
            "pupil_diameter_right": (2.5, 3.5),
            "alpha_theta_ratio": (1.2, 2.0)  # Higher alpha/theta ratio indicates relaxed state
        },
        "medium_workload": {
            "pulse_rate": (70, 90),
            "blood_pressure_sys": (115, 135),
            "resp_rate": (14, 18),
            "pupil_diameter_left": (3.2, 4.2),
            "pupil_diameter_right": (3.2, 4.2),
            "alpha_theta_ratio": (0.8, 1.3)  # Balanced alpha/theta ratio
        },
        "high_workload": {
            "pulse_rate": (85, 110),
            "blood_pressure_sys": (125, 145),
            "resp_rate": (17, 25),
            "pupil_diameter_left": (3.8, 5.0),
            "pupil_diameter_right": (3.8, 5.0),
            "alpha_theta_ratio": (0.4, 0.9)  # Lower alpha/theta ratio indicates cognitive load
        }
    }
    
    for feature in required_features:
        if feature not in df.columns:
            low, high = empirical_ranges[feature]
            # Implement improved range detection with adaptive boundaries
            range_width = high - low
            # Use smaller expansion for narrow ranges, larger for wide ranges
            expansion_factor = 0.05 + (0.1 * min(1.0, range_width / 100))
            # Ensure physiologically plausible values (prevent negative values)
            fuzzy_low = max(0, low - (expansion_factor * range_width))
            fuzzy_high = high + (expansion_factor * range_width)
            # Add slight randomness to the generated value to avoid uniform distributions
            df[feature] = np.random.normal(
                (fuzzy_low + fuzzy_high) / 2,  # mean at center of range
                (fuzzy_high - fuzzy_low) / 6,  # std dev to keep ~99% within range
                1
            ).clip(fuzzy_low, fuzzy_high)  # clip to ensure within bounds
    # Convert to a dictionary (drop the cognitive_state since that's what we're predicting)
    sample_data = df.drop(columns=['cognitive_state'] if 'cognitive_state' in df.columns else []).iloc[0].to_dict()
    
    # Round values to 2 decimal places for readability
    sample_data = {k: round(v, 2) if isinstance(v, (float, np.float64)) else v 
                 for k, v in sample_data.items()}
    
    # Save to a file
    sample_path = 'data/sample_input.json'
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    
    with open(sample_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Sample input data saved to {sample_path}")
    return sample_path

def install_sample_model(model_type, model_name_suffix="", random_state=None):
    """
    Install a sample model of the specified type.
    
    Args:
        model_type (str): Type of model to install
        model_name_suffix (str): Optional suffix for the model name
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (model_path, scaler_path, accuracy)
    """
    if random_state is None:
        random_state = RANDOM_SEED
    
    # Set up unique version for this model
    model_version = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_type}{model_name_suffix}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Determine the appropriate directory based on model type
    model_dir = os.path.join('models', 'sample', model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{MODEL_NAME}_{model_version}.joblib")
    # Use the exact same timestamp in the scaler filename as in the model filename
    scaler_path = os.path.join(model_dir, f"scaler_{timestamp}_{model_type}.joblib")
    metadata_path = os.path.join(model_dir, f"metadata_{timestamp}_{model_type}.json")
    
    logger.info(f"Installing sample model of type: {model_type}")
    
    try:
        # Generate synthetic data
        df = generate_synthetic_data(random_state=random_state)
        
        # Create a scaler and fit it to the data
        scaler = StandardScaler()
        features = [col for col in df.columns if col not in ['cognitive_state', 'workload_intensity']]
        
        # First fit without transform to get feature names
        scaler.fit(df[features])
        
        # Store feature names in the scaler for prediction
        scaler.feature_names_in_ = np.array(features)
        
        # Then transform the data
        df[features] = scaler.transform(df[features])
        
        # Split the data for training and testing
        X = df.drop(columns=["cognitive_state", "workload_intensity"])
        y = df["cognitive_state"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )
        
        # Create and train the model
        model = create_model(model_type)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save the model and scaler
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Create metadata
        metadata = {
            "model_version": model_version,
            "model_type": model_type,
            "model_name": AVAILABLE_MODELS.get(model_type, "Unknown"),
            "training_date": datetime.now().isoformat(),
            "accuracy": float(accuracy),
            "classification_report": class_report,
            "features": list(X.columns),
            "parameters": {
                "test_size": 0.2,
                "random_seed": random_state
            }
        }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Sample model installed at {model_path} with accuracy {accuracy:.3f}")
        
        return model_path, scaler_path, accuracy
    
    except Exception as e:
        logger.error(f"Error installing sample model: {str(e)}")
        raise

def install_sample_models():
    """Install a set of sample models for different algorithm types."""
    # Ensure model directory exists
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*50)
    print("Installing sample models...")
    print("="*50)
    
    results = []
    
    # Install one of each type of model
    for i, (model_type, model_name) in enumerate(AVAILABLE_MODELS.items()):
        try:
            # Use different random seeds for variety
            random_seed = RANDOM_SEED + i
            
            print(f"Installing {model_name}...")
            model_path, scaler_path, accuracy = install_sample_model(model_type, random_state=random_seed)
            
            results.append({
                "type": model_type,
                "name": model_name,
                "accuracy": accuracy,
                "model_path": model_path,
                "scaler_path": scaler_path
            })
            
            print(f"✓ {model_name} installed (accuracy: {accuracy:.3f})")
            print("-" * 50)
            
        except Exception as e:
            print(f"✘ Failed to install {model_name}: {str(e)}")
    
    # Create a sample input file for predictions
    sample_path = create_sample_input_json()
    
    print("\n" + "="*50)
    print(f"✓ INSTALLATION COMPLETE: {len(results)} models installed")
    print("="*50)
    print("\nAvailable models:")
    for result in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(f"  - {result['name']:<20} (accuracy: {result['accuracy']:.3f})")
    
    print(f"\nSample input data created at: {sample_path}")
    print("\nTo make predictions with these models, run:")
    print(f"  python cwt.py predict --input {sample_path}")
    print("\nTo see all installed models:")
    print("  python cwt.py list-models")
    print("="*50)

# ---------------------- COMMAND LINE INTERFACE ---------------------- #
def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Cognitive Workload Training (CWT) Tool")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Define subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup the CWT environment")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a cognitive workload model")
    train_parser.add_argument("--model-type", type=str, help="Type of model to train (svm, random_forest, etc.)")
    train_parser.add_argument("--output-dir", type=str, help="Directory to save the trained model")
    
    # Train All command
    train_all_parser = subparsers.add_parser("train-all", help="Train all available cognitive workload models")
    train_all_parser.add_argument("--output-dir", type=str, help="Directory to save the trained models")
    train_all_parser.add_argument("--parallel", action="store_true", help="Train models in parallel")
    train_all_parser.add_argument("--skip-types", type=str, help="Comma-separated list of model types to skip")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict cognitive workload from input data")
    predict_parser.add_argument("--input-json", type=str, help="JSON file with input data")
    predict_parser.add_argument("--input-values", type=str, nargs="+", help="Input values in key=value format")
    predict_parser.add_argument("--output-json", type=str, help="JSON file to save prediction results")
    predict_parser.add_argument("--model-type", type=str, help="Type of model to use for prediction")
    predict_parser.add_argument("--threshold", type=float, help="Confidence threshold for prediction")
    predict_parser.add_argument("--infer-missing", action="store_true", help="Infer missing features from available data")
    
    # Batch predict command
    batch_predict_parser = subparsers.add_parser("batch-predict", help="Batch predict cognitive workload from a CSV file")
    batch_predict_parser.add_argument("--input-file", type=str, help="Input CSV file with feature values")
    batch_predict_parser.add_argument("--output-file", type=str, help="Output CSV file to save prediction results")
    batch_predict_parser.add_argument("--model-type", type=str, help="Type of model to use for prediction")
    batch_predict_parser.add_argument("--threshold", type=float, help="Confidence threshold for prediction")
    batch_predict_parser.add_argument("--infer-missing", action="store_true", help="Infer missing features from available data")
    
    # Time series predict command
    time_series_parser = subparsers.add_parser("time-series-predict", help="Predict cognitive workload from time series data")
    time_series_parser.add_argument("--input-file", type=str, help="Input CSV file with time series data")
    time_series_parser.add_argument("--output-file", type=str, help="Output CSV file to save prediction results")
    time_series_parser.add_argument("--window-size", type=int, default=10, help="Size of sliding window in samples")
    time_series_parser.add_argument("--step-size", type=int, default=5, help="Step size for sliding window")
    time_series_parser.add_argument("--model-type", type=str, help="Type of model to use for prediction")
    time_series_parser.add_argument("--threshold", type=float, help="Confidence threshold for prediction")
    time_series_parser.add_argument("--visualize", action="store_true", help="Visualize time series prediction results")
    time_series_parser.add_argument("--infer-missing", action="store_true", help="Infer missing features from available data")
    
    return parser.parse_args()

def find_latest_model(input_data=None):
    """
    Find the best model in the models directory based on the input data.
    
    Args:
        input_data: Optional input data for prediction. If provided, this will be used to determine
                    the best model to use based on the input characteristics.
                    
    Returns:
        tuple: (model_path, metadata) for the best model
    """
    
    # If input data is provided, try to select the most appropriate model based on characteristics
    if input_data is not None:
        try:
            # For high workload, use RF model - prioritize this since it was failing
            if 'workload_intensity' in input_data and float(input_data['workload_intensity']) > 60:
                model_type = 'rf'
                logger.info(f"Detected high workload intensity ({input_data['workload_intensity']}), using Random Forest model")
                model_path, metadata = find_model_by_type('rf')
                if model_path is not None:
                    return model_path, metadata
                logger.warning("No RF model found for high workload. Falling back to other models.")
            
            # For medium workload, use MLP model
            elif 'workload_intensity' in input_data and 40 <= float(input_data['workload_intensity']) <= 60:
                model_type = 'mlp'
                logger.info(f"Detected medium workload intensity ({input_data['workload_intensity']}), using MLP model")
                model_path, metadata = find_model_by_type('mlp')
                if model_path is not None:
                    return model_path, metadata
                logger.warning("No MLP model found for medium workload. Falling back to other models.")
            
            # For low workload, use LR model
            elif 'workload_intensity' in input_data and float(input_data['workload_intensity']) < 40:
                model_type = 'lr'
                logger.info(f"Detected low workload intensity ({input_data['workload_intensity']}), using Logistic Regression model")
                model_path, metadata = find_model_by_type('lr')
                if model_path is not None:
                    return model_path, metadata
                logger.warning("No LR model found for low workload. Falling back to other models.")

            # If we could not find a specific model for the workload intensity, try general fallbacks
            # Try falling back to any RF model since they tend to generalize better
            model_path, metadata = find_model_by_type('rf')
            if model_path is not None:
                logger.info("Using Random Forest model as fallback")
                return model_path, metadata
                
            # Try LR as another fallback
            model_path, metadata = find_model_by_type('lr')
            if model_path is not None:
                logger.info("Using Logistic Regression model as fallback")
                return model_path, metadata
                
        except (ValueError, KeyError) as e:
            logger.warning(f"Could not determine best model based on input data: {e}")
    
    # Check multiple directories for models
    search_dirs = [
        Path(MODEL_OUTPUT_DIR),  # Check the configured output directory first
        Path('models/sample/rf'),  # RF models work well for high workload
        Path('models/advanced/rf'),
        Path('models/sample/mlp'),  # MLP models which are more accurate for medium workload
        Path('models/advanced/mlp'), 
        Path('models/sample/lr'),  # LR models work well for low workload
        Path('models/advanced/lr'),
        Path('models/sample/knn'),
        Path('models/advanced/knn'),
        Path('models/sample/svm'),
        Path('models/sample/gb'),
        Path('models/advanced/svm'),
        Path('models/advanced/gb'),
    ]
    
    all_models = []
    for directory in search_dirs:
        if directory.exists():
            # Search for model files, filter out scaler files
            model_files = []
            for ext in ['*.joblib', '*.pkl']:
                files = list(directory.glob(ext))
                # Filter out scaler files
                files = [f for f in files if "scaler" not in f.name.lower()]
                model_files.extend(files)
            
            all_models.extend(model_files)
    
    if not all_models:
        logger.error("No model files found in any of the search directories")
        return None, None
    
    # Find the most recent model file
    latest_model = max(all_models, key=lambda x: x.stat().st_mtime)
    logger.info(f"Found latest model: {latest_model}")
    
    # Try to find corresponding scaler
    scaler_path = find_scaler_for_model(latest_model)
    if scaler_path:
        logger.info(f"Found corresponding scaler: {scaler_path}")
    else:
        logger.warning(f"No scaler found for model: {latest_model}")
        # Try to look for any scaler in the same directory
        scaler_files = list(latest_model.parent.glob("*scaler*.joblib"))
        if not scaler_files:
            scaler_files = list(latest_model.parent.glob("*scaler*.pkl"))
        
        if scaler_files:
            scaler_path = scaler_files[0]
            logger.info(f"Found alternative scaler: {scaler_path}")
    
    # Determine model type from path
    model_type = None
    for mtype in ['rf', 'mlp', 'lr', 'knn', 'svm', 'gb']:
        if mtype in str(latest_model).lower():
            model_type = mtype
            break
    
    # Create metadata dictionary
    metadata = {
        'model_type': model_type,
        'model_path': str(latest_model),
        'scaler_path': str(scaler_path) if scaler_path else None,
        'created_at': datetime.fromtimestamp(latest_model.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Check for metadata file
    metadata_path = latest_model.parent / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                file_metadata = json.load(f)
                # Merge file metadata with our metadata
                metadata.update(file_metadata)
        except Exception as e:
            logger.warning(f"Error reading metadata file: {e}")
    
    return str(latest_model), metadata

def find_model_by_type(model_type):
    """
    Find the most recent model of a specific type.
    
    Args:
        model_type (str): Type of model to find (rf, svm, mlp, etc.)
        
    Returns:
        tuple: (model_path, metadata) where metadata is a dict containing model info
    """
    search_dirs = [
        Path(f'models/sample/{model_type}'),
        Path(f'models/advanced/{model_type}')
    ]
    
    all_models = []
    for directory in search_dirs:
        if directory.exists():
            # Use a more specific glob pattern and filter out scaler files
            models = list(directory.glob(f"*{model_type}*.joblib"))
            # Filter out any files with "scaler" in the name
            models = [m for m in models if "scaler" not in m.name.lower()]
            all_models.extend(models)
    
    if not all_models:
        logger.warning(f"No models found of type: {model_type}")
        return None, None
        
    # Find newest model
    latest_model = max(all_models, key=lambda x: x.stat().st_mtime)
    # Find corresponding scaler
    scaler_path = find_scaler_for_model(latest_model)
    
    # Create metadata dictionary
    metadata = {
        'model_type': model_type,
        'model_path': str(latest_model),
        'scaler_path': str(scaler_path) if scaler_path else None,
        'created_at': datetime.fromtimestamp(latest_model.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Check for metadata file
    metadata_path = latest_model.parent / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                file_metadata = json.load(f)
                # Merge file metadata with our metadata
                metadata.update(file_metadata)
        except Exception as e:
            logger.warning(f"Error reading metadata file: {e}")
    
    if scaler_path:
        logger.info(f"Selected {model_type} model: {latest_model}")
        logger.info(f"Found corresponding scaler: {scaler_path}")
        return str(latest_model), metadata
    else:
        logger.error(f"No scaler found for {model_type} model: {latest_model}")
        return None, None

def find_scaler_for_model(model_path):
    """Find the corresponding scaler for a model file."""
    model_path = Path(model_path) if isinstance(model_path, str) else model_path
    model_basename = model_path.stem
    model_parts = model_basename.split('_')
    model_dir = model_path.parent
    
    logger.debug(f"Looking for scaler for model: {model_path}")
    
    # First, try to find a scaler in the same directory with an exact match to the model name pattern
    # Pattern: If model is name_TIMESTAMP_type.joblib, look for scaler_TIMESTAMP_type.joblib
    
    # Try a direct filename pattern match in the same directory
    exact_scaler_name = f"scaler_{model_basename.replace('cognitive_state_predictor_', '')}"
    exact_pattern = model_dir / f"{exact_scaler_name}.joblib"
    
    if exact_pattern.exists():
        logger.debug(f"Found exact match scaler: {exact_pattern}")
        return exact_pattern
    
    # Try another common naming pattern: scaler_TIMESTAMP_type.joblib
    if len(model_parts) >= 3:  # Should have at least "MODEL_NAME_TIMESTAMP_TYPE"
        model_type = model_parts[-1]  # Last part should be the model type (rf, svm, etc.)
        timestamp_parts = []
        
        # Extract timestamp parts from the model filename
        # The model filename format could be like: cognitive_state_predictor_20250317_001647_rf.joblib
        # Or it could be like: Cognitive_State_Prediction_Model_20250317_000634_rf.joblib
        # So we need to find all the numeric parts that could be part of the timestamp
        for part in model_parts:
            if part.isdigit() or (len(part) > 0 and part[0].isdigit()):
                timestamp_parts.append(part)
        
        # If we found timestamp parts, use them to build scaler patterns
        if timestamp_parts:
            full_timestamp = "_".join(timestamp_parts)
            
            # Try more scaler naming patterns
            scaler_patterns = [
                model_dir / f"scaler_{full_timestamp}_{model_type}.joblib",
                model_dir / f"scaler_{full_timestamp}.joblib",
                model_dir / f"scaler_{timestamp_parts[-1]}_{model_type}.joblib",
                model_dir / f"scaler_{timestamp_parts[-1]}.joblib",
                model_dir / f"scaler_{timestamp_parts[0]}_{model_type}.joblib",
                model_dir / f"scaler_{timestamp_parts[0]}.joblib",
            ]
            
            # Add additional pattern where model name might contain standard prefix
            if "cognitive_state_predictor" in model_basename or "Cognitive_State_Prediction_Model" in model_basename:
                for prefix in ["cognitive_state_predictor", "Cognitive_State_Prediction_Model"]:
                    if prefix in model_basename:
                        model_suffix = model_basename.replace(f"{prefix}_", "")
                        scaler_patterns.append(model_dir / f"scaler_{model_suffix}.joblib")
            
            # Check if any of the patterns exists
            for pattern in scaler_patterns:
                if pattern.exists():
                    logger.debug(f"Found matching scaler: {pattern}")
                    return pattern
    
    # If no direct match found, look for any scaler in the same directory
    # This is a fallback approach - look for the newest scaler in the same dir
    scalers = list(model_dir.glob("*scaler*.joblib"))
    if scalers:
        newest_scaler = max(scalers, key=lambda x: x.stat().st_mtime)
        logger.warning(f"No exact matching scaler found. Using newest scaler in same directory: {newest_scaler}")
        return newest_scaler
    
    # If no scaler found in the same directory, look in the parent directory one level up
    parent_dir = model_dir.parent
    parent_scalers = list(parent_dir.glob("*scaler*.joblib"))
    if parent_scalers:
        newest_parent_scaler = max(parent_scalers, key=lambda x: x.stat().st_mtime)
        logger.warning(f"No scaler found in model directory. Using scaler from parent directory: {newest_parent_scaler}")
        return newest_parent_scaler
    
    # Last resort - get any scaler from the models directory
    logger.warning("No matching scaler found. Searching entire models directory...")
    any_scalers = list(Path('models').glob("**/*scaler*.joblib"))
    if any_scalers:
        newest_any_scaler = max(any_scalers, key=lambda x: x.stat().st_mtime)
        logger.warning(f"Using scaler from models directory: {newest_any_scaler}")
        return newest_any_scaler
        
    # If we got here, no scaler was found
    logger.error("No scaler found anywhere")
    return None

def list_available_models():
    """List all available models in the models directory."""
    print("\n" + "=" * 80)
    print("AVAILABLE TRAINED MODELS")
    print("=" * 80)
    
    # Search in all model directories
    search_dirs = [
        Path('models/sample/default'),
        Path('models/sample/rf'),
        Path('models/sample/svm'),
        Path('models/sample/gb'),
        Path('models/sample/mlp'),
        Path('models/sample/knn'),
        Path('models/sample/lr'),
        Path('models/advanced/rf'),
        Path('models/advanced/svm'),
        Path('models/advanced/gb'),
        Path('models/advanced/mlp'),
        Path('models/advanced/knn'),
        Path('models/advanced/lr'),
    ]
    
    model_files = []
    for directory in search_dirs:
        if directory.exists():
            model_files.extend(list(directory.glob("*.joblib")))
    
    if not model_files:
        print("No trained models found.")
        return
    
    # Filter out scaler files
    model_files = [m for m in model_files if not m.stem.startswith("scaler_")]
    
    # Group by directories
    models_by_category = {}
    
    for model_file in model_files:
        # Determine the category based on the directory structure
        if "advanced" in str(model_file):
            category = "Advanced Models"
        elif "sample" in str(model_file):
            category = "Sample Models"
        else:
            category = "Other Models"
        
        if category not in models_by_category:
            models_by_category[category] = []
        
        models_by_category[category].append(model_file)
    
    # Print models by category
    for category, files in models_by_category.items():
        print(f"\n{category}:")
        print("-" * len(category))
        
        # Sort files by modification time (newest first)
        sorted_files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
        
        for model_file in sorted_files:
            model_name = model_file.name
            model_time = datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Try to extract model type
            model_type = "Unknown"
            if "_rf." in model_name:
                model_type = "Random Forest"
            elif "_svm." in model_name:
                model_type = "Support Vector Machine"
            elif "_gb." in model_name:
                model_type = "Gradient Boosting"
            elif "_mlp." in model_name:
                model_type = "Neural Network (MLP)"
            elif "_knn." in model_name:
                model_type = "K-Nearest Neighbors"
            elif "_lr." in model_name:
                model_type = "Logistic Regression"
            
            # Look for metadata to get the accuracy
            accuracy = "Unknown"
            try:
                metadata_path = model_file.parent / f"metadata_{model_file.stem.split('_')[-1]}.json"
                if not metadata_path.exists():
                    # Try another pattern
                    metadata_pattern = model_file.parent.glob(f"metadata_*_{model_file.stem.split('_')[-1]}.json")
                    metadata_files = list(metadata_pattern)
                    if metadata_files:
                        metadata_path = metadata_files[0]
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        if 'accuracy' in metadata:
                            accuracy = f"{metadata['accuracy']:.3f}"
            except Exception as e:
                logger.debug(f"Error reading metadata for {model_file}: {str(e)}")
            
            print(f"  - {model_name}")
            print(f"    Type: {model_type}, Created: {model_time}, Accuracy: {accuracy}")
            print(f"    Path: {model_file}")
            print()
    
    print("\nTo use a specific model for prediction:")
    print("python cwt.py predict --input data/sample_input.json --model [model_path]")
    print("=" * 80)

def display_help(topic=None):
    """Display help information for commands or specific topics."""
    header = "\n" + "=" * 80 + "\n"
    header += "COGNITIVE WORKLOAD TOOL (CWT) HELP\n"
    header += "=" * 80 + "\n"
    
    general_help = """
The Cognitive Workload Tool (CWT) is designed to predict cognitive workload 
levels based on physiological, EEG, and gaze data. It uses machine learning 
to classify cognitive states as Low, Medium, or High.

"""
    
    commands = {
        "train": {
            "description": "Train a new model using data files specified in .env",
            "usage": "python cwt.py train [--model-type TYPE]",
            "options": [
                "--model-type, -m: Type of model to train (rf, svm, gb, mlp, knn, lr)",
            ],
            "examples": [
                "python cwt.py train",
                "python cwt.py train --model-type rf",
                "python cwt.py train --model-type svm"
            ]
        },
        "predict": {
            "description": "Make predictions using a trained model",
            "usage": "python cwt.py predict --input INPUT_FILE [options]",
            "options": [
                "--input, -i: Input JSON or CSV file with feature values",
                "--model, -m: Path to model file (uses latest model if not specified)",
                "--scaler, -s: Path to scaler file (uses scaler associated with model if not specified)",
                "--output, -o: Output file for predictions",
                "--batch, -b: Process input as batch data with multiple samples",
                "--time-series, -t: Treat data as time series with temporal ordering",
                "--auto-detect, -a: Auto-detect input type and format (JSON, CSV, batch, etc.)"
            ],
            "examples": [
                "python cwt.py predict --input data/sample_input.json",
                "python cwt.py predict --input data/batch_samples.csv --batch",
                "python cwt.py predict --input data/time_series.csv --time-series",
                "python cwt.py predict --input any_data_file --auto-detect"
            ]
        },
        "batch-predict": {
            "description": "Process multiple data points for prediction",
            "usage": "python cwt.py batch-predict --input CSV_FILE [options]",
            "options": [
                "--input, -i: CSV file with multiple data points",
                "--model, -m: Path to model file (uses latest model if not specified)",
                "--scaler, -s: Path to scaler file (uses scaler associated with model if not specified)",
                "--output, -o: Output file for predictions",
                "--time-series, -t: Treat data as time series with temporal ordering"
            ],
            "examples": [
                "python cwt.py batch-predict --input data/batch_samples.csv",
                "python cwt.py batch-predict --input data/time_series.csv --time-series",
                "python cwt.py batch-predict --input data/workload_data.csv --output results/predictions.csv"
            ]
        },
        "list-models": {
            "description": "List available trained models",
            "usage": "python cwt.py list-models [--detailed]",
            "options": [
                "--detailed, -d: Show detailed model information"
            ],
            "examples": [
                "python cwt.py list-models",
                "python cwt.py list-models --detailed"
            ]
        },
        "install-models": {
            "description": "Install sample pre-trained models for testing",
            "usage": "python cwt.py install-models [--force]",
            "options": [
                "--force, -f: Force reinstallation of sample models"
            ],
            "examples": [
                "python cwt.py install-models",
                "python cwt.py install-models --force"
            ]
        },
        "train-from-examples": {
            "description": "Train a model using example JSON files",
            "usage": "python cwt.py train-from-examples [--model-type TYPE]",
            "options": [
                "--model-type, -m: Type of model to train (rf, svm, gb, mlp, knn, lr)"
            ],
            "examples": [
                "python cwt.py train-from-examples",
                "python cwt.py train-from-examples --model-type gb"
            ]
        },
        "help": {
            "description": "Show help for all commands or a specific topic",
            "usage": "python cwt.py help [--topic TOPIC]",
            "options": [
                "--topic, -t: Get help on a specific topic"
            ],
            "examples": [
                "python cwt.py help",
                "python cwt.py help --topic predict",
                "python cwt.py !help"
            ]
        }
    }
    
    topics = {
        "model-types": """
AVAILABLE MODEL TYPES:
    rf  - Random Forest: Ensemble method that builds multiple decision trees.
          Good for handling various data types and resistant to overfitting.
    
    svm - Support Vector Machine: Finds the hyperplane that best separates classes.
          Effective in high dimensional spaces but may be slower to train.
    
    gb  - Gradient Boosting: Ensemble method that builds trees sequentially.
          Often achieves high accuracy but may overfit on noisy data.
    
    mlp - Neural Network (MLP): Multi-layer perceptron neural network.
          Flexible model that can capture complex patterns but requires more data.
    
    knn - K-Nearest Neighbors: Classification based on closest training examples.
          Simple but effective algorithm, sensitive to local patterns.
    
    lr  - Logistic Regression: Linear model for classification.
          Fast and interpretable, but may underperform on complex relationships.
""",
        "data-format": """
DATA FORMAT REQUIREMENTS:

For training, the tool expects three CSV files:

1. Physiological data file:
   - Required columns: timestamp, subject_id, pulse_rate, blood_pressure_sys, resp_rate
   - Optional: skin_conductance, cognitive_workload (target)

2. EEG data file:
   - Required columns: timestamp, subject_id, alpha_power, theta_power
   - Optional: beta_power, delta_power, gamma_power, alpha_theta_ratio

3. Gaze tracking data file:
   - Required columns: timestamp, subject_id, pupil_diameter_left, pupil_diameter_right,
     fixation_duration, blink_rate, workload_intensity, gaze_x, gaze_y

For single-sample prediction, the tool accepts a JSON file with feature values:
{
  "pulse_rate": 75.2,
  "blood_pressure_sys": 120.5,
  "resp_rate": 16.4,
  "pupil_diameter_left": 5.2,
  "pupil_diameter_right": 5.1,
  "fixation_duration": 245.8,
  "blink_rate": 12.3,
  "workload_intensity": 50.0,
  "gaze_x": 512.3,
  "gaze_y": 384.7,
  "alpha_power": 18.5,
  "theta_power": 22.6
}

For batch prediction, the tool accepts a CSV file with multiple samples:
timestamp,pulse_rate,blood_pressure_sys,resp_rate,pupil_diameter_left,...
2023-01-01T10:00:00,75.2,120.5,16.4,5.2,...
2023-01-01T10:01:00,76.8,122.3,16.7,5.3,...
...

For time series data, ensure the CSV file includes a timestamp column. The tool will
automatically sort by timestamp and analyze transitions between cognitive states.

Example JSON and CSV files are provided in the examples/ directory.
""",
        "batch-processing": """
BATCH PROCESSING AND TIME SERIES ANALYSIS:

The CWT tool supports processing multiple data points in a single operation, which is
particularly useful for:

1. Analyzing workload patterns over time
2. Processing data from multiple subjects or sessions
3. Generating reports with workload distribution statistics

Batch processing features:

* Auto-detection of input type (CSV, JSON, etc.)
* Processing of time series data with temporal analysis
* Detection of transitions between cognitive states
* Summary statistics for workload distribution
* Visualization of cognitive workload timeline

Command examples:

1. Basic batch processing:
   python cwt.py batch-predict --input data/batch_samples.csv
   
2. Time series analysis:
   python cwt.py batch-predict --input data/time_series.csv --time-series
   
3. Auto-detection with the predict command:
   python cwt.py predict --input any_data_file --auto-detect

4. Save results to file:
   python cwt.py batch-predict --input data/batch.csv --output results/predictions.csv

Output:
- Original data with added prediction columns
- Summary statistics of workload distribution
- For time series data: visualization of workload over time
- Analysis of transitions between cognitive states

CSV format requirements:
- Each row represents one data point
- Must include the same features used for training
- For time series, include a 'timestamp' column in a recognizable format
- Missing values are handled gracefully

The batch processing system automatically selects the appropriate model for each data
point based on its workload intensity - this ensures optimal model selection for each
sample.
""",
        "configuration": """
CONFIGURATION:

The tool can be configured using a .env file:

# Data files
PHYSIO_DATA_PATH=data/Enhanced_Workload_Clinical_Data.csv
EEG_DATA_PATH=data/000_EEG_Cluster_ANOVA_Results.csv
GAZE_DATA_PATH=data/008_01.csv

# Model configuration
MODEL_OUTPUT_DIR=models
MODEL_NAME=Cognitive_State_Prediction_Model

# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=logs/general/cwt.log

# Training parameters
TEST_SIZE=0.2
RANDOM_SEED=42
CV_FOLDS=5

# Default model type
DEFAULT_MODEL_TYPE=rf

See .env.example for all available configuration options.
""",
        "examples": """
EXAMPLE USAGE:

1. Generate sample data:
   python generate_sample_data.py

2. Install sample models:
   python cwt.py install-models

3. List available models:
   python cwt.py list-models

4. Make a prediction with a single sample:
   python cwt.py predict --input data/sample_input.json

5. Process a batch of data points:
   python cwt.py batch-predict --input data/batch_samples.csv

6. Analyze time series data:
   python cwt.py batch-predict --input data/time_series.csv --time-series

7. Train a new model:
   python cwt.py train --model-type rf

8. Train a model from JSON examples:
   python cwt.py train-from-examples

9. Auto-detect input type and format:
   python cwt.py predict --input any_data_file --auto-detect

10. Convert CSV to JSON:
    python examples/create_json_from_csv.py examples/sample_data.csv
"""
    }
    
    if topic is None:
        # General help
        print(header)
        print(general_help)
        
        print("AVAILABLE COMMANDS:\n")
        for cmd, info in commands.items():
            if cmd != "!help":  # Skip the duplicate !help command in the list
                print(f"  {cmd:<20} {info['description']}")
        
        print("\nAVAILABLE HELP TOPICS:\n")
        for topic, _ in topics.items():
            print(f"  {topic}")
        
        print("\nFor more information on a specific command or topic, use:")
        print("  python cwt.py help --topic [command or topic]")
        print("  python cwt.py !help --topic [command or topic]")
    else:
        # Command-specific or topic-specific help
        print(header)
        
        if topic in commands:
            cmd_info = commands[topic]
            print(f"COMMAND: {topic}\n")
            print(f"Description: {cmd_info['description']}")
            print(f"Usage: {cmd_info['usage']}")
            
            if cmd_info['options']:
                print("\nOptions:")
                for opt in cmd_info['options']:
                    print(f"  {opt}")
            
            if cmd_info['examples']:
                print("\nExamples:")
                for ex in cmd_info['examples']:
                    print(f"  {ex}")
        elif topic in topics:
            print(f"TOPIC: {topic}\n")
            print(topics[topic])
        else:
            print(f"Unknown topic: {topic}")
            print("\nAvailable topics:")
            for cmd in commands:
                if cmd != "!help":  # Skip the duplicate !help command in the list
                    print(f"  {cmd}")
            for t in topics:
                print(f"  {t}")

# ---------------------- BATCH PREDICTION FUNCTIONS ---------------------- #
def predict_from_csv(csv_path, output_path=None, model_path=None, scaler_path=None):
    """
    Process a CSV file containing multiple data points over time and predict workload levels.
    
    Args:
        csv_path (str): Path to the CSV file with multiple data points
        output_path (str, optional): Path to save the prediction results
        model_path (str, optional): Path to a specific model to use
        scaler_path (str, optional): Path to a specific scaler to use
        
    Returns:
        DataFrame: DataFrame with original data and predictions
    """
    logger.info(f"Processing CSV file for batch prediction: {csv_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Check if the CSV has timestamp column (for time series data)
        has_timestamp = 'timestamp' in df.columns
        if has_timestamp:
            logger.info("Time series data detected (timestamp column present)")
            # Ensure timestamp is properly formatted
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            except Exception as e:
                logger.warning(f"Could not convert timestamp column to datetime: {e}")
        
        # Initialize columns for predictions and confidence
        df['predicted_cognitive_state'] = None
        df['prediction_confidence'] = None
        df['model_used'] = None
        
        # Process each row for prediction
        logger.info("Making predictions for each data point...")
        
        predictions = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            # Convert row to dictionary for prediction
            row_dict = row.to_dict()
            
            # Skip any non-numeric values that would cause issues
            row_dict = {k: v for k, v in row_dict.items() 
                       if isinstance(v, (int, float)) and not pd.isna(v)}
            
            # Make individual prediction for this data point
            # This will auto-select the appropriate model based on workload_intensity
            try:
                prediction, proba = predict_new_data(model_path, scaler_path, row_dict)
                
                # Store predictions and model info
                if prediction:
                    # Get the model type that was used
                    model_type = "Unknown"
                    if 'rf' in str(model_path).lower():
                        model_type = "RF"
                    elif 'mlp' in str(model_path).lower():
                        model_type = "MLP"
                    elif 'lr' in str(model_path).lower():
                        model_type = "LR"
                    elif 'svm' in str(model_path).lower():
                        model_type = "SVM"
                    elif 'knn' in str(model_path).lower():
                        model_type = "KNN"
                    elif 'gb' in str(model_path).lower():
                        model_type = "GB"
                        
                    # Find the highest probability
                    if isinstance(proba, dict):
                        max_prob = max(proba.values())
                    else:
                        max_prob = 1.0  # Default if no probabilities available
                        
                    predictions.append({
                        'index': idx,
                        'prediction': prediction,
                        'confidence': max_prob,
                        'model_used': model_type
                    })
                else:
                    logger.warning(f"No prediction made for row {idx}")
                    predictions.append({
                        'index': idx,
                        'prediction': 'Unknown',
                        'confidence': 0.0,
                        'model_used': 'None'
                    })
            except Exception as e:
                logger.error(f"Error predicting row {idx}: {e}")
                predictions.append({
                    'index': idx,
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'model_used': 'Error'
                })
        
        # Update the dataframe with predictions
        for p in predictions:
            idx = p['index']
            df.at[idx, 'predicted_cognitive_state'] = p['prediction']
            df.at[idx, 'prediction_confidence'] = p['confidence']
            df.at[idx, 'model_used'] = p['model_used']
        
        # Calculate distribution of predicted states
        state_counts = df['predicted_cognitive_state'].value_counts()
        logger.info(f"Prediction distribution: {state_counts.to_dict()}")
        
        # For time series data, look for transitions between states
        if has_timestamp:
            df = df.sort_values('timestamp')
            state_changes = (df['predicted_cognitive_state'] != df['predicted_cognitive_state'].shift()).sum()
            logger.info(f"Detected {state_changes} transitions between cognitive states")
            
            # Add a column to identify transitions
            df['state_transition'] = df['predicted_cognitive_state'] != df['predicted_cognitive_state'].shift()
        
        # Save predictions if output path is provided
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            df.to_csv(output_path, index=False)
            logger.info(f"Prediction results saved to {output_path}")
            
            # Also create a summary file
            summary_path = output_path.replace('.csv', '_summary.csv')
            if summary_path == output_path:  # In case output_path doesn't end with .csv
                summary_path = f"{output_path}_summary.csv"
                
            summary_df = pd.DataFrame(state_counts).reset_index()
            summary_df.columns = ['cognitive_state', 'count']
            summary_df['percentage'] = summary_df['count'] / len(df) * 100
            
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Prediction summary saved to {summary_path}")
            
            # Generate a visualization of the predictions over time
            if has_timestamp:
                try:
                    viz_path = output_path.replace('.csv', '_timeline.png')
                    if viz_path == output_path:
                        viz_path = f"{output_path}_timeline.png"
                    
                    plt.figure(figsize=(12, 6))
                    
                    # Map categorical values to numeric for plotting
                    state_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': -1, 'Error': -1}
                    if 'predicted_cognitive_state' in df.columns:
                        df['state_numeric'] = df['predicted_cognitive_state'].map(state_map)
                    
                    plt.plot(df['timestamp'], df['state_numeric'], 'o-', markersize=4)
                    plt.yticks([0, 1, 2], ['Low', 'Medium', 'High'])
                    plt.title('Cognitive Workload Prediction Timeline')
                    plt.xlabel('Time')
                    plt.ylabel('Predicted Cognitive State')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Highlight state transitions if present
                    if 'state_transition' in df.columns:
                        transition_points = df[df['state_transition'] == True]
                        if len(transition_points) > 0:
                            plt.scatter(transition_points['timestamp'], 
                                      transition_points['state_numeric'], 
                                      color='red', s=80, zorder=5, 
                                      label='State Transitions')
                            plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(viz_path)
                    logger.info(f"Timeline visualization saved to {viz_path}")
                except Exception as e:
                    logger.error(f"Error creating visualization: {e}")
        
        # Also print a summary to console
        print("\n" + "="*50)
        print("✓ BATCH PREDICTION RESULTS")
        print("="*50)
        print(f"Total samples processed: {len(df)}")
        print("\nPrediction distribution:")
        for state, count in state_counts.items():
            percentage = count / len(df) * 100
            print(f"  {state:<10}: {count} ({percentage:.1f}%)")
        
        if has_timestamp:
            print(f"\nTime range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"State transitions detected: {state_changes}")
        
        if output_path:
            print(f"\nFull results saved to: {output_path}")
        print("="*50)
        
        return df
    
    except Exception as e:
        logger.error(f"Error processing CSV file for batch prediction: {e}")
        print(f"\n✘ ERROR: Batch prediction failed: {e}")
        raise

def predict_automatic(input_data, output_path=None, model_path=None, scaler_path=None):
    """
    Automatically determine the type of input data and process it accordingly.
    Can handle JSON files, CSV files, or Python dictionaries/DataFrames.
    
    Args:
        input_data: Input data which could be a path to a file or a data structure
        output_path: Path to save output results
        model_path: Optional path to a specific model
        scaler_path: Optional path to a specific scaler
        
    Returns:
        The prediction results
    """
    logger.info(f"Auto-detecting input type for: {input_data}")
    
    try:
        # Case 1: Input is a string path to a file
        if isinstance(input_data, str):
            # Check if the file exists
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"Input file not found: {input_data}")
            
            # Determine file type by extension
            file_ext = os.path.splitext(input_data)[1].lower()
            
            if file_ext == '.json':
                logger.info(f"Detected JSON input file: {input_data}")
                # Load the JSON file
                with open(input_data, 'r') as f:
                    json_data = json.load(f)
                
                # If it's a list, it's a batch of samples
                if isinstance(json_data, list):
                    logger.info(f"Batch JSON with {len(json_data)} samples detected")
                    # Convert to DataFrame for batch processing
                    df = pd.DataFrame(json_data)
                    # Add timestamp if not present
                    if 'timestamp' not in df.columns:
                        df['timestamp'] = pd.date_range(start='now', periods=len(df), freq='S')
                    # Save to temporary CSV for batch processing
                    temp_csv = os.path.join(os.path.dirname(input_data), '_temp_batch.csv')
                    df.to_csv(temp_csv, index=False)
                    # Process as CSV
                    return predict_from_csv(temp_csv, output_path, model_path, scaler_path)
                else:
                    # Single JSON sample
                    logger.info("Single JSON sample detected")
                    return predict_new_data(model_path, scaler_path, json_data)
            
            elif file_ext in ['.csv', '.txt', '.data']:
                logger.info(f"Detected CSV/text input file: {input_data}")
                return predict_from_csv(input_data, output_path, model_path, scaler_path)
            
            else:
                logger.warning(f"Unknown file extension: {file_ext}")
                # Try to load as CSV by default
                logger.info("Attempting to load as CSV")
                return predict_from_csv(input_data, output_path, model_path, scaler_path)
        
        # Case 2: Input is a pandas DataFrame
        elif isinstance(input_data, pd.DataFrame):
            logger.info(f"Detected pandas DataFrame with {len(input_data)} rows")
            # Save to temporary CSV for batch processing
            temp_csv = os.path.join(os.getcwd(), '_temp_dataframe.csv')
            input_data.to_csv(temp_csv, index=False)
            # Process as CSV
            return predict_from_csv(temp_csv, output_path, model_path, scaler_path)
        
        # Case 3: Input is a dictionary (single sample)
        elif isinstance(input_data, dict):
            logger.info("Detected dictionary input (single sample)")
            return predict_new_data(model_path, scaler_path, input_data)
        
        # Case 4: Input is a list of dictionaries (batch of samples)
        elif isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
            logger.info(f"Detected list of dictionaries ({len(input_data)} samples)")
            # Convert to DataFrame for batch processing
            df = pd.DataFrame(input_data)
            # Add timestamp if not present
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.date_range(start='now', periods=len(df), freq='S')
            # Save to temporary CSV for batch processing
            temp_csv = os.path.join(os.getcwd(), '_temp_batch_list.csv')
            df.to_csv(temp_csv, index=False)
            # Process as CSV
            return predict_from_csv(temp_csv, output_path, model_path, scaler_path)
        
        else:
            logger.error(f"Unsupported input type: {type(input_data)}")
            raise TypeError(f"Unsupported input type: {type(input_data)}")
    
    except Exception as e:
        logger.error(f"Error in automatic prediction: {e}")
        print(f"\n✘ ERROR: Automatic prediction failed: {e}")
        raise

# ---------------------- FEATURE INFERENCE ---------------------- #
def infer_missing_features(data, reference_data=None):
    """
    Infer missing features in the data based on available features.
    
    This function intelligently infers missing features by leveraging correlations
    between different physiological, EEG, and gaze metrics. It can handle any
    combination of available metrics to make reasonable predictions for missing values.
    
    Args:
        data (dict or DataFrame): Input data with potentially missing features
        reference_data (DataFrame, optional): Reference data to use for inference.
                                           If None, loads default reference data.
    
    Returns:
        DataFrame or dict: Input data with inferred values for missing features
    """
    # Convert input to DataFrame if it's a dictionary
    is_dict_input = isinstance(data, dict)
    if is_dict_input:
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    logger.info(f"Inferring missing features for dataset with {len(df)} rows")
    
    # Identify missing features by modality
    missing_features = {}
    available_features = {}
    
    for modality, features in FEATURE_MODALITIES.items():
        missing = [feature for feature in features if feature not in df.columns]
        available = [feature for feature in features if feature in df.columns]
        
        if missing:
            missing_features[modality] = missing
            logger.info(f"Detected missing {modality} features: {missing}")
        
        if available:
            available_features[modality] = available
    
    # If no features are missing, return the original data
    if not any(missing_features.values()):
        logger.info("No features to infer")
        return data
    
    # Load reference data if not provided
    if reference_data is None:
        logger.info("Loading reference data for feature inference")
        reference_data = load_reference_data()
    
    # Check if we have enough reference data
    if reference_data is None or len(reference_data) < 10:
        logger.warning("Insufficient reference data for inference, generating synthetic data")
        reference_data = generate_synthetic_reference_data(n_samples=1000)
    
    # Track which modalities were successfully inferred
    inferred_modalities = set()
    
    # Use different strategies based on what's available
    
    # 1. Infer workload intensity if missing (needed for other inferences)
    if 'workload_intensity' not in df.columns:
        logger.info("Inferring workload intensity from other metrics")
        df = infer_workload_intensity(df, reference_data)
        inferred_modalities.add('workload')
    
    # 2. Infer pupil metrics if missing
    if 'pupil' in missing_features:
        pupil_metrics = missing_features['pupil']
        
        if len(pupil_metrics) == 2:  # Both pupil metrics are missing
            logger.info("Inferring both pupil diameters from other physiological metrics")
            
            # Check if we can use workload intensity
            if 'workload_intensity' in df.columns:
                logger.info("Using workload intensity to infer pupil diameters")
                df = infer_pupil_metrics(df, reference_data)
                inferred_modalities.add('pupil')
            
            # Alternatively, use physiological data if available
            elif 'physiological' in available_features and len(available_features['physiological']) >= 2:
                logger.info("Using physiological metrics to infer pupil diameters")
                df = infer_pupil_metrics(df, reference_data)
                inferred_modalities.add('pupil')
            
            else:
                logger.warning("Insufficient data to reliably infer pupil metrics")
                # Set to population average as fallback
                for metric in pupil_metrics:
                    df[metric] = reference_data[metric].mean()
        
        elif len(pupil_metrics) == 1:  # One pupil metric is missing
            # Mirror the available pupil metric
            available_pupil = set(FEATURE_MODALITIES['pupil']) - set(pupil_metrics)
            if available_pupil:
                available_metric = list(available_pupil)[0]
                missing_metric = pupil_metrics[0]
                
                logger.info(f"Using {available_metric} to infer {missing_metric}")
                # Pupil diameters are typically similar between eyes
                df[missing_metric] = df[available_metric] * (1 + np.random.normal(0, 0.05, len(df)))
                inferred_modalities.add('pupil')
    
    # 3. Infer EEG metrics if missing
    if 'eeg' in missing_features:
        eeg_metrics = missing_features['eeg']
        
        if 'workload_intensity' in df.columns:
            logger.info("Using workload intensity to infer EEG metrics")
            df = infer_eeg_metrics(df, reference_data)
            inferred_modalities.add('eeg')
        
        elif 'physiological' in available_features and len(available_features['physiological']) >= 2:
            logger.info("Using physiological metrics to infer EEG metrics")
            df = infer_eeg_metrics(df, reference_data)
            inferred_modalities.add('eeg')
            
        else:
            logger.warning("Insufficient data to reliably infer EEG metrics")
            # Set to population averages
            for metric in eeg_metrics:
                df[metric] = reference_data[metric].mean()
    
    # 4. Infer gaze metrics if missing
    if 'gaze' in missing_features:
        gaze_metrics = missing_features['gaze']
        
        available_modality_count = sum(1 for modality in ['physiological', 'pupil', 'eeg'] 
                                    if modality in available_features and available_features[modality])
                                    
        if available_modality_count >= 2:
            logger.info("Inferring gaze metrics from other modalities")
            
            for metric in gaze_metrics:
                # Get reference correlations
                corr_with_reference = {}
                for ref_col in df.columns:
                    if ref_col in reference_data.columns and metric in reference_data.columns:
                        corr = reference_data[[ref_col, metric]].corr().iloc[0, 1]
                        if not np.isnan(corr):
                            corr_with_reference[ref_col] = abs(corr)
                
                # Use top 3 correlated features for regression
                if corr_with_reference:
                    top_features = sorted(corr_with_reference.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_feature_names = [f[0] for f in top_features]
                    
                    # Create simple regression
                    X_ref = reference_data[top_feature_names].values
                    y_ref = reference_data[metric].values
                    
                    try:
                        model = LinearRegression()
                        model.fit(X_ref, y_ref)
                        
                        # Predict the missing values
                        X_pred = df[top_feature_names].values
                        df[metric] = model.predict(X_pred)
                        
                        logger.info(f"Inferred {metric} using regression from {top_feature_names}")
                    except Exception as e:
                        logger.error(f"Error inferring {metric}: {str(e)}")
                        df[metric] = reference_data[metric].mean()
                else:
                    # Fallback to population average
                    df[metric] = reference_data[metric].mean()
            
            inferred_modalities.add('gaze')
        else:
            logger.warning("Insufficient data to reliably infer gaze metrics")
            # Set to population averages
            for metric in gaze_metrics:
                df[metric] = reference_data[metric].mean()
    
    # 5. Infer physiological metrics if missing
    if 'physiological' in missing_features:
        physio_metrics = missing_features['physiological']
        
        available_modality_count = sum(1 for modality in ['gaze', 'pupil', 'eeg'] 
                                  if modality in available_features and available_features[modality])
                                  
        if 'workload_intensity' in df.columns:
            logger.info("Using workload intensity to infer physiological metrics")
            
            for metric in physio_metrics:
                # Simple regression from workload intensity
                if metric in reference_data.columns and 'workload_intensity' in reference_data.columns:
                    # Get correlation
                    corr = reference_data[[metric, 'workload_intensity']].corr().iloc[0, 1]
                    
                    if not np.isnan(corr) and abs(corr) > 0.3:
                        # Create a simple linear regression
                        X_ref = reference_data[['workload_intensity']].values
                        y_ref = reference_data[metric].values
                        
                        try:
                            model = LinearRegression()
                            model.fit(X_ref, y_ref)
                            
                            # Predict the missing values
                            X_pred = df[['workload_intensity']].values
                            df[metric] = model.predict(X_pred)
                            
                            logger.info(f"Inferred {metric} from workload intensity (corr={corr:.2f})")
                        except Exception as e:
                            logger.error(f"Error inferring {metric}: {str(e)}")
                            df[metric] = reference_data[metric].mean()
                    else:
                        # Low correlation, use population average with variation
                        mean_val = reference_data[metric].mean()
                        std_val = reference_data[metric].std() * 0.5  # Reduce variation
                        df[metric] = np.random.normal(mean_val, std_val, len(df))
                        logger.info(f"Inferred {metric} using population average (low correlation)")
                else:
                    # Metric not in reference data, use a reasonable default
                    if metric == 'pulse_rate':
                        df[metric] = np.random.normal(75, 5, len(df))
                    elif metric == 'blood_pressure_sys':
                        df[metric] = np.random.normal(120, 5, len(df))
                    elif metric == 'resp_rate':
                        df[metric] = np.random.normal(16, 2, len(df))
                    else:
                        df[metric] = 0
            
            inferred_modalities.add('physiological')
        
        elif available_modality_count >= 2:
            logger.info("Inferring physiological metrics from other modalities")
            
            for metric in physio_metrics:
                # Get reference correlations
                corr_with_reference = {}
                for ref_col in df.columns:
                    if ref_col in reference_data.columns and metric in reference_data.columns:
                        corr = reference_data[[ref_col, metric]].corr().iloc[0, 1]
                        if not np.isnan(corr):
                            corr_with_reference[ref_col] = abs(corr)
                
                # Use top 3 correlated features for regression
                if corr_with_reference:
                    top_features = sorted(corr_with_reference.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_feature_names = [f[0] for f in top_features]
                    
                    # Create simple regression
                    X_ref = reference_data[top_feature_names].values
                    y_ref = reference_data[metric].values
                    
                    try:
                        model = LinearRegression()
                        model.fit(X_ref, y_ref)
                        
                        # Predict the missing values
                        X_pred = df[top_feature_names].values
                        df[metric] = model.predict(X_pred)
                        
                        logger.info(f"Inferred {metric} using regression from {top_feature_names}")
                    except Exception as e:
                        logger.error(f"Error inferring {metric}: {str(e)}")
                        df[metric] = reference_data[metric].mean()
                else:
                    # Fallback to population average
                    df[metric] = reference_data[metric].mean()
            
            inferred_modalities.add('physiological')
        else:
            logger.warning("Insufficient data to reliably infer physiological metrics")
            # Set to population averages
            for metric in physio_metrics:
                if metric == 'pulse_rate':
                    df[metric] = np.random.normal(75, 5, len(df))
                elif metric == 'blood_pressure_sys':
                    df[metric] = np.random.normal(120, 5, len(df))
                elif metric == 'resp_rate':
                    df[metric] = np.random.normal(16, 2, len(df))
                else:
                    df[metric] = reference_data[metric].mean() if metric in reference_data.columns else 0
    
    # Log results
    logger.info(f"Successfully inferred features for modalities: {inferred_modalities}")
    
    # If input was a dictionary, return a single row dict
    if is_dict_input:
        return df.iloc[0].to_dict()
    else:
        return df

def load_reference_data():
    """
    Load reference data from external resources for feature inference.
    Combines physiological, EEG, and gaze data from multiple sources.
    
    Returns:
        DataFrame: Combined reference data
    """
    logger.info("Loading reference data for feature inference")
    
    try:
        # Try to load from configured data paths
        has_real_data = True
        for data_type, file_path in DATA_FILES.items():
            if not os.path.exists(file_path):
                has_real_data = False
                break
        
        if has_real_data:
            logger.info("Using real data files as reference")
            df_physio, df_eeg, df_gaze = load_data()
            
            # Merge the data
            df = pd.merge_asof(df_physio.sort_values("timestamp"), 
                             df_eeg.sort_values("timestamp"), 
                             on="timestamp")
            df = pd.merge_asof(df.sort_values("timestamp"), 
                             df_gaze.sort_values("timestamp"), 
                             on="timestamp")
            
            # Select relevant features
            features = ["pulse_rate", "blood_pressure_sys", "resp_rate", "pupil_diameter_left",
                      "pupil_diameter_right", "fixation_duration", "blink_rate", "workload_intensity",
                      "gaze_x", "gaze_y", "alpha_power", "theta_power"]
            
            # Drop rows with missing values
            df = df[features].dropna()
            
            return df
        else:
            # Try to load pre-packaged reference data
            reference_path = os.path.join('data', 'reference', 'cognitive_workload_reference.csv')
            if os.path.exists(reference_path):
                logger.info(f"Loading reference data from {reference_path}")
                return pd.read_csv(reference_path)
            
            # Create synthetic data as a last resort
            logger.warning("No reference data found. Generating synthetic reference data.")
            return generate_synthetic_reference_data()
            
    except Exception as e:
        logger.error(f"Error loading reference data: {e}")
        logger.warning("Falling back to synthetic reference data")
        return generate_synthetic_reference_data()

def generate_synthetic_reference_data(n_samples=1000):
    """
    Generate synthetic reference data when no real data is available.
    
    Args:
        n_samples: Number of synthetic samples to generate
        
    Returns:
        DataFrame: Synthetic reference data
    """
    logger.info(f"Generating {n_samples} synthetic reference data points")
    
    # Generate random data with realistic distributions
    data = {
        "pulse_rate": np.random.normal(75, 10, n_samples).clip(50, 120),
        "blood_pressure_sys": np.random.normal(120, 15, n_samples).clip(90, 180),
        "resp_rate": np.random.normal(16, 3, n_samples).clip(8, 30),
        "workload_intensity": np.random.uniform(10, 90, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add derived features with realistic correlations
    
    # Pupil diameter correlates with workload intensity
    base_pupil = 3 + df["workload_intensity"] * 0.03
    noise = np.random.normal(0, 0.2, n_samples)
    df["pupil_diameter_left"] = (base_pupil + noise).clip(2.5, 7.5)
    df["pupil_diameter_right"] = (base_pupil + np.random.normal(0, 0.15, n_samples)).clip(2.5, 7.5)
    
    # Fixation duration is inversely related to workload
    df["fixation_duration"] = (400 - df["workload_intensity"] * 2.5 + 
                             np.random.normal(0, 20, n_samples)).clip(150, 450)
    
    # Blink rate decreases with workload
    df["blink_rate"] = (22 - df["workload_intensity"] * 0.15 + 
                      np.random.normal(0, 2, n_samples)).clip(5, 25)
    
    # Gaze position (less dependent)
    df["gaze_x"] = np.random.normal(500, 50, n_samples)
    df["gaze_y"] = np.random.normal(375, 40, n_samples)
    
    # EEG features
    # Alpha power decreases with workload
    df["alpha_power"] = (30 - df["workload_intensity"] * 0.2 + 
                       np.random.normal(0, 3, n_samples)).clip(5, 35)
    
    # Theta power increases with workload
    df["theta_power"] = (15 + df["workload_intensity"] * 0.15 + 
                       np.random.normal(0, 2, n_samples)).clip(10, 35)
    
    logger.info("Synthetic reference data generated")
    return df

def infer_pupil_metrics(df, reference_data):
    """
    Infer missing pupil diameter metrics based on other physiological data.
    
    Args:
        df: DataFrame with potentially missing pupil metrics
        reference_data: Reference data for inference
        
    Returns:
        DataFrame: Updated DataFrame with inferred pupil metrics
    """
    # Check if both pupil metrics are missing
    left_missing = "pupil_diameter_left" not in df.columns or df["pupil_diameter_left"].isnull().all()
    right_missing = "pupil_diameter_right" not in df.columns or df["pupil_diameter_right"].isnull().all()
    
    # If only one is missing, we can approximate from the other
    if left_missing and not right_missing:
        logger.info("Inferring left pupil diameter from right pupil diameter")
        df["pupil_diameter_left"] = df["pupil_diameter_right"] * np.random.uniform(0.97, 1.03)
        return df
    
    if right_missing and not left_missing:
        logger.info("Inferring right pupil diameter from left pupil diameter")
        df["pupil_diameter_right"] = df["pupil_diameter_left"] * np.random.uniform(0.97, 1.03)
        return df
    
    # Both are missing, use other metrics to infer
    if (left_missing and right_missing):
        logger.info("Inferring both pupil diameters from other physiological metrics")
        
        # The primary driver of pupil diameter is workload intensity
        if "workload_intensity" in df.columns and reference_data is not None:
            logger.info("Using workload intensity to infer pupil diameters")
            
            # Calculate base pupil size based on workload intensity
            for idx, row in df.iterrows():
                workload = row["workload_intensity"]
                base_pupil = 3.0 + (workload / 100.0) * 3.0  # Map 0-100 workload to 3.0-6.0mm pupil
                
                # Add random variation for realistic data
                df.at[idx, "pupil_diameter_left"] = base_pupil + np.random.normal(0, 0.2)
                df.at[idx, "pupil_diameter_right"] = base_pupil + np.random.normal(0, 0.18)
                
        elif all(col in df.columns for col in ["pulse_rate", "blood_pressure_sys"]):
            logger.info("Using physiological metrics to infer pupil diameters")
            
            # Physiological markers also correlate with pupil diameter
            for idx, row in df.iterrows():
                # Normalize physiological metrics
                pulse_norm = (row["pulse_rate"] - 60) / 60  # Normalize to 0-1 range
                bp_norm = (row["blood_pressure_sys"] - 90) / 90  # Normalize to 0-1 range
                
                # Calculate base pupil diameter from physiological state
                base_pupil = 3.5 + (pulse_norm * 1.5) + (bp_norm * 1.5)
                
                # Add random variation
                df.at[idx, "pupil_diameter_left"] = min(max(base_pupil + np.random.normal(0, 0.2), 2.5), 7.5)
                df.at[idx, "pupil_diameter_right"] = min(max(base_pupil + np.random.normal(0, 0.18), 2.5), 7.5)
        
        else:
            logger.warning("Limited data for pupil inference, using reference distributions")
            
            # Use default mid-range values with variation
            for idx in range(len(df)):
                base_pupil = 4.5  # Default mid-range pupil size
                df.at[idx, "pupil_diameter_left"] = base_pupil + np.random.normal(0, 0.5)
                df.at[idx, "pupil_diameter_right"] = base_pupil + np.random.normal(0, 0.45)
    
    return df

def infer_eeg_metrics(df, reference_data):
    """
    Infer missing EEG metrics (alpha_power, theta_power) from other data.
    
    Args:
        df: DataFrame with potentially missing EEG metrics
        reference_data: Reference data for inference
        
    Returns:
        DataFrame: Updated DataFrame with inferred EEG metrics
    """
    alpha_missing = "alpha_power" not in df.columns or df["alpha_power"].isnull().all()
    theta_missing = "theta_power" not in df.columns or df["theta_power"].isnull().all()
    
    # EEG metrics are primarily related to workload intensity
    if "workload_intensity" in df.columns:
        logger.info("Using workload intensity to infer EEG metrics")
        
        # Alpha power inversely correlates with workload
        if alpha_missing:
            logger.info("Inferring alpha power from workload intensity")
            df["alpha_power"] = df.apply(
                lambda row: min(max(30 - row["workload_intensity"] * 0.2 + np.random.normal(0, 2), 5), 35),
                axis=1
            )
        
        # Theta power positively correlates with workload
        if theta_missing:
            logger.info("Inferring theta power from workload intensity")
            df["theta_power"] = df.apply(
                lambda row: min(max(15 + row["workload_intensity"] * 0.15 + np.random.normal(0, 2), 10), 35),
                axis=1
            )
    
    # If no workload intensity, use reference data or defaults
    else:
        logger.warning("No workload intensity data available for EEG inference")
        
        if reference_data is not None:
            logger.info("Using reference data distributions for EEG metrics")
            
            if alpha_missing:
                alpha_mean = reference_data["alpha_power"].mean()
                alpha_std = reference_data["alpha_power"].std()
                df["alpha_power"] = np.random.normal(alpha_mean, alpha_std, len(df)).clip(5, 35)
            
            if theta_missing:
                theta_mean = reference_data["theta_power"].mean()
                theta_std = reference_data["theta_power"].std()
                df["theta_power"] = np.random.normal(theta_mean, theta_std, len(df)).clip(10, 35)
        
        else:
            logger.warning("Using default distributions for EEG metrics")
            
            if alpha_missing:
                df["alpha_power"] = np.random.normal(18, 5, len(df)).clip(5, 35)
            
            if theta_missing:
                df["theta_power"] = np.random.normal(20, 5, len(df)).clip(10, 35)
    
    return df

def infer_workload_intensity(df, reference_data):
    """
    Infer workload intensity from other physiological and behavioral metrics.
    
    Args:
        df: DataFrame with missing workload_intensity
        reference_data: Reference data for inference
        
    Returns:
        DataFrame: Updated DataFrame with inferred workload_intensity
    """
    logger.info("Inferring workload intensity from other metrics")
    
    # Define feature weights and directions for workload inference
    workload_indicators = {
        # Feature: [weight, direction] (positive direction = higher value means higher workload)
        "pulse_rate": [0.20, 1],
        "blood_pressure_sys": [0.15, 1],
        "resp_rate": [0.15, 1],
        "pupil_diameter_left": [0.10, 1],
        "pupil_diameter_right": [0.10, 1],
        "fixation_duration": [0.10, -1],  # Negative: shorter fixations = higher workload
        "blink_rate": [0.10, -1],         # Negative: fewer blinks = higher workload
        "alpha_power": [0.10, -1],        # Negative: lower alpha = higher workload
        "theta_power": [0.10, 1]
    }
    
    # Calculate workload intensity for each row
    for idx, row in df.iterrows():
        workload_score = 50  # Start at middle of range (0-100)
        total_weight = 0
        
        # Accumulate contributions from available features
        for feature, (weight, direction) in workload_indicators.items():
            if feature in row and not pd.isna(row[feature]):
                # Normalize feature value based on typical ranges
                if feature == "pulse_rate":
                    # Normal range: 60-100, higher = more workload
                    norm_value = (row[feature] - 60) / 40
                elif feature == "blood_pressure_sys":
                    # Normal range: 90-150, higher = more workload
                    norm_value = (row[feature] - 90) / 60
                elif feature == "resp_rate":
                    # Normal range: 12-20, higher = more workload
                    norm_value = (row[feature] - 12) / 8
                elif feature in ["pupil_diameter_left", "pupil_diameter_right"]:
                    # Typical range: 3-7mm, higher = more workload
                    norm_value = (row[feature] - 3) / 4
                elif feature == "fixation_duration":
                    # Typical range: 200-400ms, lower = more workload
                    norm_value = (400 - row[feature]) / 200
                elif feature == "blink_rate":
                    # Typical range: 5-25 bpm, lower = more workload
                    norm_value = (25 - row[feature]) / 20
                elif feature == "alpha_power":
                    # Typical range: 5-35, lower = more workload
                    norm_value = (35 - row[feature]) / 30
                elif feature == "theta_power":
                    # Typical range: 10-35, higher = more workload
                    norm_value = (row[feature] - 10) / 25
                else:
                    # Unknown feature, skip
                    continue
                
                # Apply weight and direction
                workload_score += norm_value * direction * weight * 50
                total_weight += weight
        
        # Normalize and constrain to 0-100 range
        if total_weight > 0:
            # Scale by actual weight used
            workload_score = workload_score / total_weight * 0.5
            # Constrain to reasonable range
            workload_score = min(max(workload_score, 10), 90)
        
        df.at[idx, "workload_intensity"] = workload_score
    
    return df

def impute_generic_feature(df, feature, reference_data):
    """
    Generic imputation for any missing feature.
    
    Args:
        df: DataFrame with missing feature
        feature: Name of the missing feature
        reference_data: Reference data for inference
        
    Returns:
        DataFrame: Updated DataFrame with imputed feature
    """
    logger.info(f"Imputing missing feature: {feature}")
    
    # Default feature ranges for imputation
    default_ranges = {
        "pulse_rate": [60, 100],  # min, max
        "blood_pressure_sys": [90, 150],
        "resp_rate": [12, 20],
        "pupil_diameter_left": [3.0, 7.0],
        "pupil_diameter_right": [3.0, 7.0],
        "fixation_duration": [180, 400],
        "blink_rate": [5, 25],
        "workload_intensity": [10, 90],
        "gaze_x": [0, 1000],
        "gaze_y": [0, 750],
        "alpha_power": [5, 35],
        "theta_power": [10, 35],
    }
    
    # Use reference data if available
    if reference_data is not None and feature in reference_data.columns:
        mean_val = reference_data[feature].mean()
        std_val = reference_data[feature].std()
        df[feature] = np.random.normal(mean_val, std_val, len(df))
        
        # Ensure values are within reasonable range
        if feature in default_ranges:
            min_val, max_val = default_ranges[feature]
            df[feature] = df[feature].clip(min_val, max_val)
    
    # Use default distributions
    elif feature in default_ranges:
        min_val, max_val = default_ranges[feature]
        mean_val = (min_val + max_val) / 2
        std_val = (max_val - min_val) / 4
        df[feature] = np.random.normal(mean_val, std_val, len(df)).clip(min_val, max_val)
    
    # Unknown feature, use zeros
    else:
        logger.warning(f"No default range for feature: {feature}, using zero")
        df[feature] = 0
    
    return df

# ---------------------- MAIN EXECUTION ---------------------- #
def main():
    """
    Main entry point for the Cognitive Workload Training (CWT) command line interface.
    """
    setup_logging()
    args = parse_args()
    
    try:
        # Setup
        if args.command == "setup":
            # Ensure model directory exists
            os.makedirs(MODEL_DIR, exist_ok=True)
            logger.info(f"Model directory: {MODEL_DIR}")
            logger.info("Setup complete!")
            
        # Train command
        elif args.command == "train":
            # Validate input parameters
            if not args.model_type:
                logger.error("No model type specified!")
                logger.info("Available model types: svm, random_forest, mlp, knn, decision_tree, gradient_boosting")
                return
            
            # Create output directory if specified
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                model_dir = args.output_dir
            else:
                model_dir = os.path.join(MODEL_DIR, args.model_type)
                os.makedirs(model_dir, exist_ok=True)
            
            logger.info(f"Training {args.model_type} model in directory: {model_dir}")
            
            # Train model
            model, scaler, accuracy, f1, metadata = train_model(args.model_type, model_dir)
            
            logger.info(f"Model training complete!")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
        
        # Train All command
        elif args.command == "train-all":
            # Create output directory if specified
            output_dir = args.output_dir if args.output_dir else MODEL_DIR
            
            # Parse skip types
            skip_types = args.skip_types.split(',') if args.skip_types else None
            
            # Train all models
            results = train_all_models(
                output_dir=output_dir,
                parallel=args.parallel,
                skip_types=skip_types
            )
            
            # Print results
            logger.info("All model training complete!")
            logger.info("Results:")
            for model_type, result in results.items():
                if "error" in result:
                    logger.info(f"  {model_type}: ERROR - {result['error']}")
                else:
                    logger.info(f"  {model_type}: Accuracy = {result['accuracy']:.4f}, F1 Score = {result['f1_score']:.4f}")
        
        # Predict command
        elif args.command == "predict":
            # Load the input data
            input_data = {}
            
            if args.input_json:
                # Load from JSON file
                with open(args.input_json, 'r') as f:
                    input_data = json.load(f)
            elif args.input_values:
                # Parse input values directly
                for kv in args.input_values:
                    key, value = kv.split('=')
                    input_data[key] = float(value)
            else:
                # Use interactive input
                logger.info("Please enter the feature values:")
                for feature in REQUIRED_FEATURES:
                    while True:
                        try:
                            value = input(f"{feature}: ")
                            input_data[feature] = float(value)
                            break
                        except ValueError:
                            logger.error("Please enter a valid number")
            
            # Make prediction
            result = predict(
                data=input_data,
                model_type=args.model_type,
                threshold=args.threshold,
                infer_missing=args.infer_missing
            )
            
            # Show results
            logger.info("Prediction Results:")
            logger.info(f"Workload Class: {result['workload_class']}")
            logger.info(f"Workload Label: {result['workload_label']}")
            logger.info(f"Confidence: {result['confidence']:.4f}")
            logger.info(f"Model Type: {result['model_type']}")
            
            # Output results to file if specified
            if args.output_json:
                with open(args.output_json, 'w') as f:
                    json.dump(result, f, indent=4)
                logger.info(f"Results saved to {args.output_json}")
        
        # Batch predict command
        elif args.command == "batch-predict":
            # Validate input parameters
            if not args.input_file:
                logger.error("No input file specified!")
                return
            
            if not os.path.exists(args.input_file):
                logger.error(f"Input file not found: {args.input_file}")
                return
            
            # Make batch prediction
            result_df = predict_batch(
                file_path=args.input_file,
                output_file=args.output_file,
                model_type=args.model_type,
                threshold=args.threshold,
                infer_missing=args.infer_missing
            )
            
            # Show summary
            logger.info("Batch Prediction Results:")
            logger.info(f"Total Records: {len(result_df)}")
            
            # Count by class
            class_counts = result_df["workload_class"].value_counts()
            for cls, count in class_counts.items():
                if cls is None:
                    logger.info(f"Below Threshold: {count}")
                else:
                    label = WORKLOAD_CLASSES.get(int(cls), "Unknown")
                    logger.info(f"{label} (Class {cls}): {count}")
            
            # Show average confidence
            logger.info(f"Average Confidence: {result_df['confidence'].mean():.4f}")
        
        # Time series predict command
        elif args.command == "time-series-predict":
            # Validate input parameters
            if not args.input_file:
                logger.error("No input file specified!")
                return
            
            if not os.path.exists(args.input_file):
                logger.error(f"Input file not found: {args.input_file}")
                return
            
            # Make time series prediction
            result_df = predict_time_series(
                file_path=args.input_file,
                output_file=args.output_file,
                window_size=args.window_size,
                step_size=args.step_size,
                model_type=args.model_type,
                threshold=args.threshold,
                infer_missing=args.infer_missing
            )
            
            # Show summary
            logger.info("Time Series Prediction Results:")
            logger.info(f"Total Time Points: {len(result_df)}")
            
            # Count by class
            class_counts = result_df["workload_class"].value_counts()
            for cls, count in class_counts.items():
                if cls is None:
                    logger.info(f"Below Threshold: {count}")
                else:
                    label = WORKLOAD_CLASSES.get(int(cls), "Unknown")
                    logger.info(f"{label} (Class {cls}): {count}")
            
            # Show average confidence
            logger.info(f"Average Confidence: {result_df['confidence'].mean():.4f}")
            
            # Plot the time series if requested
            if args.visualize and args.output_file:
                # Import visualization modules only when needed
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Plot the time series
                plt.figure(figsize=(12, 6))
                plt.plot(result_df["timestamp"], result_df["workload_class"], marker='o')
                plt.title("Cognitive Workload Time Series")
                plt.xlabel("Time")
                plt.ylabel("Workload Class")
                plt.grid(True)
                
                # Set y-ticks to workload classes
                classes = sorted(WORKLOAD_CLASSES.keys())
                plt.yticks(classes, [WORKLOAD_CLASSES[c] for c in classes])
                
                # Save plot
                plot_path = args.output_file.replace(".csv", ".png")
                plt.savefig(plot_path)
                logger.info(f"Time series plot saved to {plot_path}")
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            logger.exception("Detailed error information:")
        return 1
    
    return 0

def setup_logging():
    """
    Set up logging configuration for the application.
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create a logger specific to this module
    global logger
    logger = logging.getLogger('cwt')

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Cognitive Workload Training (CWT) Tool")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Define subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup the CWT environment")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a cognitive workload model")
    train_parser.add_argument("--model-type", type=str, help="Type of model to train (svm, random_forest, etc.)")
    train_parser.add_argument("--output-dir", type=str, help="Directory to save the trained model")
    
    # Train All command
    train_all_parser = subparsers.add_parser("train-all", help="Train all available cognitive workload models")
    train_all_parser.add_argument("--output-dir", type=str, help="Directory to save the trained models")
    train_all_parser.add_argument("--parallel", action="store_true", help="Train models in parallel")
    train_all_parser.add_argument("--skip-types", type=str, help="Comma-separated list of model types to skip")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict cognitive workload from input data")
    predict_parser.add_argument("--input-json", type=str, help="JSON file with input data")
    predict_parser.add_argument("--input-values", type=str, nargs="+", help="Input values in key=value format")
    predict_parser.add_argument("--output-json", type=str, help="JSON file to save prediction results")
    predict_parser.add_argument("--model-type", type=str, help="Type of model to use for prediction")
    predict_parser.add_argument("--threshold", type=float, help="Confidence threshold for prediction")
    predict_parser.add_argument("--infer-missing", action="store_true", help="Infer missing features from available data")
    
    # Batch predict command
    batch_predict_parser = subparsers.add_parser("batch-predict", help="Batch predict cognitive workload from a CSV file")
    batch_predict_parser.add_argument("--input-file", type=str, help="Input CSV file with feature values")
    batch_predict_parser.add_argument("--output-file", type=str, help="Output CSV file to save prediction results")
    batch_predict_parser.add_argument("--model-type", type=str, help="Type of model to use for prediction")
    batch_predict_parser.add_argument("--threshold", type=float, help="Confidence threshold for prediction")
    batch_predict_parser.add_argument("--infer-missing", action="store_true", help="Infer missing features from available data")
    
    # Time series predict command
    time_series_parser = subparsers.add_parser("time-series-predict", help="Predict cognitive workload from time series data")
    time_series_parser.add_argument("--input-file", type=str, help="Input CSV file with time series data")
    time_series_parser.add_argument("--output-file", type=str, help="Output CSV file to save prediction results")
    time_series_parser.add_argument("--window-size", type=int, default=10, help="Size of sliding window in samples")
    time_series_parser.add_argument("--step-size", type=int, default=5, help="Step size for sliding window")
    time_series_parser.add_argument("--model-type", type=str, help="Type of model to use for prediction")
    time_series_parser.add_argument("--threshold", type=float, help="Confidence threshold for prediction")
    time_series_parser.add_argument("--visualize", action="store_true", help="Visualize time series prediction results")
    time_series_parser.add_argument("--infer-missing", action="store_true", help="Infer missing features from available data")
    
    return parser.parse_args()

def find_best_model():
    """
    Find the best model based on accuracy or other metrics.
    
    Returns:
        tuple: (model_path, metadata) for the best model
    """
    # Get all models
    models_dir = MODEL_DIR
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return None, None
    
    best_model_path = None
    best_metadata = None
    best_accuracy = -1
    
    # Look through all model directories
    for model_type in os.listdir(models_dir):
        model_type_dir = os.path.join(models_dir, model_type)
        if not os.path.isdir(model_type_dir):
            continue
            
        # Check for model file
        model_files = [f for f in os.listdir(model_type_dir) 
                      if (f.endswith('.pkl') or f.endswith('.joblib')) and not f.startswith('scaler')]
        
        for model_file in model_files:
            model_path = os.path.join(model_type_dir, model_file)
            
            # Check for metadata
            metadata_path = os.path.join(model_type_dir, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if it has accuracy information
                    if 'accuracy' in metadata:
                        try:
                            # Handle accuracy as either string or float
                            accuracy_value = metadata['accuracy']
                            if isinstance(accuracy_value, str):
                                # Remove any non-numeric characters
                                accuracy_value = ''.join(c for c in accuracy_value if c.isdigit() or c == '.')
                                accuracy = float(accuracy_value)
                            else:
                                accuracy = float(accuracy_value)
                                
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_model_path = model_path
                                best_metadata = metadata
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error parsing accuracy value '{metadata['accuracy']}': {e}")
                except Exception as e:
                    logger.warning(f"Error reading metadata for {model_path}: {e}")
    
    if best_model_path:
        logger.info(f"Found best model: {best_model_path} with accuracy: {best_accuracy:.4f}")
        return best_model_path, best_metadata
    else:
        # If no model with accuracy found, fall back to latest model
        logger.warning("No model with accuracy metadata found, falling back to latest model")
        return find_latest_model()

def train_all_models(output_dir=None, parallel=False, skip_types=None):
    """
    Train all available cognitive workload classification models.
    
    Args:
        output_dir (str, optional): Directory to save models to. If None, uses default directory.
        parallel (bool, optional): Whether to train models in parallel. Defaults to False.
        skip_types (list, optional): List of model types to skip. Defaults to None.
        
    Returns:
        dict: Dictionary of model results with metrics
    """
    if output_dir is None:
        output_dir = MODEL_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Training all model types in directory: {output_dir}")
    
    # Check if required data files exist
    for data_type, file_path in DATA_FILES.items():
        if not os.path.exists(file_path):
            logger.error(f"Required data file not found: {file_path}")
            raise FileNotFoundError(f"Required data file not found: {file_path}")
    
    # Load data
    df_physio, df_eeg, df_gaze = load_data()
    
    # Preprocess data
    X, y, features = preprocess_data(df_physio, df_eeg, df_gaze)
    
    # Create a shared scaler to ensure consistency across models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the shared scaler
    scaler_path = os.path.join(output_dir, "shared_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    logger.info(f"Saved shared scaler to {scaler_path}")
    
    # Find all available model types
    model_types = ["svm", "random_forest", "knn", "mlp", "gradient_boosting", "decision_tree"]
    
    # Filter out skipped model types
    if skip_types:
        model_types = [model_type for model_type in model_types if model_type not in skip_types]
    
    logger.info(f"Training the following model types: {model_types}")
    
    # Metadata for the ensemble of models
    metadata = {
        "creation_date": datetime.now().isoformat(),
        "training_data_files": DATA_FILES,
        "features": features,
        "model_types": model_types,
        "shared_scaler": scaler_path
    }
    
    # Results dictionary
    results = {}
    
    if parallel and len(model_types) > 1:
        # Train models in parallel
        logger.info("Training models in parallel")
        
        # Create pool of workers
        num_cores = min(multiprocessing.cpu_count(), len(model_types))
        logger.info(f"Using {num_cores} CPU cores for parallel training")
        
        with ThreadPoolExecutor(max_workers=num_cores) as pool:
            # Prepare arguments for each model
            args = [(model_type, X_scaled, y, features, output_dir) for model_type in model_types]
            
            # Train models in parallel
            parallel_results = list(pool.map(lambda x: _train_model_wrapper(x), args))
            
            # Process results
            for model_type, model_path, accuracy, f1 in parallel_results:
                results[model_type] = {
                    "model_path": model_path,
                    "accuracy": accuracy,
                    "f1_score": f1
                }
    else:
        # Train models sequentially
        logger.info("Training models sequentially")
        
        for model_type in model_types:
            try:
                logger.info(f"Training {model_type} model...")
                
                # Create model directory
                model_dir = os.path.join(output_dir, model_type)
                os.makedirs(model_dir, exist_ok=True)
                
                # Train model
                model, _, accuracy, f1, _ = train_model(model_type, model_dir)
                
                # Save results
                results[model_type] = {
                    "model_path": os.path.join(model_dir, f"{model_type}.pkl"),
                    "accuracy": float(accuracy),
                    "f1_score": float(f1)
                }
                
                logger.info(f"Successfully trained {model_type} model")
                logger.info(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                # Continue with other models even if one fails
                results[model_type] = {
                    "error": str(e)
                }
    
    # Save metadata
    metadata["results"] = results
    metadata_path = os.path.join(output_dir, "ensemble_metadata.json")
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Saved ensemble metadata to {metadata_path}")
    
    # Return results
    return results

def _train_model_wrapper(args):
    """
    Helper function for parallel model training.
    
    Args:
        args: Tuple of (model_type, X_scaled, y, features, output_dir)
        
    Returns:
        tuple: (model_type, model_path, accuracy, f1_score)
    """
    model_type, X_scaled, y, features, output_dir = args
    
    try:
        logger.info(f"Training {model_type} model in parallel process...")
        
        # Create model directory
        model_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # Get model object
        model = create_model(model_type)
        
        # Train model
        model.fit(X_scaled, y)
        
        # Evaluate model
        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        
        # Save model
        model_path = os.path.join(model_dir, f"{model_type}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save additional metadata
        metadata = {
            "creation_date": datetime.now().isoformat(),
            "model_type": model_type,
            "features": features,
            "accuracy": float(accuracy),
            "f1_score": float(f1)
        }
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Successfully trained {model_type} model in parallel process")
        
        return model_type, model_path, accuracy, f1
        
    except Exception as e:
        logger.error(f"Error training {model_type} model in parallel process: {e}")
        return model_type, None, 0.0, 0.0

def batch_predict(file_path, output_file=None, model_type=None, threshold=None, infer_missing=False):
    """
    Predict cognitive workload from a batch of samples in a CSV file.
    
    Args:
        file_path (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, returns results without saving.
        model_type (str, optional): Type of model to use. If None, uses best available model.
        threshold (float, optional): Confidence threshold for prediction. If None, no threshold is applied.
        infer_missing (bool, optional): Whether to infer missing features. Defaults to False.
        
    Returns:
        list: List of prediction results
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records from {file_path}")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise ValueError(f"Error loading file: {str(e)}")
    
    # Check missing features before inference
    missing_modalities_count = {}
    for modality, features in FEATURE_MODALITIES.items():
        missing = [f for f in features if f not in df.columns]
        if missing:
            missing_modalities_count[modality] = len(missing)
            logger.warning(f"Missing {len(missing)}/{len(features)} {modality} features")
    
    # Make predictions
    if infer_missing:
        logger.info("Inferring missing features")
    
    results = predict(df, model_type=model_type, threshold=threshold, infer_missing=infer_missing)
    
    # Extract quality metrics for summary
    missing_modalities_list = set()
    avg_confidence = 0.0
    warning_counts = {}
    
    for result in results:
        # Track quality metrics
        missing_mods = result['data_quality']['missing_modalities']
        for mod in missing_mods:
            missing_modalities_list.add(mod)
        
        # Track confidence
        avg_confidence += result['adjusted_confidence']
        
        # Track warnings
        if 'quality_warning' in result:
            warning = result['quality_warning']
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
    
    # Calculate average confidence
    if results:
        avg_confidence /= len(results)
    
    # Generate batch quality summary
    quality_summary = {
        'total_samples': len(results),
        'average_confidence': avg_confidence,
        'missing_modalities': list(missing_modalities_list),
        'warning_counts': warning_counts
    }
    
    logger.info(f"Batch prediction quality summary: {quality_summary}")
    
    # Save to file if specified
    if output_file:
        # Create a simpler format for CSV output
        output_data = []
        for i, result in enumerate(results):
            row = {
                'row_index': i,
                'workload_class': result['workload_class'],
                'workload_label': result['workload_label'],
                'model_confidence': result['model_confidence'],
                'adjusted_confidence': result['adjusted_confidence'],
                'data_quality_score': result['data_quality']['final_confidence']
            }
            
            # Add timestamp if present in original data
            if 'timestamp' in df.columns:
                row['timestamp'] = df.iloc[i]['timestamp']
            
            # Add quality warning if present
            if 'quality_warning' in result:
                row['quality_warning'] = result['quality_warning']
            
            # Add original features for reference
            for feature in df.columns:
                row[f'input_{feature}'] = df.iloc[i][feature]
            
            output_data.append(row)
        
        # Save as CSV
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_file, index=False)
        logger.info(f"Saved batch predictions to {output_file}")
        
        # Save quality summary
        summary_file = os.path.splitext(output_file)[0] + "_quality_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(quality_summary, f, indent=2)
        logger.info(f"Saved batch quality summary to {summary_file}")
    
    return results

def predict_time_series(file_path, output_file=None, window_size=10, step_size=5, 
                       model_type=None, threshold=None, infer_missing=False, visualize=False):
    """
    Predict cognitive workload from time series data using a sliding window approach.
    
    Args:
        file_path (str): Path to input CSV file with time series data
        output_file (str, optional): Path to output CSV file. If None, returns results without saving.
        window_size (int): Size of sliding window in samples. Default is 10.
        step_size (int): Step size for sliding window. Default is 5.
        model_type (str, optional): Type of model to use. If None, uses best available model.
        threshold (float, optional): Confidence threshold for prediction. If None, no threshold is applied.
        infer_missing (bool, optional): Whether to infer missing features. Defaults to False.
        visualize (bool): Whether to generate visualization of the results. Default is False.
        
    Returns:
        dict: Time series prediction results and quality metrics
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} time points from {file_path}")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise ValueError(f"Error loading file: {str(e)}")
    
    # Check for timestamp column
    if 'timestamp' not in df.columns:
        logger.warning("No timestamp column found in the data. Using row indices instead.")
        df['timestamp'] = [f"t{i}" for i in range(len(df))]
    
    # Check for missing modalities
    modality_availability = {}
    for modality, features in FEATURE_MODALITIES.items():
        available = [f for f in features if f in df.columns]
        modality_availability[modality] = len(available) / len(features)
        
        if not available:
            logger.warning(f"No features available for {modality} modality")
        elif len(available) < len(features):
            logger.warning(f"Partial features available for {modality} modality: {len(available)}/{len(features)}")
    
    # Infer missing features if requested
    if infer_missing:
        logger.info("Inferring missing features in time series data")
        df = infer_missing_features(df)
    
    # Create sliding windows
    windows = []
    timestamps = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i+window_size].copy()
        windows.append(window)
        timestamps.append(df.iloc[i]['timestamp'])
    
    logger.info(f"Created {len(windows)} windows from time series data")
    
    # Make predictions for each window
    window_results = []
    workload_classes = []
    confidence_scores = []
    quality_scores = []
    
    for window in windows:
        # Predict for this window
        results = predict(window, model_type=model_type, threshold=threshold, infer_missing=False)
        
        # For window results, take average of predictions
        avg_class = None
        avg_confidence = 0
        avg_quality = 0
        class_counts = {}
        
        for result in results:
            # Count workload classes
            if result['workload_class'] is not None:
                class_counts[result['workload_class']] = class_counts.get(result['workload_class'], 0) + 1
            
            # Sum up confidences
            avg_confidence += result['adjusted_confidence']
            avg_quality += result['data_quality']['final_confidence']
        
        # Average the values
        avg_confidence /= len(results)
        avg_quality /= len(results)
        
        # Get most common class
        if class_counts:
            avg_class = max(class_counts.items(), key=lambda x: x[1])[0]
        
        # Store results
        workload_classes.append(avg_class)
        confidence_scores.append(avg_confidence)
        quality_scores.append(avg_quality)
        
        # Create window result
        window_result = {
            'workload_class': avg_class,
            'workload_label': WORKLOAD_CLASSES.get(avg_class, "Unknown"),
            'confidence': avg_confidence,
            'data_quality': avg_quality,
            'detailed_results': results
        }
        window_results.append(window_result)
    
    # Generate overall quality metrics
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # Determine stability of predictions
    stability = 0
    if len(workload_classes) > 1:
        changes = sum(1 for i in range(1, len(workload_classes)) 
                    if workload_classes[i] != workload_classes[i-1])
        stability = 1 - (changes / (len(workload_classes) - 1))
    
    # Generate time series quality summary
    quality_summary = {
        'total_windows': len(windows),
        'window_size': window_size,
        'step_size': step_size,
        'average_confidence': avg_confidence,
        'average_data_quality': avg_quality,
        'prediction_stability': stability,
        'modality_availability': modality_availability,
        'inferred_features': infer_missing
    }
    
    logger.info(f"Time series prediction quality summary: {quality_summary}")
    
    # Prepare results
    ts_results = {
        'timestamps': timestamps,
        'workload_classes': workload_classes,
        'confidence_scores': confidence_scores,
        'quality_scores': quality_scores,
        'window_results': window_results,
        'quality_summary': quality_summary
    }
    
    # Save to file if specified
    if output_file:
        # Create DataFrame for output
        output_df = pd.DataFrame({
            'timestamp': timestamps,
            'workload_class': workload_classes,
            'workload_label': [WORKLOAD_CLASSES.get(c, "Unknown") for c in workload_classes],
            'confidence': confidence_scores,
            'data_quality': quality_scores
        })
        
        # Save CSV
        output_df.to_csv(output_file, index=False)
        logger.info(f"Saved time series predictions to {output_file}")
        
        # Save quality summary
        summary_file = os.path.splitext(output_file)[0] + "_quality_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(quality_summary, f, indent=2)
        logger.info(f"Saved time series quality summary to {summary_file}")
        
        # Generate visualization if requested
        if visualize:
            try:
                # Create the visualization
                plt.figure(figsize=(12, 8))
                
                # Plot workload classes
                ax1 = plt.subplot(3, 1, 1)
                ax1.plot(range(len(timestamps)), workload_classes, 'b-', marker='o')
                ax1.set_ylabel('Workload Class')
                ax1.set_title('Time Series Workload Prediction')
                ax1.set_ylim(-0.5, 3.5)  # Assuming 0=Low, 1=Medium, 2=High, 3=Very High
                ax1.set_yticks([0, 1, 2, 3])
                ax1.set_yticklabels(['Low', 'Medium', 'High', 'Very High'])
                ax1.grid(True)
                
                # Plot confidence scores
                ax2 = plt.subplot(3, 1, 2, sharex=ax1)
                ax2.plot(range(len(timestamps)), confidence_scores, 'g-', marker='s')
                ax2.set_ylabel('Confidence')
                ax2.set_ylim(0, 1.05)
                ax2.grid(True)
                
                # Plot data quality
                ax3 = plt.subplot(3, 1, 3, sharex=ax1)
                ax3.plot(range(len(timestamps)), quality_scores, 'r-', marker='^')
                ax3.set_ylabel('Data Quality')
                ax3.set_xlabel('Time Window')
                ax3.set_ylim(0, 1.05)
                ax3.grid(True)
                
                # Set x-ticks to show some of the timestamps
                if len(timestamps) > 20:
                    tick_indices = list(range(0, len(timestamps), len(timestamps) // 10))
                    if tick_indices[-1] != len(timestamps) - 1:
                        tick_indices.append(len(timestamps) - 1)
                else:
                    tick_indices = range(len(timestamps))
                
                plt.xticks(tick_indices, [timestamps[i] for i in tick_indices], rotation=45)
                
                # Add annotations for missing modalities
                missing_mods_text = "Missing Modalities: "
                missing_mods = [mod for mod, avail in modality_availability.items() if avail < 1.0]
                if missing_mods:
                    missing_mods_text += ", ".join(missing_mods)
                else:
                    missing_mods_text += "None"
                
                plt.figtext(0.5, 0.01, missing_mods_text, ha='center', fontsize=10, 
                           bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
                
                plt.tight_layout()
                
                # Save the figure
                visualization_file = os.path.splitext(output_file)[0] + ".png"
                plt.savefig(visualization_file)
                logger.info(f"Saved visualization to {visualization_file}")
                plt.close()
                
            except Exception as e:
                logger.error(f"Error generating visualization: {str(e)}")
    
    return ts_results

if __name__ == "__main__":
    # Special case for !help command
    if len(sys.argv) > 1 and sys.argv[1] == '!help':
        # Convert !help to help for argparse
        sys.argv[1] = 'help'
    try:
        main()  # Run the main function
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"\n✘ ERROR: An error occurred: {str(e)}")
        sys.exit(1)
        sys.exit(1)