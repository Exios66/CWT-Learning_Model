# Cognitive Workload Training (CWT) Tool

A comprehensive tool for training machine learning models to detect and classify cognitive workload levels based on physiological and behavioral metrics.

## Features

- Multi-modal data integration (physiological, EEG, and gaze tracking)
- Support for multiple machine learning algorithms:
  - Random Forest
  - Support Vector Machine
  - Gradient Boosting
  - Neural Network (MLP)
  - K-Nearest Neighbors
  - Logistic Regression
- Automatic data preprocessing and feature standardization
- Model persistence and metadata tracking
- Visualization of model performance
- Pre-trained sample models for immediate use

## Recent Enhancements

The CWT Tool has been enhanced with the following new features:

1. **Unified Training Command**: A new `train-all` command allows training all model types in a single operation.
   - Options to train in parallel for faster execution
   - Ability to skip specific model types
   - Shared scaler to ensure consistent scaling across models

2. **Feature Inference Capabilities**: The system can now handle missing features by inferring them from available data.
   - Specialized inference for pupil metrics, EEG signals, and workload intensity
   - Uses physiological correlations from reference data
   - Falls back to synthetic reference data generation when needed

3. **Confidence Thresholds**: Prediction functions now support confidence thresholds to improve reliability.
   - Predictions below the specified threshold are marked as uncertain
   - Enhances the quality of workload predictions

4. **Enhanced Batch Processing**: Improved capabilities for processing files with multiple data points.
   - Better support for CSV files with missing features
   - Time series analysis with sliding window approach
   - Visualization options for time series predictions

5. **New Utilities**:
   - **Command Dictionary**: Interactive help system with detailed information on all commands
   - **Automated Start Script**: One-command setup of the entire environment with model training and analysis preparation

## Quick Start

To get up and running quickly with the CWT Tool, simply run:

```bash
./start.py
```

This will:

1. Set up the CWT environment
2. Train all available model types (with parallel processing)
3. Prepare sample predictions and analyses
4. Generate an overview of model performance

After completion, you'll have trained models ready for use and examples of predictions in the `results` directory.

## Command Reference

For a complete list of available commands and their options, run:

```bash
./commands.py
```

For detailed help on a specific command:

```bash
./commands.py [command]
```

For example: `./commands.py train-all` will show all options for training all models at once.

## Working with Missing Features

The CWT Tool can now handle data with missing features by using the `--infer-missing` flag:

```bash
python cwt.py predict --input-json data/incomplete_sample.json --infer-missing
```

This works by:

1. Detecting which features are missing
2. Loading reference data to understand feature correlations
3. Using specialized algorithms to infer the missing values
4. Applying the standard prediction pipeline with the complete dataset

## Command Reference

### setup

```bash
python cwt.py setup
```

### train

```bash
python cwt.py train [--model-type TYPE] [--output-dir DIR]
```

- `--model-type`: Type of model to train (svm, random_forest, knn, mlp, gradient_boosting, decision_tree)
- `--output-dir`: Directory to save the trained model

### train-all

```bash
python cwt.py train-all [--output-dir DIR] [--parallel] [--skip-types TYPE1,TYPE2]
```

- `--output-dir`: Directory to save trained models (default: models/)
- `--parallel`: Train models in parallel for faster execution
- `--skip-types`: Comma-separated list of model types to skip (e.g., "svm,knn")

### predict

```bash
python cwt.py predict [--input-json FILE] [--input-values KEY=VALUE...] [--model-type TYPE] [--threshold FLOAT] [--infer-missing] [--output-json FILE]
```

### batch-predict

```bash
python cwt.py batch-predict --input-file FILE [--output-file FILE] [--model-type TYPE] [--threshold FLOAT] [--infer-missing]
```

### time-series-predict

```bash
python cwt.py time-series-predict --input-file FILE [--output-file FILE] [--window-size INT] [--step-size INT] [--model-type TYPE] [--threshold FLOAT] [--infer-missing] [--visualize]
```

## Example Script

An example script `example.sh` is provided to demonstrate the key features:

```bash
# Make it executable
chmod +x example.sh

# Run the example
./example.sh
```

## Reference Data

To improve feature inference accuracy, you can provide custom reference data:

```
data/reference/cognitive_workload_reference.csv
```

This file should contain representative samples with all the features used by the models.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/CWT-Learning_Model.git
cd CWT-Learning_Model
```

2. Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Initialize the environment:

```bash
python cwt.py setup
```

## Usage

The CWT provides a command-line interface with several commands:

### Getting Help

```bash
python cwt.py help
# or
./!help
# or for complete command reference
./commands.py
```

### Training a Model

Train a specific model type:

```bash
python cwt.py train --model-type svm
```

Train all available model types:

```bash
python cwt.py train-all --output-dir models/ensemble --parallel
```

### Making Predictions

Predict with explicit input values:

```bash
python cwt.py predict --input-values "pulse_rate=75" "blood_pressure_sys=120" "resp_rate=16"
```

Predict from JSON file:

```bash
python cwt.py predict --input-json data/sample_input.json --model-type mlp
```

Batch predict from CSV with feature inference:

```bash
python cwt.py batch-predict --input-file data/batch_samples.csv --output-file results.csv --infer-missing
```

Time series analysis:

```bash
python cwt.py time-series-predict --input-file data/time_series.csv --window-size 10 --step-size 5 --visualize
```

### Installing Sample Models

```bash
python cwt.py install-models
```

### Downloading Advanced Models

To improve prediction accuracy, you can download advanced pre-trained models:

```bash
# Download all advanced models
./download-models --all

# Download a specific model type
./download-models --model-type gb
```

These advanced models provide several benefits:

- Higher accuracy (up to 0.92 compared to ~0.35 for sample models)
- Trained on larger datasets (up to 35,000 samples)
- More sophisticated feature engineering
- More robustness to missing or noisy data

Use the advanced models for prediction:

```bash
python cwt.py predict --input data/sample_input.json --model models/advanced/gb/Advanced_gb_model.joblib --scaler models/advanced/gb/Advanced_gb_scaler.joblib
```

### Listing Available Models

```bash
python cwt.py list-models
```

## Project Structure

The CWT project follows an organized directory structure:

```bash
CWT-Learning_Model/
├── data/                        # Data files for training and prediction
├── examples/                    # Example files and utilities
│   └── json_samples/            # Example JSON files for testing
├── logs/                        # Log files
│   ├── general/                 # General logs
│   ├── training/                # Training-specific logs
│   ├── prediction/              # Prediction-specific logs
│   └── installation/            # Installation logs
├── models/                      # Trained models
│   ├── sample/                  # Models trained on synthetic data
│   │   ├── default/             # Default models
│   │   ├── rf/                  # Random Forest models
│   │   ├── svm/                 # Support Vector Machine models
│   │   ├── gb/                  # Gradient Boosting models
│   │   ├── mlp/                 # Neural Network models
│   │   ├── knn/                 # K-Nearest Neighbors models
│   │   └── lr/                  # Logistic Regression models
│   ├── advanced/                # Advanced pre-trained models
│   │   ├── rf/                  # Advanced Random Forest models
│   │   ├── svm/                 # Advanced SVM models
│   │   ├── gb/                  # Advanced Gradient Boosting models
│   │   ├── mlp/                 # Advanced Neural Network models
│   │   ├── knn/                 # Advanced KNN models
│   │   └── lr/                  # Advanced Logistic Regression models
│   └── visualizations/          # Model performance visualizations
├── utilities/                   # Helper scripts and utilities
│   ├── check_models.py          # Script to check model and scaler compatibility
│   ├── download_advanced_models.py # Script to download advanced models
│   ├── generate_sample_data.py  # Script to generate sample data
│   ├── organize_outputs.py      # Script to organize models and logs
│   └── test_imports.py          # Script to test Python imports
├── cwt.py                       # Main script
├── commands.py                  # Command dictionary and help system
├── start.py                     # Automated setup and training script
├── requirements.txt             # Python dependencies
├── .env                         # Environment configuration
└── README.md                    # This file
```

## Utility Shortcuts

The CWT comes with several utility scripts that can be executed directly from the root directory:

| Shortcut | Description | Usage |
|----------|-------------|-------|
| `./check-models` | Check model and scaler compatibility | `./check-models` or `./check-models --fix` |
| `./generate-data` | Generate sample data files | `./generate-data` |
| `./organize` | Organize models and logs | `./organize` |
| `./download-models` | Download advanced models | `./download-models --all` or `./download-models --model-type gb` |
| `./!help` | Display help information | `./!help` or `./!help --topic model-types` |
| `./commands.py` | Interactive command reference | `./commands.py` or `./commands.py [command]` |
| `./start.py` | Automated environment setup and training | `./start.py` |

To set up these shortcuts (or recreate them if needed), run:

```bash
python utilities/setup_links.py
```

## Data Format

The tool expects data in the following format:

1. Physiological data with heart rate, blood pressure, etc.
2. EEG data with brain wave measurements
3. Gaze tracking data with eye movement metrics

Each data file should include a timestamp column for synchronization.

Sample data can be generated using:

```bash
./generate-data
```

## Configuration

You can customize the tool by modifying the `.env` file:

```
# Data files
PHYSIO_DATA_PATH=data/Enhanced_Workload_Clinical_Data.csv
EEG_DATA_PATH=data/000_EEG_Cluster_ANOVA_Results.csv
GAZE_DATA_PATH=data/008_01.csv

# Model configuration
MODEL_OUTPUT_DIR=models/sample/default
MODEL_NAME=Cognitive_State_Prediction_Model

# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=logs/general/cwt.log

# Training parameters
TEST_SIZE=0.2
RANDOM_SEED=42

# Default model type
DEFAULT_MODEL_TYPE=rf
```

## Organizing Your Models and Logs

If you have existing models and logs that need to be reorganized to match the new directory structure, run:

```bash
python utilities/organize_outputs.py
```

This script will:

1. Organize model files by type (rf, svm, gb, mlp, knn, lr)
2. Separate advanced models from sample models
3. Organize logs by operation type

## Troubleshooting

If you encounter issues with data paths:

1. Ensure your data files are in the locations specified in `.env`
2. Or run `python cwt.py install-models` to use sample data and models

### Scaler File Issues

If you encounter errors related to scaler files not being found:

1. Use explicit model and scaler paths when making predictions:

   ```bash
   python cwt.py predict --input data/sample_input.json --model models/sample/rf/your_model.joblib --scaler models/sample/rf/your_scaler.joblib
   ```

2. Verify the scaler file exists in the same directory as the model:

   ```bash
   # List models and their scalers
   find models -name "cognitive_state_predictor_*.joblib" | sort
   find models -name "scaler_*.joblib" | sort
   ```

3. Reinstall sample models to ensure consistent naming:

   ```bash
   python cwt.py install-models
   ```

4. Run the organize script to organize your models and scalers:

   ```bash
   ./organize
   ```

5. Use the check-models utility:

   ```bash
   # Check all models and their scalers
   ./check-models
   
   # Automatically fix scaler issues
   ./check-models --fix
   ```

   This tool will identify models missing scalers and create them automatically when run with the `--fix` flag.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite it as:

```
@software{CognitiveWorkloadTool,
  author = {Your Name},
  title = {Cognitive Workload Assessment Tool},
  year = {2025},
  url = {https://github.com/yourusername/CWT-Learning_Model}
}
```
