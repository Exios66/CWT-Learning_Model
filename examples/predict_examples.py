#!/usr/bin/env python3
"""
Example script demonstrating how to use the Cognitive Workload Tool (CWT)
with the provided JSON example files.

This script shows how to:
1. Load example JSON files
2. Make predictions using the trained model
3. Process batch predictions
4. Handle incomplete data

Usage:
    python predict_examples.py
"""

import os
import json
import sys
import subprocess
from pathlib import Path

# Add project root to path so we can import the cwt modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Constants
EXAMPLES_DIR = Path(__file__).resolve().parent / "json_samples"
CWT_SCRIPT = Path(__file__).resolve().parent.parent / "cwt.py"

# Ensure the CWT tool is available
if not CWT_SCRIPT.exists():
    print(f"Error: CWT script not found at {CWT_SCRIPT}")
    print("Make sure you're running this script from the examples directory")
    sys.exit(1)

def run_prediction(input_file, output_file=None):
    """Run the CWT prediction on an input file"""
    cmd = [sys.executable, str(CWT_SCRIPT), "predict", "--input-json", str(input_file)]
    
    if output_file:
        cmd.extend(["--output-json", str(output_file)])
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running prediction: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def process_all_examples():
    """Process all example JSON files"""
    print("=" * 80)
    print("COGNITIVE WORKLOAD PREDICTION EXAMPLES")
    print("=" * 80)
    
    # Get all JSON files in the examples directory
    json_files = list(EXAMPLES_DIR.glob("*.json"))
    
    if not json_files:
        print(f"No JSON examples found in {EXAMPLES_DIR}")
        return
    
    print(f"Found {len(json_files)} example files")
    
    # Process each file
    for json_file in json_files:
        print("\n" + "-" * 60)
        print(f"Processing: {json_file.name}")
        print("-" * 60)
        
        # Read and display the file contents
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            print(f"Batch file with {len(data)} samples")
            # For batch files, we process each sample individually
            output_dir = Path("results")
            output_dir.mkdir(exist_ok=True)
            
            for i, sample in enumerate(data):
                sample_file = output_dir / f"sample_{i+1}.json"
                with open(sample_file, 'w') as f:
                    json.dump(sample, f, indent=2)
                
                print(f"\n>> Sample {i+1} (ID: {sample.get('sample_id', i+1)}):")
                run_prediction(sample_file, output_dir / f"result_{i+1}.json")
        else:
            # Single sample
            print("Single sample file")
            if "workload_intensity" in data:
                print(f"Workload intensity: {data['workload_intensity']}")
            
            run_prediction(json_file)

def main():
    """Main entry point"""
    # Make sure the environment is set up
    if not Path("../models").exists() or not list(Path("../models").glob("**/*.joblib")):
        print("Models directory not found or empty. Setting up CWT environment first...")
        subprocess.run([sys.executable, str(CWT_SCRIPT), "setup"])
    
    process_all_examples()
    
    print("\n" + "=" * 80)
    print("Example predictions complete!")
    print("=" * 80)

if __name__ == "__main__":
    main() 