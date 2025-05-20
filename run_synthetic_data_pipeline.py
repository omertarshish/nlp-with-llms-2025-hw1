#!/usr/bin/env python
"""
End-to-End Pipeline for POS Tagging with Synthetic Data

This script runs the complete pipeline:
1. Analyze errors from the LR model and extract error categories
2. Generate synthetic data with challenging examples
3. Retrain the model with combined original and synthetic data
4. Evaluate and visualize the improvements

Usage:
    python run_synthetic_data_pipeline.py [--examples NUM] [--errors-per-example NUM]
"""

import os
import sys
import argparse
import subprocess
import time

def run_command(command, description):
    """Run a shell command and print its output."""
    print(f"\n{'='*10} {description} {'='*10}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Stream output in real-time
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
    
    process.stdout.close()
    process.wait()
    
    if process.returncode != 0:
        print(f"Error running command: {command}")
        for line in process.stderr:
            print(line, end="")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run the complete POS tagging pipeline with synthetic data")
    parser.add_argument('--examples', type=int, default=200, help='Number of synthetic examples to generate (default: 200)')
    parser.add_argument('--errors-per-example', type=int, default=3, help='Number of errors per example (default: 3)')
    parser.add_argument('--skip-generation', action='store_true', help='Skip synthetic data generation (use existing files)')
    parser.add_argument('--grok-key', help='Grok API key (if not set in environment)')
    parser.add_argument('--google-key', help='Google API key (if not set in environment)')
    args = parser.parse_args()

    # Start timing
    start_time = time.time()

    print("Starting POS Tagging Pipeline with Synthetic Data...")

    # Step 1: Verify environment and dependencies
    print("\nVerifying environment...")

    # Set API keys from args if provided
    if args.grok_key:
        os.environ["GROK_API_KEY"] = args.grok_key
    if args.google_key:
        os.environ["GOOGLE_API_KEY"] = args.google_key

    # Try to read from file
    try:
        if not os.environ.get("GROK_API_KEY") and os.path.exists("grok_key.ini"):
            with open("grok_key.ini", "r") as f:
                for line in f:
                    if line.startswith("GROK_API_KEY="):
                        os.environ["GROK_API_KEY"] = line.split("=")[1].strip()
                        break
    except Exception as e:
        print(f"Error reading grok_key.ini: {e}")

    try:
        if not os.environ.get("GOOGLE_API_KEY") and os.path.exists("gemini_key.ini"):
            with open("gemini_key.ini", "r") as f:
                for line in f:
                    if line.startswith("GOOGLE_API_KEY="):
                        os.environ["GOOGLE_API_KEY"] = line.split("=")[1].strip()
                        break
    except Exception as e:
        print(f"Error reading gemini_key.ini: {e}")

    # Check API keys
    api_key_set = False
    if os.environ.get("GROK_API_KEY"):
        print("✓ GROK_API_KEY is set")
        api_key_set = True
    elif os.environ.get("GOOGLE_API_KEY"):
        print("✓ GOOGLE_API_KEY is set")
        api_key_set = True
    else:
        print("✗ No API key found. Please set either GROK_API_KEY or GOOGLE_API_KEY")
        print("  You can provide it with --grok-key or --google-key")
        return
    
    # Check for original LR model
    if not os.path.exists("lr_model.pkl") or not os.path.exists("vectorizer.pkl"):
        print("⚠ Original LR model files not found. Comparison with original model will be skipped.")
        print("  To enable comparison, run the ud_pos_tagger_sklearn.ipynb notebook first.")
    else:
        print("✓ Original LR model found")
    
    # Step 2: Generate synthetic data if not skipped
    if not args.skip_generation:
        command = f"python error_explanation.py --generate --examples {args.examples} --errors-per-example {args.errors_per_example}"
        if not run_command(command, "Generating Synthetic Data"):
            return
    else:
        print("\nSkipping synthetic data generation as requested.")
    
    # Step 3: Retrain model with synthetic data
    if not run_command("python lr_with_synthetic_data.py", "Retraining Model with Synthetic Data"):
        return
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*50}")
    print(f"Pipeline completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"New model saved as: lr_model_with_synthetic.pkl")
    print(f"Visualizations saved as: confusion_matrix_comparison.png and fixed_error_patterns.png")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()