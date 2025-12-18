"""
Run training and display metrics every epoch in real-time
This script runs training in a way that displays output
"""

import subprocess
import sys
import os

def main():
    print("=" * 80)
    print("BIDIRECTIONAL RC TRAINING WITH REAL-TIME EPOCH DISPLAY")
    print("=" * 80)
    print("")
    print("Starting training...")
    print("Metrics will be displayed every epoch")
    print("Press Ctrl+C to stop training\n")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run training with output visible
    cmd = [
        sys.executable, "train_with_logging.py",
        "--data_path", "..\\nturgbd_skeletons_s001_to_s017\\nturgb+d_skeletons",
        "--max_samples", "500",
        "--epochs", "50",
        "--batch_size", "16",
        "--reservoir_size", "500",
        "--learning_rate", "0.001"
    ]
    
    try:
        # Run with output visible
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            # Check if it's an epoch result line
            if "EPOCH" in line and "RESULTS" in line:
                # This is the formatted epoch display
                pass
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        process.terminate()
        process.wait()

if __name__ == '__main__':
    main()










