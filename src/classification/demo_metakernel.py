#!/usr/bin/env python3
"""
Demo script for MetaKernel vs MetaVRF comparison
This script demonstrates the improvements brought by conditional normalizing flows
in the MetaKernel framework over the original MetaVRF implementation.

Based on: "MetaKernel: Learning Variational Random Features with Limited Labels"
"""

import sys
import os
import subprocess
import argparse

def run_experiment(dataset, use_flow=True, iterations=100, shot=1, way=5):
    """
    Run a single experiment with specified parameters
    
    Args:
        dataset: Dataset name ('Omniglot' or 'miniImageNet')
        use_flow: Whether to use conditional normalizing flows
        iterations: Number of training iterations
        shot: Number of support examples per class
        way: Number of classes
    
    Returns:
        Command execution result
    """
    
    # Determine which script to run
    if use_flow:
        script_name = "run_metakernel_classifier.py"
        checkpoint_dir = f"./checkpoints_metakernel_{dataset.lower()}"
        experiment_name = f"MetaKernel (with flows)"
    else:
        script_name = "run_classifier.py"  # Original MetaVRF
        checkpoint_dir = f"./checkpoints_metavrf_{dataset.lower()}"
        experiment_name = f"MetaVRF (baseline)"
    
    print(f"\n{'='*60}")
    print(f"Running {experiment_name}")
    print(f"Dataset: {dataset}, {shot}-shot {way}-way")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        "python3", script_name,
        "--dataset", dataset,
        "--mode", "train_test",
        "--shot", str(shot),
        "--way", str(way),
        "--iterations", str(iterations),
        "--checkpoint_dir", checkpoint_dir,
        "--print_freq", "10"
    ]
    
    # Add flow-specific parameters if using MetaKernel
    if use_flow:
        cmd.extend([
            "--use_flow", "True",
            "--num_flow_layers", "4",
            "--flow_hidden_size", "128",
            "--flow_weight", "0.01"
        ])
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✓ {experiment_name} completed successfully")
            # Extract final accuracy from output
            lines = result.stdout.split('\n')
            for line in lines:
                if "Held out accuracy:" in line:
                    print(f"Final Test Accuracy: {line.split('Held out accuracy:')[1].strip()}")
                    break
        else:
            print(f"✗ {experiment_name} failed")
            print("Error output:", result.stderr)
            
        return result
        
    except Exception as e:
        print(f"✗ Error running {experiment_name}: {str(e)}")
        return None


def compare_metakernel_vs_metavrf(dataset="miniImageNet", iterations=100, shot=1, way=5):
    """
    Compare MetaKernel (with flows) vs MetaVRF (without flows)
    
    Args:
        dataset: Dataset to use for comparison
        iterations: Number of training iterations
        shot: Number of support examples per class
        way: Number of classes
    """
    
    print("MetaKernel vs MetaVRF Comparison")
    print("=" * 80)
    print(f"Configuration: {dataset} dataset, {shot}-shot {way}-way learning")
    print(f"Training iterations: {iterations}")
    print()
    
    results = {}
    
    # Run MetaVRF baseline (without flows)
    print("Phase 1: Running MetaVRF baseline...")
    result_metavrf = run_experiment(
        dataset=dataset, 
        use_flow=False, 
        iterations=iterations, 
        shot=shot, 
        way=way
    )
    results['MetaVRF'] = result_metavrf
    
    # Run MetaKernel (with flows)
    print("\nPhase 2: Running MetaKernel with conditional normalizing flows...")
    result_metakernel = run_experiment(
        dataset=dataset, 
        use_flow=True, 
        iterations=iterations, 
        shot=shot, 
        way=way
    )
    results['MetaKernel'] = result_metakernel
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    for method, result in results.items():
        if result and result.returncode == 0:
            # Extract accuracy from output
            lines = result.stdout.split('\n')
            accuracy = "Not found"
            for line in lines:
                if "Held out accuracy:" in line:
                    accuracy = line.split('Held out accuracy:')[1].strip()
                    break
            print(f"{method:15}: {accuracy}")
        else:
            print(f"{method:15}: Failed to complete")
    
    print("\nKey improvements in MetaKernel:")
    print("• Conditional normalizing flows for richer posterior distributions")
    print("• Enhanced random Fourier feature learning")
    print("• Better adaptation to few-shot learning tasks")
    print("• More informative and discriminative kernels")


def ablation_study(dataset="miniImageNet", iterations=50):
    """
    Ablation study on different components of MetaKernel
    
    Args:
        dataset: Dataset to use
        iterations: Number of training iterations
    """
    
    print("MetaKernel Ablation Study")
    print("=" * 80)
    
    configs = [
        {"name": "MetaVRF (baseline)", "use_flow": False, "num_layers": 0},
        {"name": "MetaKernel (2 layers)", "use_flow": True, "num_layers": 2},
        {"name": "MetaKernel (4 layers)", "use_flow": True, "num_layers": 4},
        {"name": "MetaKernel (6 layers)", "use_flow": True, "num_layers": 6},
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running: {config['name']}")
        print(f"{'='*60}")
        
        if config['use_flow']:
            cmd = [
                "python3", "run_metakernel_classifier.py",
                "--dataset", dataset,
                "--mode", "train_test",
                "--iterations", str(iterations),
                "--use_flow", "True",
                "--num_flow_layers", str(config['num_layers']),
                "--checkpoint_dir", f"./ablation_{config['num_layers']}layers"
            ]
        else:
            cmd = [
                "python3", "run_classifier.py",
                "--dataset", dataset,
                "--mode", "train_test", 
                "--iterations", str(iterations),
                "--checkpoint_dir", f"./ablation_baseline"
            ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {config['name']} completed")
                # Extract accuracy
                lines = result.stdout.split('\n')
                for line in lines:
                    if "Held out accuracy:" in line:
                        print(f"Test Accuracy: {line.split('Held out accuracy:')[1].strip()}")
                        break
            else:
                print(f"✗ {config['name']} failed")
        except Exception as e:
            print(f"✗ Error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="MetaKernel Demo Script")
    parser.add_argument("--mode", choices=["compare", "ablation"], default="compare",
                        help="Demo mode: compare MetaKernel vs MetaVRF, or run ablation study")
    parser.add_argument("--dataset", choices=["Omniglot", "miniImageNet"], default="miniImageNet",
                        help="Dataset to use")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--shot", type=int, default=1,
                        help="Number of support examples per class")
    parser.add_argument("--way", type=int, default=5,
                        help="Number of classes")
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        compare_metakernel_vs_metavrf(
            dataset=args.dataset,
            iterations=args.iterations,
            shot=args.shot,
            way=args.way
        )
    elif args.mode == "ablation":
        ablation_study(
            dataset=args.dataset,
            iterations=args.iterations
        )


if __name__ == "__main__":
    main() 