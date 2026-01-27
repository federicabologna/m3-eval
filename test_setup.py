#!/usr/bin/env python3
"""
Quick test to verify M3-Eval setup is working correctly.

Run this after setup.sh to ensure everything is installed properly.
"""

import sys
import os

def test_import(module_name, display_name=None):
    """Test if a module can be imported."""
    if display_name is None:
        display_name = module_name

    try:
        __import__(module_name)
        print(f"✓ {display_name}")
        return True
    except ImportError as e:
        print(f"✗ {display_name} - {e}")
        return False


def main():
    print("="*50)
    print("M3-Eval Setup Verification")
    print("="*50)
    print()

    all_passed = True

    # Test Python version
    print("Python Version:")
    version = sys.version_info
    if version.major == 3 and version.minor in [11, 12]:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"⚠ Python {version.major}.{version.minor}.{version.micro} (recommended: 3.11 or 3.12)")
    print()

    # Test core dependencies
    print("Core Dependencies:")
    all_passed &= test_import("torch", "PyTorch")
    all_passed &= test_import("transformers", "Transformers")
    all_passed &= test_import("spacy", "spaCy")
    all_passed &= test_import("Levenshtein", "python-Levenshtein")
    print()

    # Test API libraries (optional)
    print("API Libraries (optional):")
    test_import("openai", "OpenAI")
    test_import("anthropic", "Anthropic")
    test_import("google.genai", "Google GenAI")
    print()

    # Test unsloth (optional)
    print("Optimization Libraries (optional):")
    test_import("unsloth", "Unsloth")
    print()

    # Test CUDA availability
    print("GPU Support:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            capability = torch.cuda.get_device_capability(0)
            print(f"  Compute capability: {capability[0]}.{capability[1]}")
        else:
            print("⚠ CUDA not available (will use CPU)")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
    print()

    # Test data files
    print("Data Files:")
    data_dir = "data"
    required_files = [
        "coarse_5pt_expert+llm_consolidated.jsonl",
        "fine_5pt_expert+llm_consolidated.jsonl"
    ]

    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} not found")
            all_passed = False
    print()

    # Test prompt files
    print("Prompt Files:")
    prompts_dir = "code/prompts"
    required_prompts = [
        "coarseprompt_system.txt",
        "fineprompt_system.txt"
    ]

    for filename in required_prompts:
        filepath = os.path.join(prompts_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} not found")
            all_passed = False
    print()

    # Test environment file
    print("Configuration:")
    if os.path.exists(".env"):
        print("✓ .env file found")
    else:
        print("⚠ .env file not found (only needed for API models)")
    print()

    # Test if experiment runner exists
    print("Experiment System:")
    runner_path = "code/experiment_runner.py"
    if os.path.exists(runner_path):
        print("✓ experiment_runner.py")
    else:
        print("✗ experiment_runner.py not found")
        all_passed = False
    print()

    # Summary
    print("="*50)
    if all_passed:
        print("✓ Setup verification PASSED")
        print()
        print("You're ready to run experiments!")
        print()
        print("Try:")
        print("  python code/experiment_runner.py --experiment baseline --model Qwen3-8B --level coarse")
    else:
        print("✗ Setup verification FAILED")
        print()
        print("Some components are missing. Please:")
        print("  1. Run: bash setup.sh")
        print("  2. Ensure data files are in data/ directory")
        print("  3. Check the error messages above")
    print("="*50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
