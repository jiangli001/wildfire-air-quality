"""
Validation script to check environment setup before training.
Run this before executing train_pm25_model.py
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  WARNING: Python 3.8+ recommended")
        return False
    else:
        print("  ✓ OK")
        return True


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")

    required_packages = {
        'torch': 'PyTorch',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'joblib': 'Joblib'
    }

    all_ok = True

    for package, name in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = module.__version__

            print(f"  ✓ {name}: {version}")
        except ImportError:
            print(f"  ✗ {name}: NOT INSTALLED")
            all_ok = False

    return all_ok


def check_cuda():
    """Check CUDA/GPU availability."""
    print("\nChecking CUDA/GPU...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.version.cuda}")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ Number of GPUs: {torch.cuda.device_count()}")

            # Check memory
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

            print(f"  ✓ GPU Memory: {total_memory:.2f} GB total")
            print(f"    - Allocated: {memory_allocated:.2f} GB")
            print(f"    - Reserved: {memory_reserved:.2f} GB")
            print(f"    - Available: {total_memory - memory_reserved:.2f} GB")

            return True
        else:
            print("  ⚠️  CUDA not available - will use CPU (slower)")
            print("     To use GPU, install PyTorch with CUDA support:")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False

    except ImportError:
        print("  ✗ PyTorch not installed")
        return False


def check_data_files():
    """Check if required data files exist."""
    print("\nChecking data files...")

    # Get the project root (parent of src/)
    src_dir = Path(__file__).parent
    project_root = src_dir.parent

    required_files = {
        'data/final_merged_data.csv': 'Main dataset',
        'src/matrix.py': 'Data preparation module'
    }

    all_ok = True

    for file_path, description in required_files.items():
        full_path = project_root / file_path

        if full_path.exists():
            if file_path.endswith('.csv'):
                # Check file size
                size_mb = full_path.stat().st_size / 1024**2
                print(f"  ✓ {description}: {full_path}")
                print(f"    Size: {size_mb:.2f} MB")
            else:
                print(f"  ✓ {description}: {full_path}")
        else:
            print(f"  ✗ {description}: NOT FOUND at {full_path}")
            all_ok = False

    return all_ok


def check_data_quality():
    """Check data quality and structure."""
    print("\nChecking data quality...")

    try:
        import pandas as pd
        import numpy as np

        # Get the project root
        src_dir = Path(__file__).parent
        data_path = src_dir.parent / 'data' / 'final_merged_data.csv'

        if not data_path.exists():
            print("  ✗ Data file not found")
            return False

        # Load data
        df = pd.read_csv(data_path)

        print(f"  ✓ Loaded {len(df):,} records")

        # Check required columns
        required_cols = ['site', 'date', 'start_hour', 'pm25']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  ✗ Missing required columns: {missing_cols}")
            return False
        else:
            print(f"  ✓ Required columns present: {required_cols}")

        # Check for additional columns
        other_cols = [col for col in df.columns if col not in required_cols]
        if other_cols:
            print(f"  ℹ Additional columns: {other_cols}")

        # Check data types and quality
        print(f"\n  Data Quality:")
        print(f"    - Unique sites: {df['site'].nunique()}")
        print(f"    - Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"    - Hour range: {df['start_hour'].min()} to {df['start_hour'].max()}")

        # PM2.5 statistics
        pm25_stats = df['pm25'].describe()
        print(f"\n  PM2.5 Statistics:")
        print(f"    - Count: {pm25_stats['count']:.0f}")
        print(f"    - Mean: {pm25_stats['mean']:.2f}")
        print(f"    - Std: {pm25_stats['std']:.2f}")
        print(f"    - Min: {pm25_stats['min']:.2f}")
        print(f"    - Max: {pm25_stats['max']:.2f}")

        # Missing values
        missing_pm25 = df['pm25'].isna().sum()
        missing_pct = (missing_pm25 / len(df)) * 100

        if missing_pct > 10:
            print(f"  ⚠️  High percentage of missing PM2.5 values: {missing_pct:.2f}% ({missing_pm25:,} records)")
        elif missing_pm25 > 0:
            print(f"  ℹ Missing PM2.5 values: {missing_pct:.2f}% ({missing_pm25:,} records)")
        else:
            print(f"  ✓ No missing PM2.5 values")

        # Check for negative values
        negative_pm25 = (df['pm25'] < 0).sum()
        if negative_pm25 > 0:
            print(f"  ⚠️  Found {negative_pm25} negative PM2.5 values (should be >= 0)")

        # Check site distribution
        print(f"\n  Records per site:")
        site_counts = df['site'].value_counts().sort_index()
        for site, count in site_counts.items():
            print(f"    - Site {site}: {count:,} records")

            # Check if enough data for windowing
            min_required = 24 + 24  # window_size + forecast_horizon
            if count < min_required:
                print(f"      ⚠️  Site {site} has insufficient data for 24h window + 24h forecast")

        return True

    except Exception as e:
        print(f"  ✗ Error checking data: {e}")
        return False


def check_output_directories():
    """Check/create output directories."""
    print("\nChecking output directories...")

    src_dir = Path(__file__).parent
    project_root = src_dir.parent

    directories = {
        'models': 'Model checkpoints',
        'predictions': 'Prediction outputs'
    }

    for dir_name, description in directories.items():
        dir_path = project_root / dir_name

        if dir_path.exists():
            print(f"  ✓ {description}: {dir_path}")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Created {description}: {dir_path}")
            except Exception as e:
                print(f"  ✗ Failed to create {description}: {e}")
                return False

    return True


def test_data_loading():
    """Test data loading and windowing."""
    print("\nTesting data preparation...")

    try:
        import pandas as pd
        import numpy as np
        from matrix import create_multivariate_windows

        # Load data
        src_dir = Path(__file__).parent
        data_path = src_dir.parent / 'data' / 'final_merged_data.csv'

        df = pd.read_csv(data_path)
        df = df.dropna(subset=['pm25'])

        # Create a small test
        print("  Testing sliding window creation...")
        X, y, metadata = create_multivariate_windows(
            df,
            window_size=24,
            forecast_horizon=24,
            feature_cols=['pm25'],
            stride=1
        )

        print(f"  ✓ Created {len(X)} windows")
        print(f"  ✓ X shape: {X.shape}")
        print(f"  ✓ y shape: {y.shape}")
        print(f"  ✓ Sites in windows: {metadata['site'].unique().tolist()}")

        return True

    except Exception as e:
        print(f"  ✗ Error in data preparation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks."""
    print("=" * 80)
    print("PM2.5 Model Setup Validation")
    print("=" * 80)

    results = {}

    results['python'] = check_python_version()
    results['dependencies'] = check_dependencies()
    results['cuda'] = check_cuda()
    results['data_files'] = check_data_files()
    results['data_quality'] = check_data_quality()
    results['output_dirs'] = check_output_directories()
    results['data_loading'] = test_data_loading()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)

    if all_passed:
        print("✓ All checks passed! Ready to train the model.")
        print("\nNext steps:")
        print("  1. Review configuration in train_pm25_model.py")
        print("  2. Run: python train_pm25_model.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above before training.")

        if not results['dependencies']:
            print("\nTo install dependencies:")
            print("  pip install -r requirements_dl.txt")

        if not results['cuda']:
            print("\nFor GPU support:")
            print("  Install PyTorch with CUDA (check https://pytorch.org/)")

    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
