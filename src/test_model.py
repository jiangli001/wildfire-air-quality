"""
Quick test script to verify model can run without full training.
Useful for debugging before committing to a long training run.
"""

import torch
import numpy as np
from train_pm25_model import PM25LSTM, PM25Dataset
from torch.utils.data import DataLoader


def test_model_instantiation():
    """Test that model can be instantiated."""
    print("Testing model instantiation...")

    try:
        model = PM25LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            forecast_horizon=24,
            output_size=1
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created successfully")
        print(f"  ✓ Total parameters: {total_params:,}")

        return True, model

    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        return False, None


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\nTesting forward pass...")

    try:
        # Create dummy input: (batch_size, window_size, input_size)
        batch_size = 4
        window_size = 24
        input_size = 1

        dummy_input = torch.randn(batch_size, window_size, input_size)

        # Forward pass
        output = model(dummy_input)

        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Input shape: {dummy_input.shape}")
        print(f"  ✓ Output shape: {output.shape}")

        # Check output shape
        expected_shape = (batch_size, 24, 1)  # forecast_horizon = 24
        if output.shape == expected_shape:
            print(f"  ✓ Output shape correct: {output.shape}")
        else:
            print(f"  ✗ Output shape incorrect: {output.shape} (expected {expected_shape})")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model):
    """Test backward pass and gradient computation."""
    print("\nTesting backward pass...")

    try:
        # Create dummy data
        dummy_input = torch.randn(4, 24, 1)
        dummy_target = torch.randn(4, 24, 1)

        # Forward pass
        output = model(dummy_input)

        # Compute loss
        criterion = torch.nn.MSELoss()
        loss = criterion(output, dummy_target)

        print(f"  ✓ Loss computed: {loss.item():.6f}")

        # Backward pass
        loss.backward()

        # Check gradients
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )

        if has_gradients:
            print(f"  ✓ Gradients computed successfully")
        else:
            print(f"  ✗ No gradients computed")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_step(model):
    """Test optimizer step."""
    print("\nTesting optimizer step...")

    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Create dummy data
        dummy_input = torch.randn(4, 24, 1)
        dummy_target = torch.randn(4, 24, 1)

        # Training step
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = torch.nn.MSELoss()(output, dummy_target)
        loss.backward()
        optimizer.step()

        # Check if parameters changed
        params_changed = any(
            not torch.equal(initial, current)
            for initial, current in zip(initial_params, model.parameters())
        )

        if params_changed:
            print(f"  ✓ Parameters updated successfully")
        else:
            print(f"  ✗ Parameters did not change")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Optimizer step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cuda_transfer(model):
    """Test model transfer to CUDA if available."""
    print("\nTesting CUDA transfer...")

    if not torch.cuda.is_available():
        print("  ℹ CUDA not available, skipping test")
        return True

    try:
        device = torch.device('cuda')

        # Move model to GPU
        model_gpu = model.to(device)

        # Create dummy input on GPU
        dummy_input = torch.randn(4, 24, 1).to(device)

        # Forward pass on GPU
        output = model_gpu(dummy_input)

        # Check output is on GPU
        if output.is_cuda:
            print(f"  ✓ Model runs on CUDA")
            print(f"  ✓ GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        else:
            print(f"  ✗ Output not on CUDA")
            return False

        # Move back to CPU
        model.cpu()
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"  ✗ CUDA transfer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_and_dataloader():
    """Test dataset and dataloader."""
    print("\nTesting Dataset and DataLoader...")

    try:
        # Create dummy data
        n_samples = 100
        window_size = 24
        forecast_horizon = 24
        n_features = 1

        X = np.random.randn(n_samples, window_size, n_features)
        y = np.random.randn(n_samples, forecast_horizon, n_features)

        # Create dataset
        dataset = PM25Dataset(X, y)

        print(f"  ✓ Dataset created: {len(dataset)} samples")

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        print(f"  ✓ DataLoader created: {len(dataloader)} batches")

        # Test iteration
        for X_batch, y_batch in dataloader:
            print(f"  ✓ Batch shapes: X={X_batch.shape}, y={y_batch.shape}")
            break  # Just test one batch

        return True

    except Exception as e:
        print(f"  ✗ Dataset/DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_overfitting_on_small_batch():
    """Test if model can overfit on a small batch (sanity check)."""
    print("\nTesting overfitting capability...")

    try:
        # Create a small dataset
        X = torch.randn(8, 24, 1)
        y = torch.randn(8, 24, 1)

        # Create model
        model = PM25LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0.0,  # No dropout for overfitting test
            forecast_horizon=24,
            output_size=1
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        # Train for a few iterations
        initial_loss = None
        final_loss = None

        for epoch in range(50):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 49:
                final_loss = loss.item()

        print(f"  ✓ Initial loss: {initial_loss:.6f}")
        print(f"  ✓ Final loss: {final_loss:.6f}")

        # Check if loss decreased significantly
        if final_loss < initial_loss * 0.1:  # Loss should drop to at least 10% of initial
            print(f"  ✓ Model can learn (loss decreased by {(1 - final_loss/initial_loss)*100:.1f}%)")
        else:
            print(f"  ⚠️  Loss did not decrease significantly")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Overfitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("PM2.5 Model Quick Tests")
    print("=" * 80)

    results = {}

    # Test 1: Model instantiation
    success, model = test_model_instantiation()
    results['instantiation'] = success

    if not success:
        print("\n⚠️  Cannot proceed without model instantiation")
        return False

    # Test 2: Forward pass
    results['forward_pass'] = test_forward_pass(model)

    # Test 3: Backward pass
    results['backward_pass'] = test_backward_pass(model)

    # Test 4: Optimizer step
    results['optimizer_step'] = test_optimizer_step(model)

    # Test 5: CUDA transfer
    results['cuda_transfer'] = test_cuda_transfer(model)

    # Test 6: Dataset and DataLoader
    results['dataset_dataloader'] = test_dataset_and_dataloader()

    # Test 7: Overfitting capability
    results['overfitting'] = test_overfitting_on_small_batch()

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)

    if all_passed:
        print("✓ All tests passed! Model is ready for training.")
    else:
        print("⚠️  Some tests failed. Please fix the issues before training.")

    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
