# Examples

This directory contains example scripts demonstrating how to use Physics-Informed ML.

## Training Examples

### `train_pendulum_pinn.py`

Complete example of training a Physics-Informed Neural Network for pendulum dynamics.

**Features:**
- Data generation from high-fidelity simulator
- PINN model configuration
- Training with physics constraints
- Visualization of training progress
- Model evaluation and comparison

**Usage:**
```bash
python examples/train_pendulum_pinn.py
```

**Expected Output:**
- Training loss curves
- Prediction vs. ground truth comparison
- Error analysis
- Saved model checkpoint

**Requirements:**
```bash
pip install -e ".[dev,visualization]"
```

## Coming Soon

- `double_pendulum_pinn.py` - Chaotic dynamics with PINNs
- `burgers_equation_fno.py` - Fourier Neural Operator for PDEs
- `navier_stokes.py` - Fluid dynamics simulation
- `interactive_demo.py` - Real-time visualization

## Tips for Best Results

1. **Data Generation:**
   - Use multiple trajectories with varying initial conditions
   - Balance labeled data and physics collocation points

2. **Model Architecture:**
   - Start with 3-4 hidden layers of 64-128 units
   - Use `tanh` activation for smooth solutions
   - Avoid batch norm for time-series problems

3. **Training:**
   - Balance `lambda_physics` and `lambda_data` (try 1:1 first)
   - Use learning rate scheduling for better convergence
   - Monitor both data loss and physics loss

4. **Validation:**
   - Test on unseen initial conditions
   - Check energy conservation for physical systems
   - Verify long-term stability