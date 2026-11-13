# Gas Turbine Degradation Prediction – Neural Network from Scratch in NumPy

A fully connected neural network built **from scratch in NumPy** (no PyTorch, TensorFlow or scikit-learn models) that predicts the degradation state of a naval gas turbine from 16 sensor measurements.

Forward pass, backpropagation, He initialization and the Adam optimizer are all implemented by hand.

## Results

Mean squared error on the held-out test set (original units):

| Output | Test MSE | Target |
|---|---|---|
| Compressor decay coefficient | **1.98 × 10⁻⁸** | ≤ 5.0 × 10⁻⁸ |
| Turbine decay coefficient | **9.81 × 10⁻⁹** | ≤ 1.5 × 10⁻⁸ |

Both outputs beat the target error, the compressor by a factor of about 2.5.

<img width="1487" height="1083" alt="image" src="https://github.com/user-attachments/assets/d0e71d27-5634-41ad-8e84-da83dfbd73e2" />


Training and validation loss fall together and stabilize, which shows the model converged without significant overfitting.

## Dataset

[Condition Based Maintenance of Naval Propulsion Plants](https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants) (UCI Machine Learning Repository)

- 11,934 samples from a gas turbine simulator
- **Inputs (16):** lever position, ship speed, shaft torque, rotational speeds, temperatures, pressures, fuel flow and more
- **Outputs (2):** compressor decay coefficient and turbine decay coefficient
- Random split: 50% training, 25% validation, 25% test

Download the data and place it as `data/data.txt` before running.

## Model

```
Input (16) → Dense 128 → Dense 64 → Dense 32 → Dense 16 → Output (2)
              ReLU        ReLU       ReLU       ReLU       Linear
```

| Setting | Value |
|---|---|
| Weight initialization | He initialization |
| Optimizer | Adam, learning rate 1 × 10⁻³ |
| Training | Mini-batch gradient descent, batch size 1024, 20,000 epochs |
| Loss | Weighted MSE: MSE_compressor + 2 × MSE_turbine |
| Preprocessing | Min-max scaling of inputs and targets to [0, 1] |

The turbine error is weighted twice as heavily because it has the stricter target.

## How to run

```bash
pip install -r requirements.txt
python train.py
```

## What I learned

- How backpropagation works in practice: deriving and implementing the gradients of ReLU and MSE by hand
- Why He initialization matters for deep ReLU networks
- How Adam's momentum and adaptive learning rates speed up convergence compared with plain gradient descent
- How a weighted loss lets you prioritize one output over another

## Author

Afram Goro
