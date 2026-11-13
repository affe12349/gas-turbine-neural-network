# train.py
# Neural network built from scratch in NumPy to predict gas turbine degradation.
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# -------------------
# Create output folder
# -------------------
os.makedirs("outputs_numpy", exist_ok=True)

# -------------------
# Load data
# -------------------
data = np.loadtxt("maintenance.txt")
X = data[:, :16].astype(np.float64)
Y = data[:, 16:].astype(np.float64)  # 2 outputs: compressor, turbine

# -------------------
# Train/val/test split (50/25/25)
# -------------------
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.5, random_state=1337)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=1337)

# -------------------
# Scaling (fitted on training data only)
# -------------------
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_Y0 = MinMaxScaler()
scaler_Y1 = MinMaxScaler()
Y_train_scaled = np.column_stack((
    scaler_Y0.fit_transform(Y_train[:, 0:1]).flatten(),
    scaler_Y1.fit_transform(Y_train[:, 1:2]).flatten()
))
Y_val_scaled = np.column_stack((
    scaler_Y0.transform(Y_val[:, 0:1]).flatten(),
    scaler_Y1.transform(Y_val[:, 1:2]).flatten()
))
Y_test_scaled = np.column_stack((
    scaler_Y0.transform(Y_test[:, 0:1]).flatten(),
    scaler_Y1.transform(Y_test[:, 1:2]).flatten()
))

# -------------------
# Activation functions
# -------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float64)

# -------------------
# Weighted MSE
# -------------------
def weighted_mse(y_true, y_pred, w_turbine=2.0):
    mse_comp = np.mean((y_true[:, 0] - y_pred[:, 0])**2)
    mse_turb = np.mean((y_true[:, 1] - y_pred[:, 1])**2)
    return mse_comp + w_turbine * mse_turb

# -------------------
# Initialize weights with He initialization
# -------------------
layer_sizes = [16, 128, 64, 32, 16, 2]
params = {}
np.random.seed(42)
for i in range(len(layer_sizes) - 1):
    params[f"W{i+1}"] = np.random.randn(layer_sizes[i], layer_sizes[i+1]).astype(np.float64) * np.sqrt(2.0 / layer_sizes[i])
    params[f"b{i+1}"] = np.zeros((1, layer_sizes[i+1]), dtype=np.float64)

# -------------------
# Forward pass
# -------------------
def forward(X, params):
    cache = {"A0": X}
    L = len(params) // 2
    for l in range(1, L):
        Z = cache[f"A{l-1}"] @ params[f"W{l}"] + params[f"b{l}"]
        A = relu(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A
    # Last layer (linear)
    ZL = cache[f"A{L-1}"] @ params[f"W{L}"] + params[f"b{L}"]
    cache[f"Z{L}"] = ZL
    cache[f"A{L}"] = ZL
    return ZL, cache

# -------------------
# Backpropagation
# -------------------
def backward(Y_true, cache, params, w_turbine=2.0):
    grads = {}
    L = len(params) // 2
    m = len(Y_true)
    Y_pred = cache[f"A{L}"]

    # dZ for the last layer
    dZL = (Y_pred - Y_true) / m
    dZL[:, 1] *= w_turbine  # extra weight on turbine error
    grads[f"dW{L}"] = cache[f"A{L-1}"].T @ dZL
    grads[f"db{L}"] = np.sum(dZL, axis=0, keepdims=True)

    dA_prev = dZL @ params[f"W{L}"].T

    for l in reversed(range(1, L)):
        dZ = dA_prev * relu_derivative(cache[f"Z{l}"])
        grads[f"dW{l}"] = cache[f"A{l-1}"].T @ dZ
        grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)
        if l > 1:
            dA_prev = dZ @ params[f"W{l}"].T

    return grads

# -------------------
# Adam optimizer
# -------------------
def update_params(params, grads, lr, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1, m_cache=None, v_cache=None):
    if m_cache is None:
        m_cache = {k: np.zeros_like(v) for k, v in grads.items()}
    if v_cache is None:
        v_cache = {k: np.zeros_like(v) for k, v in grads.items()}

    for key in params.keys():
        grad = grads["d" + key]
        m_cache["d" + key] = beta1 * m_cache["d" + key] + (1 - beta1) * grad
        v_cache["d" + key] = beta2 * v_cache["d" + key] + (1 - beta2) * (grad**2)
        m_hat = m_cache["d" + key] / (1 - beta1**t)
        v_hat = v_cache["d" + key] / (1 - beta2**t)
        params[key] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    return params, m_cache, v_cache

# -------------------
# Training
# -------------------
epochs = 20000
lr = 1e-3
batch_size = 1024
w_turbine = 2.0

history = {"loss": [], "val_loss": []}
m_cache, v_cache = None, None
step = 0  # counts every Adam update (one per mini-batch)

for epoch in range(epochs):
    idx = np.random.permutation(len(X_train_scaled))
    X_shuff = X_train_scaled[idx]
    Y_shuff = Y_train_scaled[idx]

    for i in range(0, len(X_train_scaled), batch_size):
        X_batch = X_shuff[i:i+batch_size]
        Y_batch = Y_shuff[i:i+batch_size]

        Y_pred, cache = forward(X_batch, params)
        grads = backward(Y_batch, cache, params, w_turbine)
        step += 1
        params, m_cache, v_cache = update_params(params, grads, lr, t=step, m_cache=m_cache, v_cache=v_cache)

    # Log loss
    Y_pred_train, _ = forward(X_train_scaled, params)
    Y_pred_val, _ = forward(X_val_scaled, params)
    loss_train = weighted_mse(Y_train_scaled, Y_pred_train, w_turbine)
    loss_val = weighted_mse(Y_val_scaled, Y_pred_val, w_turbine)
    history["loss"].append(loss_train)
    history["val_loss"].append(loss_val)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Train loss: {loss_train:.6e}, Val loss: {loss_val:.6e}")

# -------------------
# Evaluation on the test set (used once, at the end)
# -------------------
Y_pred_test_scaled, _ = forward(X_test_scaled, params)
Y_pred_test = np.column_stack((
    scaler_Y0.inverse_transform(Y_pred_test_scaled[:, 0:1]).flatten(),
    scaler_Y1.inverse_transform(Y_pred_test_scaled[:, 1:2]).flatten()
))

mse_compressor = np.mean((Y_test[:, 0] - Y_pred_test[:, 0])**2)
mse_turbine = np.mean((Y_test[:, 1] - Y_pred_test[:, 1])**2)

print("\nFINAL TEST RESULTS:")
print(f"Compressor MSE: {mse_compressor:.3e}")
print(f"Turbine MSE:    {mse_turbine:.3e}")

with open("outputs_numpy/test_metrics.json", "w") as f:
    json.dump({
        "compressor_mse": float(mse_compressor),
        "turbine_mse": float(mse_turbine)
    }, f)

# -------------------
# Plot
# -------------------
plt.figure(figsize=(10, 6))
plt.plot(history["loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Weighted MSE")
plt.legend()
plt.grid()
plt.savefig("outputs_numpy/val_loss.png")
plt.show()
