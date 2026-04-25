"""Debug WMMSE implementation by comparing intermediate values."""
import numpy as np

# Test case
np.random.seed(42)
N = 10
H = np.abs(np.random.randn(N, N) + 1j * np.random.randn(N, N))
Pmax = 6.31
noise_var = 3.98e-15
weights = np.ones(N)

print("Testing WMMSE implementation")
print("=" * 70)

# Initialize
b = np.sqrt(Pmax) * np.ones(N)
f = np.zeros(N)
w = np.zeros(N)

print(f"\nInitial b: {b[:3]}")

# First iteration
for i in range(N):
    f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + noise_var)
    w[i] = 1.0 / (1.0 - f[i] * b[i] * H[i, i])

vnew = np.sum(np.log2(w))
print(f"Initial objective: {vnew:.6f}")
print(f"Initial f: {f[:3]}")
print(f"Initial w: {w[:3]}")

# One update iteration
vold = vnew
for i in range(N):
    denom = np.sum(weights * w * np.square(f) * np.square(H[:, i]))
    btmp = weights[i] * w[i] * f[i] * H[i, i] / (denom + 1e-15)
    b[i] = np.clip(btmp, 0.0, np.sqrt(Pmax))
    if i < 3:
        print(f"  User {i}: btmp={btmp:.6f}, denom={denom:.6e}, b[{i}]={b[i]:.6f}")

print(f"\nUpdated b: {b[:3]}")

# Recompute f, w
vnew = 0.0
for i in range(N):
    f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + noise_var)
    w[i] = 1.0 / max(1.0 - f[i] * b[i] * H[i, i], 1e-15)
    vnew += np.log2(w[i])

print(f"Updated objective: {vnew:.6f}")
print(f"Objective change: {vnew - vold:.6f}")

# Compute final power and sum-rate
p = np.square(b)
print(f"\nFinal power: {p[:3]}")
print(f"Max power: {np.max(p):.6f} (should be <= {Pmax:.6f})")

# Compute sum-rate
H2 = H ** 2
rates = []
for k in range(N):
    signal = H2[k, k] * p[k]
    total = H2[k, :] @ p + noise_var
    interference = total - signal
    sinr = signal / interference
    rate = np.log2(1.0 + sinr)
    rates.append(rate)
    if k < 3:
        print(f"  User {k}: signal={signal:.2e}, interference={interference:.2e}, SINR={sinr:.2e}, rate={rate:.4f}")

print(f"\nSum-rate: {np.sum(rates):.4f} bps/Hz")
