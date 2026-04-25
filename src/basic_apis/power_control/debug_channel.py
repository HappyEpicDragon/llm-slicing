"""Check channel matrix numerical properties."""
import numpy as np

np.random.seed(42)
N = 10
H_complex = np.random.randn(N, N) + 1j * np.random.randn(N, N)
H = np.abs(H_complex)

print("Channel Matrix Analysis")
print("=" * 70)
print(f"H shape: {H.shape}")
print(f"H range: [{H.min():.6f}, {H.max():.6f}]")
print(f"H mean: {H.mean():.6f}")
print(f"H diagonal: {np.diag(H)[:5]}")
print(f"\nH[0, :5]:\n{H[0, :5]}")
print(f"\nH²[0, :5]:\n{(H**2)[0, :5]}")

# This is the issue - we're using |h| directly, but the original code
# expects H to already include path loss + shadowing gains!
# The random H we generate has values ~1-2, but real channel gains
# after path loss are much smaller (e.g., 1e-10 to 1e-8)

print("\n" + "=" * 70)
print("The problem: we're testing with H~1, but real channels have")
print("H ~ 1e-9 after path loss. This makes interference dominate.")
print("\nSolution: test with realistic channel gains from generate_channel_sequence()")
