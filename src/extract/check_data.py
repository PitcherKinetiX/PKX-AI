import numpy as np

data = np.load(r"C:\Users\Yul\PycharmProjects\PitcherKinetiX\data\processed\2d_data.npz")

windows = data["windows"]   # shape (N, 30, 42)

print("min:", windows.min())
print("max:", windows.max())
print("mean:", windows.mean())
print("std:", windows.std())
