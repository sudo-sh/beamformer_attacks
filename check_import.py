import numpy as np

loaded_data = np.load("Experiment_1/epoch_test.npz")

loaded_arr1 =loaded_data["range"]
loaded_arr2 =loaded_data["speed"]
loaded_arr3 =loaded_data["angle"]
loaded_arr4 =loaded_data["cov_mat"]

print("Loaded array1:", loaded_arr1)
print("Loaded array2:", loaded_arr2)
print("Loaded array3:", loaded_arr3)
print("Loaded array4:", loaded_arr4)
