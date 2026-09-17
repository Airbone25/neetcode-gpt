import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        ans = []
        for i in z:
            val = 1 / (1+(np.exp(-i)))
            ans.append(np.round(val,5))
        return np.array(ans)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        ans = []
        for i in z:
            val = max(0,i)
            ans.append(float(val))
        return np.array(ans)
