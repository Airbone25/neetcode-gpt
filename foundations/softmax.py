import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        maxZ = np.max(z)

        ans = []

        total = 0
        for i in z:
            total += np.exp(i-maxZ)

        for i in z:
            val = np.exp(i-maxZ)/total
            ans.append(round(val,4))        
        
        return np.array(ans)
