import sys
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
losses = np.loadtxt(filename)

plt.plot(losses)
plt.ylabel('some numbers')
plt.show()