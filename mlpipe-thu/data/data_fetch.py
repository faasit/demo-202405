import numpy as np
from sklearn.datasets import fetch_openml
import sys

sample_size = int(sys.argv[1])

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser="pandas")
X, y = mnist.data, mnist.target.astype(np.int32)
X = X[:sample_size]
y = y[:sample_size]
np.save('mnist_X.npy', X)
np.save('mnist_y.npy', y)