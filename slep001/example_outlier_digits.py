"""
Small example doing data filtering on digits for t-SNE embedding.
"""
from time import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold, datasets, decomposition, pipeline

from outlier_filtering import EllipticEnvelopeFilter
from subsampler import SubSampler

digits = datasets.load_digits()
X = digits.data
y = digits.target
n_samples, n_features = X.shape


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    plt.subplot(111)
    for this_x, this_y in zip(X, y):
        plt.text(this_x[0], this_x[1], str(this_y),
                 color=plt.cm.Set1(this_y / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


print("Computing t-SNE embedding")

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

subsampler = SubSampler(random_state=1, ratio=.5)

filtering = EllipticEnvelopeFilter(random_state=1)

t0 = time()

# We need a PCA reduction of X because MinCovDet crashes elsewhere
X_pca = decomposition.RandomizedPCA(n_components=30).fit_transform(X)
filtering.fit_pipe(*subsampler.transform_pipe(X_pca))

print("Fitting filtering done: %.2fs" % (time() - t0))

X_red, y_red = filtering.transform_pipe(X_pca, y)

X_tsne = tsne.fit_transform(X_red)

plot_embedding(X_tsne, y_red,
               "With outlier_filtering")


# Now without outlier_filtering
X_tsne = tsne.fit_transform(X_pca)

plot_embedding(X_tsne, y,
               "Without outlier_filtering")

plt.show()

