#! /usr/bin/env python

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
import numpy as np
sys.path.append('../../../')
import itertools
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DATA = sys.argv[2]
LAT = np.load(sys.argv[1])

# Plot 2d projection of latent represtations z, given their labels


def plot_class_space_PCA2d(z, labels, out=None, distinct=False):
    z = PCA(n_components=2).fit_transform(z)
    fig = plt.figure()
    if distinct:
        color = plt.get_cmap("viridis")
        color = color(np.linspace(0, 1, 10))
    else:
        color = cm.rainbow(np.linspace(0, 1, 10))
    for l, c in zip(range(10), color):
        ix = np.where(labels == l)[0]
        plt.scatter(z[ix, 0], z[ix, 1], c=c, label=l, s=8, linewidth=0)
    # plt.legend()
    if out:
        fig.savefig(out + ".pdf", dpi=1000)
    else:
        plt.show()

def plot_class_space_tSNE(z, labels, out=None, distinct=False):
    mark = ['x','o']
    if os.path.exists(sys.argv[1].split('.npy')[0]+'_tsne.npy'):
        print 'Using pretrained tSNE: '+sys.argv[1].split('.npy')[0]+'_tsne.npy .....'
        z = np.load(sys.argv[1].split('.npy')[0]+'_tsne.npy')
    else:
        z = TSNE(n_components=2, verbose=2, n_iter=10000).fit_transform(z)
        np.save(sys.argv[1].split('.npy')[0]+'_tsne.npy', z)
    fig = plt.figure()
    if distinct:
        color = plt.get_cmap("viridis")
        color = color(np.linspace(0, 1, len(np.unique(labels))))
    else:
        color = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
    for l, c in zip(range(len(np.unique(labels))), color):
        ix = np.where(labels == l)[0]
        #  plt.scatter(z[ix, 0], z[ix, 1], c=c, label=l, s=8, linewidth=0)
        if l == 0:
            plt.scatter(z[ix, 0], z[ix, 1], c=c, label='Anomalous class (Windstorm)',  s=8, linewidth=1.0, marker=mark[l])
        else:
            plt.scatter(z[ix, 0], z[ix, 1], c=c, label='Normal class (Non-severe + Flood)',  s=8, linewidth=1.0, marker=mark[l])
    plt.legend()
    if out:
        fig.savefig(out + ".pdf", dpi=1000)
    else:
        plt.show()


import utils
if DATA == '':
    dset = utils.load_mnist(val_size=0)
else:
    dset = utils.load_data(DATA)

lset1 = dset.train.labels
lset2 = dset.test.labels
if not(np.array_equal(lset2, np.zeros(shape=(1,1)))):
    lset = np.concatenate((lset1, lset2))
else:
    lset = lset1
#  perm = np.load('CIFAR_perm.npy')
#  lset = lset[perm]

plot_class_space_tSNE(LAT, lset, out=sys.argv[3])
#  plot_class_space_PCA2d(LAT, lset, out=sys.argv[2], distinct=True)
