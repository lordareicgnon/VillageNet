# VillageNet

(c) 2024 <Author 1>, <Author 2>, <Author 3>

VillageNet is a scalable graph-based clustering framework that combines fast prototype-based partitioning with walk-likelihood community detection. The algorithm first partitions the dataset into a fixed number of representative "villages" using K-Means clustering, constructs a sparse graph based on local neighborhood relationships between observations and villages, and then applies the Walk-Likelihood Community Finder (WLCF) to identify communities in the resulting graph.

The primary motivation behind VillageNet is to efficiently perform community detection and clustering on large, high-dimensional datasets by reducing the complexity of graph construction while preserving meaningful local structure.

More information on Walk-Likelihood Community Finder (WLCF) can be found here:

* Paper: <Paper Link>
* GitHub: <GitHub Link>

More information about VillageNet can be found here:

* Paper: <Paper Link>
* Demo: <Demo Link>
* GitHub: <Repository Link>

# Installation

The scripts have been tested with Python 3.x.

Required modules:

* numpy
* scipy
* scikit-learn
* walk_likelihood
* time

Install the required dependencies using:

```bash
pip install numpy scipy scikit-learn
```

If using the accompanying Walk-Likelihood implementation, install or clone it according to its repository instructions.

# Overview

Here is an overview of the files included in the repository:

1. `VillageNet.py`: Main implementation of the VillageNet algorithm.
2. `walk_likelihood.py`: Walk-Likelihood Community Finder implementation.
3. `example.ipynb`: Example notebook demonstrating VillageNet usage.
4. `README.md`: Documentation and usage instructions.

# class VillageNet

```python
class VillageNet
```

## Initialization

```python
def __init__(self, villages=60, normalize=1, neighbors=20)
```

### Parameters

* **villages (int, default=60):** Number of prototype villages (K-Means clusters) used to summarize the dataset.
* **normalize (bool/int, default=1):** Whether to standardize the input features before clustering.
* **neighbors (int, default=20):** Number of nearest observations assigned to each village during sparse graph construction.

### Attributes

**villages:** Number of village prototypes.

**normalize:** Flag indicating whether feature normalization is performed.

**neighbors:** Number of nearest observations connected to each village.

---

# fit

```python
def fit(self, X, comms=None, ref=None)
```

Fits the complete VillageNet pipeline to the input dataset. The procedure consists of:

1. Optional feature normalization.
2. K-Means village construction.
3. Sparse graph construction.
4. Community detection using Walk-Likelihood.
5. Optional evaluation against reference labels.

The fitted object itself is returned.

## Parameters

* **X (numpy array):** Dataset consisting of N observations and P features.
* **comms (optional):** Desired number of communities or WLCF initialization parameter passed to the Walk-Likelihood algorithm.
* **ref (optional):** Ground-truth labels used for evaluation via Normalized Mutual Information (NMI).

## Attributes

**X:** Input dataset.

**N:** Number of observations.

**comm_id:** Community assignment for every observation.

---

# kmeans

```python
def kmeans(self)
```

Performs the village construction step using K-Means clustering and computes the quantities required for sparse graph generation.

## Attributes

**labels:** Village assignment for every observation.

**cluster_centers:** Coordinates of the village centroids.

**ds:** Squared distances from every observation to every village center.

**distance_matrix:** Pairwise distance matrix between village centers.

**U:** Binary membership matrix indicating village assignments.

**D:** Relative distance metric used for sparse neighborhood construction.

---

# grapher

```python
def grapher(self)
```

Constructs the sparse village graph by assigning each village its nearest observations according to the computed distance metric.

## Attributes

**M:** Sparse observation-village membership matrix.

**village_list:** List containing the observations assigned to each village.

---

# get_communities

```python
def get_communities(self, thr_clusters=128, comms=None, **WLCF_args)
```

Constructs the village adjacency graph and performs community detection using the Walk-Likelihood Community Finder (WLCF).

For large numbers of villages, a randomized initialization matrix is generated before optimization.

## Parameters

* **thr_clusters (int, default=128):** Threshold above which randomized initialization is used.
* **comms (optional):** Number of desired communities or WLCF initialization parameter.
* **WLCF_args:** Additional keyword arguments passed directly to the Walk-Likelihood implementation.

## Attributes

**A:** Village adjacency matrix.

**comm_id:** Community assignment of every observation obtained by mapping village communities back to the original dataset.

## Returns

Returns the fitted Walk-Likelihood model.

# Example

Import the required packages:

```python
>>> import numpy as np
>>> from VillageNet import VillageNet
```

Load a dataset:

```python
>>> X = np.load("data.npy")
>>> X.shape
(10000, 50)
```

Create a VillageNet model:

```python
>>> model = VillageNet(
...     villages=60,
...     neighbors=20,
...     normalize=1
... )
```

Fit the model:

```python
>>> model.fit(X)
time=0.12
time=0.03
time=0.08
```

Retrieve community assignments:

```python
>>> model.comm_id
array([...])
```

The resulting `comm_id` array contains the community assignment for every observation in the original dataset.

