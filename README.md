# VillageNet

(c) 2024 Aditya Ballal, Gregory A. DePaul, Asuka Hatano, Esha Datta, Erik Carlsson, Ye Chen-Izu, Javier E. L´opez and Leighton T. Izu

VillageNet is a scalable graph-based clustering framework that combines fast prototype-based partitioning with walk-likelihood community detection. The algorithm first partitions the dataset into a fixed number of representative "villages" using K-Means clustering, constructs a sparse graph based on local neighborhood relationships between observations and villages, and then applies the Walk-Likelihood Community Finder (WLCF) to identify communities in the resulting graph.

The primary motivation behind VillageNet is to efficiently perform community detection and clustering on large, high-dimensional datasets by reducing the complexity of graph construction while preserving meaningful local structure.

More information on Walk-Likelihood Community Finder (WLCF) can be found here:

* Paper: https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.043117
* GitHub: https://github.com/lordareicgnon/Walk_likelihood

More information about VillageNet can be found here:

* Streamlit App: https://villagenet.streamlit.app/

# Installation

The scripts have been tested with Python 3.x.

Required modules:

* numpy
* scipy
* scikit-learn

Install the required dependencies using:

```bash
pip install numpy scipy scikit-learn
```

If using the accompanying Walk-Likelihood implementation, install or clone it according to its repository instructions.

# Overview

Here is an overview of the files included in the repository:

1. `VillageNet.py`: Main implementation of the VillageNet algorithm.
2. `walk_likelihood.py`: Walk-Likelihood Community Finder implementation.
3. `README.md`: Documentation and usage instructions.

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
3. Graph construction.
4. Community detection using Walk-Likelihood.
5. Optional evaluation against reference labels.

The fitted object itself is returned.

## Parameters

* **X (numpy array):** Dataset consisting of N observations and P features.

## Attributes

**X:** Input dataset.

**N:** Number of observations.

**comm_id:** Community assignment for every observation.

# Example

Import the required packages:

```python
>>> import numpy as np
>>> from VillageNet import VillageNet
```

Load a dataset to be clustered:

```python
>>> X = np.load("data.npy")
>>> X.shape
(10000, 50)
```

Create a VillageNet model:

```python
>>> model = VillageNet(
...     villages=100,
...     neighbors=20,
...     normalize=1
... )
```

Fit the model:

```python
>>> model.fit(X)
```

Retrieve community assignments:

```python
>>> model.comm_id
array([3, 3, 1, 7, 2, 2, 0, 5, 5, 1,
       4, 4, 3, 7, 7, 2, 0, 0, 6, 6,
       ...,
       1, 3, 3, 0, 5, 2, 7, 4, 4, 1])
```

The resulting numpy array `model.comm_id` contains the community assignment for every observation in the original dataset.

