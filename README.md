# VillageNet: Robust and Scalable Graph-Based Clustering for Broad Biomedical Applications

VillageNet is a robust and scalable, unsupervised graph-based clustering framework designed to effectively cluster high-dimensional data without a priori knowledge of the number of existing clusters. Integrating topological principles, graph-based network theory, and random-walk analysis, VillageNet reduces the complexity of high-dimensional datasets by constructing a structured multi-scale representation that captures non-linear structures. 

Building upon our previous work, **MapperPlus** (an extension of the topological Mapper algorithm), VillageNet adopts an alternative strategy to identify local linear regions. Instead of uniform discretization across a feature space, VillageNet coarse-grains the dataset using a targeted overclustering strategy to capture underlying structures efficiently, making it highly robust to noise and scalable across broad biomedical applications.

## Overview of the Data Pipeline

The VillageNet algorithm operates via a structured four-phase pipeline:

1. **Coarse-graining into Villages:** The original dataset is subdivided using K-Means into $\nu$ partial-clusters or "villages." Each instance in the dataset is assigned to a unique village corresponding to its Voronoi cell.
2. **Finding the Exterior of Villages:** For each village, its exterior boundary is mapped by identifying the $\eta$ data points outside the village that are nearest to its boundary.
3. **Inter-Village Graph Construction:** Based on the identified exteriors, a weighted graph—termed the *village network*—is constructed. Nodes represent villages, and edge weights between any two villages are computed based on the mutual count of exterior points assigned to each other.
4. **Community Detection on the Village Network:** The village network is partitioned into distinct communities using the Walk-Likelihood Community Finder (WLCF). This step automatically and objectively determines the optimal number of final clusters directly from the data.

---

## Repository Structure

Here is an overview of the files included in this repository:

* `village_net.py` : Main implementation file containing the core `VillageNet` class and pipeline methods.
* `walk_likelihood.py` : File defining the `walk_likelihood` class used for community detection on the constructed graph.
* `example.ipynb` : A Jupyter notebook explaining step-by-step usage and visualization of VillageNet.
