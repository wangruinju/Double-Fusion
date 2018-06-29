---
output:
  html_document: default
  pdf_document: default
---
# Double-Fusion
This repository documentation is used to explain the model in the papar by Kang, Hakmook, et al. "[A bayesian double fusion model for resting-state brain connectivity using joint functional and structural data](https://www.liebertpub.com/doi/abs/10.1089/brain.2016.0447)." Brain connectivity 7.4 (2017): 219-227.

# Introduction

Our brain network, as a complex integrative system, consists of many different regions that have each own task and function and simultaneously share structural and functional information. With the developed imaging techniques such as functional magnetic resonance imaging (fMRI) and diffusion tensor imaging (DTI), researchers can investigate the underlying brain functions related to human behaviors and some diseases or disorders in the nervous system such as major depressive disorder (MDD).

We developed a Bayesian hierarchical spatiotemporal model that combined fMRI and DTI data jointly to enhance the estimation of resting-state functional connectivity. Structural connectivity from DTI data was utilized to construct an informative prior for functional connectivity based on resting-state fMRI data through the Cholesky decomposition in a mixture model. The analysis took the advantages of probabilistic programming package as PyMC3 and next-generation Markov Chain Monte Carlo (MCMC) sampling algorithm as No-U-Turn Sampler (NUTS). 

# Installation

To install the Python packages for the project, clone the repository and run:

```
conda env create -f environment.yml
```

from inside the cloned directory. This assumes that [Anaconda Python](https://www.continuum.io/downloads) is installed.

For Windows users, environmental variables need to be added in the system setting to use `conda` command in the terminal and `theano` package in Python environment. C++ compiler such as Cygwin needs to be installed before running `pymc3` and `theano` in Windows if it does not exist.

```
# activate the environment
source activate double-fusion
# run the model
python model.py
```

# Spatiotemporal Structure

In a resting-state fMRI study, we define the time-series data at voxel $v​$ in RO $c​$ as $Y_{cv}(t)​$, where $t​$ = 1, … , $T​$. In the same ROI $c​$, a spatiotemporal model for the resting-state fMRI time-series can be expressed as the following:

$$Y_{cv}(t) = \beta_c + b_{c}(v) + d_c + \epsilon_{cv}(t)$$

In the formula above, $\beta_c$ is the grand mean in the ROI $c$. $b_c(v)$ represents a zero mean voxel-specific random effect in the ROI $c$ and captures the local spatial dependency between voxels. A kernel function $K_c(v, v')$ is defined as the covariance structure for local spatial covariance. It is a function of Euclidean distance:

$$Cov(b_c(v), b_c(v')) = K_c(||v-v'||)$$

Note that the voxel-specific random effect $b$ values are uncorrelated when two voxels

correspond to different ROIs ($c \neq c'$), which means the expression below:

$$Cov(b_c(v), b_c(v')) = 0 \text{ at } c \neq c'$$

This kernel function can be any valid spatial covariance function. In our model, we apply the exponential function to represent the covariance structure between voxels:

$$Cov(b_c(v), b_c(v')) = \sigma_{b_c}^2exp(-||v-v'||/\phi_c)$$

where $ \sigma_{b_c}^2$ is defined as the spatial variance at each voxel in the ROI $c$ and $||v-v'||$ denotes the Euclidean distance between two voxels, $c$ and $c'$. $\phi_c$ represents the ROI-specific decaying parameter in the exponential structure.