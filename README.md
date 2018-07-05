# Double-Fusion
This repository documentation is used to explain the model in the papar by Kang, Hakmook, et al. "[A bayesian double fusion model for resting-state brain connectivity using joint functional and structural data](https://www.liebertpub.com/doi/abs/10.1089/brain.2016.0447)." Brain connectivity 7.4 (2017): 219-227.

Since GitHub doest not render the equation in Markdown, you can read the [Readme](https://htmlpreview.github.io/?https://github.com/wangruinju/Double-Fusion/blob/master/README.html)in HTML or [slides](https://github.com/wangruinju/Double-Fusion/blob/master/slides.pdf). 

# Introduction

Our brain network, as a complex integrative system, consists of many different regions that have each own task and function and simultaneously share structural and functional information. With the developed imaging techniques such as functional magnetic resonance imaging (fMRI) and diffusion tensor imaging (DTI), researchers can investigate the underlying brain functions related to human behaviors and some diseases or disorders in the nervous system such as major depressive disorder (MDD).

We developed a Bayesian hierarchical spatiotemporal model that combined fMRI and DTI data jointly to enhance the estimation of resting-state functional connectivity. Structural connectivity from DTI data was utilized to construct an informative prior for functional connectivity based on resting-state fMRI data through the Cholesky decomposition in a mixture model. The analysis took the advantages of probabilistic programming package as [PyMC3](https://github.com/pymc-devs/pymc3) and next-generation Markov Chain Monte Carlo (MCMC) sampling algorithm as No-U-Turn Sampler ([NUTS](https://arxiv.org/abs/1111.4246)). PyMC3 is new, open-source framework with a readable but powerful syntax close to the natural syntax statisticians will use to describe models. NUTS avoids the random walk behavior and sensitivity to correlated parameters by taking a series of steps informed by first-order gradient information. In other words, the NUTS method is a self-tuning variant of Hamiltonian Monte Carlo (HMC).

# Installation

To install the Python packages for the project, clone the repository and run:

```
conda env create -f environment.yml
```

from inside the cloned directory. This assumes that [Anaconda Python](https://www.continuum.io/downloads) is installed. And `theanorc` file can follows as:

```
# cpu
[global]
floatX = float 32
config.compile.timeout = 1000

# gpu
[global]
floatX = float32
config.compile.timeout = 1000
device = cuda0
```

For Windows users, environmental variables need to be added in the system setting to use `conda` command in the terminal and `theano` package in Python environment. C++ compiler such as Cygwin needs to be installed before running `pymc3` and `theano` in Windows if it does not exist.

To run the model:

```
# activate the environment
source activate double-fusion
# run the model
python model.py
```

We also add some examples using CPU/GPU on Vanderbilt Advanced Computing Center for Research and Education (ACCRE).

# Spatiotemporal Structure

In a resting-state fMRI study, we define the time-series data at voxel $v$ in ROI $c$ as $Y_{cv}(t)$, where $t$=1, ..., $T$. In the same ROI $c$, a spatiotemporal model for the resting-state fMRI time-series can be expressed as the following:

$$Y_{cv}(t) = \beta_c + b_{c}(v) + d_c + \epsilon_{cv}(t)$$

In the formula above, $\beta_c$ is the grand mean in the ROI $c$. $b_c(v)$ represents a zero mean voxel-specific random effect in the ROI $c$ and captures the local spatial dependency between voxels. A kernel function $K_c(v, v')$ is defined as the covariance structure for local spatial covariance. It is a function of Euclidean distance:

$$Cov(b_c(v), b_c(v')) = K_c(||v-v'||)$$

Note that the voxel-specific random effect $b$ values are uncorrelated when two voxels correspond to different ROIs ($c \neq c'$), which means the expression below:

$$Cov(b_c(v), b_c(v')) = 0 \text{ at } c \neq c'$$


This kernel function can be any valid spatial covariance function like the following:

$$r=\lVert v-v' \rVert \psi_c$$

$$Cov(b_c(v), b_c(v')) = \sigma_{b_c}^2exp(-r)$$

$$Cov(b_c(v), b_c(v')) = \sigma_{b_c}^2exp(-r^2)$$

$$Cov(b_c(v), b_c(v')) = \sigma_{b_c}^2(1+\sqrt{5}r+\frac{5}{3}r^2)exp(-\sqrt{5}r)$$

$$Cov(b_c(v), b_c(v')) = \sigma_{b_c}^2(1+\sqrt{3}r)exp(-\sqrt{3}r)$$

In our model, we apply the exponential function to represent the covariance structure between voxels:

$$Cov(b_c(v), b_c(v')) = \sigma_{b_c}^2exp(-\lVert v-v' \rVert \psi_c)$$
where $\sigma_{b_c}^2$ is defined as the spatial variance at each voxel in the ROI $c$ and $||v-v'||$ denotes the Euclidean distance between two voxels, $c$ and $c'$. $\psi_c$ represents the reversed ROI-specific decaying parameter in the exponential structure. 

$d_c$ is a zero-mean ROI-specific random effect. Its covariance structure is used to model functional connectivity and expressed as $Cov(b_c(v), b_c(v'))$. We will explain how this effect results from naive FC and DTI data with a series of prior information. 

Finally, $\epsilon_{cv}(t)$ is the noise part. We assume this voxel-specific noise follows an autoregressive (AR) temporal process with order one, that is AR (1). So, the expression of the noise follows:    

$$\epsilon_{cv}(t)=\delta_c + \phi_{cv}\epsilon_{cv}(t-1)+w(t)$$

where $\delta_c$ is the constant shift, $\phi_{cv}$ is the AR (1) coefficient with a requirement of $|\phi_{cv}| < 1$. And $w(t)$ is Gaussian random noise with a distribution as $N(0, \sigma_{cv}^2)$ and is independent of $\epsilon_{cv}(t)$. It is straightforward to calculate the mean and variance of $\epsilon_{cv}(t)$ as the following:

$$E[\epsilon_{cv}(t)] = \frac{\delta_c}{1-\phi_{cv}}$$

$$Var[\epsilon_{cv}(t)]=\frac{\sigma_{cv}^2}{1-\phi_{cv}^2}$$

# Hierarchical Structure

Our goal is to estimate each functional connectivity through its corresponding posterior distribution. To obtain the posterior distribution, each component in the spatiotemporal structure from last section can be rewritten as a hierarchical structure in one ROI level:

$$\boldsymbol{Y}_{cv}= \boldsymbol{\beta}_c + \boldsymbol{b}_c + \boldsymbol{d}_c +\boldsymbol{\epsilon}_c(t)$$

$\boldsymbol{Y}_{cv}$ denotes a vector ($(1 \times V)$) of signals at each voxel as $[Y_{c1}(t), Y_{c2}(t),...,Y_{cv}(t)]^T$. We use $\boldsymbol{J}$ and $\boldsymbol{I}$ to indicate the all-one vector and identity matrix, respectively. Therefore, each component can be vectorized as:
$$\boldsymbol{\beta}_c = \beta_c\boldsymbol{J}_{(1 \times V)}$$
$$\boldsymbol{b}_c=[b_{c1}, b_{c2},...,b_{cV}]^T$$
$$\boldsymbol{d}_c = d_c\boldsymbol{J}_{(1 \times V)}$$
$$\boldsymbol{\epsilon}_c(t)=[\epsilon_{c1}(t),\epsilon_{c2}(t),...,\epsilon_{cV}(t)]^T$$
And the hierarchical structure follows:
$$\beta_c \sim N(0, \sigma_{\beta_c}^2)$$
$$\boldsymbol{b}_c \sim N(0, \Sigma_{b_c})$$
$$d_c \sim N(0, \Sigma_d)$$
$$\epsilon_{cv}(t) \sim N(\frac{\delta_c}{1-\phi_{cv}}, \frac{\sigma_{cv}^2}{1-\phi_{cv}^2})$$

In details, each term $\beta_c$ follows a Gaussian distribution with mean zero and variance $\sigma_{\beta_c}^2$. In addition, for different ROIs ($c \neq c'$), $\beta_c$ is independent of $\beta_{c'}$. For the term $\boldsymbol{b}_c$, it follows a Gaussian distribution with the covariance $\Sigma_{b_c}$, which applies the distant-dependent exponential function. For the term $d_c$, we assume it to follow a Gaussian distribution as $N(0, \Sigma_d)$. Here the correlation matrix of $d_c$ represents the functional connectivity among all ROIs, which is considered as the most important parameters estimated in the whole Bayesian framework. For the noise part with AR (1) time-series structure, it follows a voxel-specific Gaussian distribution as $N(\frac{\delta_c}{1-\phi_{cv}}, \frac{\sigma_{cv}^2}{1-\phi_{cv}^2})$.

# Double Fusion

$\Sigma_d$, as the covariance matrix of $d_c$, is obtained through a novel method considering both structural and functional information. In other words, the prior distribution of the correlation matrix can be established from the structural and naive functional connectivity of each ROI in two steps and we name this method as "a double fusion model".

We combine the structural connectivity and naive functional connectivity together because the effect of direct structural connectivity is different from that of indirect structural connectivity. For example, relatively lower values of structural connectivity imply no direct correlated pathways between two ROIs. However, it is likely that there exist indirect structural connections between two ROIs, resulting in high functional coupling. And the low structural connectivity will indicate very low functional connectivity if no structural connection exists. Therefore, we should treat the indirect structural connectivity differently from direct structural connectivity in the fusion step.

Then the prior distribution of the covariance matrix $\Sigma_d$ is further considered to be a function of structural and naive functional connectivity matrix. With the Cholesky decomposition, we name $L_{sc}$, $L_{nfc}$ and $L_d$ as the lower triangular matrix from the structural covariance matrix, naive functional covariance matrix and functional covariance matrix. To identify the different effects from direct structural connectivity and indirect structural connectivity, we denote $L_d(direct)$ and $L_d(indirect)$ as the lower triangular matrix from direct and indirect structural information, separately. To regulate each source of information, we assume a weighted combination to represent $L_d(direct)$, $L_d(indirect)$ and $L_d$:
$$L_d(direct) = \lambda L_{sc} + (1-\lambda) L_{nfc}$$
$$L_d(indirect) = M_{sc}\lambda L_{sc} + (1- M_{sc}\lambda)L_{nfc}$$
$$L_d = wL_d(direct) + (1-w) L_d(indirect)$$
where $\lambda$ and $w$ are weighed parameter. $M_{sc}$ is the measurement of structural connectivity and denotes another mixture weight, implying the indirect links between structural connectivity and naive functional connectivity. Finally, $\Sigma_d$ is reconstructed as $L_d \times {L_d}^T$ and the corresponding correlation matrix $\rho_d$ can be obtained to denote the resting-state functional connectivity through a normalization step. It is also important to mention that the estimated $\Sigma_d$ and $\rho_d$ are demonstrated to be positive semidefinite due to the Cholesky decomposition and reconstruction. For a correlation matrix within $n$ ROIs, it follows:

$$\rho_d = 
{\left(\begin{array}{cc} 
1 & \rho_{12} & \dots & \rho_{1n} \\
  & 1 & \ddots & \vdots \\
  & & 1 & \rho_{(n-1)n} \\
  & & & 1
\end{array}\right)}_{n \times n}$$

The elements in the upper triangular part can be vectorized as:
$$[\rho_{12},...,\rho_{1n}, \rho_{23},..., \rho_{2n},..., \rho_{(n-1)n}]_{n_{vec}}$$
The total number of estimation is $n_{vec} = \frac{n(n-1)}{2}$.

# Prior Distribution

Because we have no prior information about the values of each parameter, we decide to apply uninformative priors. In the exponential function, we assume the corresponding parameters follow:
$$\psi_c \sim Unif(0, 20)$$
$$ \sigma_{b_c} \sim Unif (0, 100)$$
In the temporal correlation, we assume the prior distribution of each parameter as:
$$\phi_{cv} \sim Unif(0, 1)$$
$$\sigma_{cv} \sim Unif(0, 100)$$
And the grand mean $\beta_c$:
$$\beta_c \sim N(0, 0.01^2)$$
Two weighted parameters $\lambda$ and $w$ in the double fusion model:
$$\lambda \sim Beta(1, 1)$$
$$w \sim Beta(1, 1)$$
Also, the covariance matrix for functional connectivity and structural connectivity is constructed via a prior diagonal matrix. The diagonal element is generated from a function of a $\sigma_{d_c}$ parameter in a logarithmic scale:
$$\sigma_{d_c} \sim Unif(-8, 8)$$
Finally, with adding all the separate components, we assume the observed values $\boldsymbol{Y}_{obs}$ follow a Gaussian distribution $\boldsymbol{Y}_{obs} \sim N(\boldsymbol{Y}_{cv}, \sigma^2)$ under the Bayesian framework and $\sigma$ has a prior distribution:
$$\sigma \sim Unif(0, 100)$$
