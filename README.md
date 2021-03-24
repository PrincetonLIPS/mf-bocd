## Active multi-fidelity Bayesian online changepoint detection

_Abstract._ Online algorithms for detecting changepoints, or abrupt shifts in the behavior of a time series, are often deployed with limited resources, e.g., to edge computing settings such as mobile phones or industrial sensors. In these scenarios it may be beneficial to trade the cost of collecting an environmental measurement against the quality or "fidelity" of this measurement and how the measurement affects changepoint estimation. For instance, one might decide between inertial measurements or GPS to determine changepoints for motion. A Bayesian approach to changepoint detection is particularly appealing because we can represent our posterior uncertainty about changepoints and make active, cost-sensitive decisions about data fidelity to reduce this posterior uncertainty. Moreover, the total cost could be dramatically lowered through active fidelity switching, while remaining robust to changes in data distribution. We propose a multi-fidelity approach that makes cost-sensitive decisions about which data fidelity to collect based on maximizing information gain with respect to changepoints. We evaluate this framework on synthetic, video, and audio data and show that this information-based approach results in accurate predictions while reducing total cost.


![Illustration of MF-BOCD on Gaussian data.](https://raw.githubusercontent.com/PrincetonLIPS/mf-bocd/master/images/mi_illustration.png)

## Demo

To replicate the synthetic results in the paper, please see the Jupyter notebooks, `ablation_normal.ipynb` or `ablation_bernoulli.ipynb`.

## Installation

This implementation supports Python 3.X. See [`environment.yml`](https://github.com/PrincetonLIPS/mf-bocd/blob/master/environment.yml) for a list installed packages and their versions. The main packages required for the notebooks are:

```bash
numpy
scipy
matplotlib
```
