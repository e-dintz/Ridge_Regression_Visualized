# Ridge_Regression_Visualized
In this repository, I share a nice visual exploration of ridge regression through animations. First, a ground truth low degree polynomial is generated with noise, and the (not so great) least squares fit with a much higher order polynomial is shown. Then, the ridge regression hyperparameter is tuned to make a sparse model with lower variance.

### How to use this codebook
In the first chunk, you can choose any (preferably low-degree) polynomial as a ground truth model and tune the noise of the generated data.
<div align="center">
  <img src="chunk1.png" width="500">
</div>

Next, you can compare the least squares fit of a high degree polynomial (typically overfit) with one ridge regression fit, for instance $\alpha=10$. The \texttt{sklearn} ridge regression package solves the optimization problem $\left\{\vec{y} \vec{X}\alpha\vec{\beta}\right\}$, in other it solves for a parameter vector that minimizes the residuals and the $\ell^2$ norm of the parameter vector $\vec{\beta}$.
<div align="center">
  <img src="chunk2_1.png" width="300"><img src="chunk2_2.png" width="300">
</div>

To explore the ridge fit further, the next chunk allows you to do a train-test split and choose a hyperparameter to try sparsify the high degree polynomial fit.
<div align="center">
  <img src="chunk3.png" width="300">
</div>
