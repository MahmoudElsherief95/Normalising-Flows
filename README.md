# Normalising-Flows

In this assignment, we take a look at the change of variables formula
for distributions and how it can be used to train a generative model.
These generative models are known as normalising flows,
and make it possible to do density estimation with neural networks.

## Functions and Distributions

One of the foundations of normalising flows is the *change of variable* formula,
which allows to reason about the effect functions have on distributions.
Given a random variable $Y = f(X)$, where $f$ is a bijective function
and $X$ is a random variable from a distribution with pdf $p_X$,
the change of variable allows us to write down the pdf, $p_Y$,
that defines the distribution of $Y$.

For a scalar function $f : \mathbb{R} \to \mathbb{R}$,
the change of variable formula is given as follows:

$$\begin{aligned}
  p_Y(y) = p_X(f^{-1}(y)) \cdot \left|\frac{\partial f^{-1}(y)}{\partial y}\right|.
\end{aligned}$$

This formula reveals that the probability density of $Y = f(X)$
is proportional to the probability density of $X$.
Moreover, the proportion of densities is the gradient of the inverse,
or, equivalently, the inverse of the gradient, $f'(X)$.

To gain some intuition for the change of variable formula,
it might help to consider the visualisation of the histogram below.
Consider inputs distributed according to $X \sim \mathcal{U}(0, 2)$,
which is illustrated by the histogram below the $x$-axis.
To get a feeling for the distribution of $Y = X^2$,
we map each bin in the histogram to its corresponding range on the $y$-axis.

The *probability densities* are the height of each bin.
The area of each bin represents the actual probability,
and must be conserved through function application.
This means that if a bin on the $x$-axis is mapped to a smaller or larger range,
the density on the $y$-axis will proportionally grow or shrink, respecively.
Therefore, the change in probability density is proportional
to the the change of range due to the function.
For infinitesimally small bins, this "change of range"
eventually corresponds to the derivative.

<figure>
  <figcaption style="text-align: center"> Change of Variables through the function $f(x) = x^2$ </figcaption>
  <figure style="display: inline-block; max-width: 33%; margin: 0;">
    <img alt="visualisation of change of variables formula with one histogram bin" src="data:image/svg+xml,%3C%3Fxml%20version%3D%221.0%22%20encoding%3D%22utf-8%22%3F%3E%0A%3Csvg%20viewBox%3D%220%200%20500%20500%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22200%22%20y1%3D%22320%22%20x2%3D%22200%22%20y2%3D%22300%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22180%22%20y1%3D%22300%22%20x2%3D%22200%22%20y2%3D%22300%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Crect%20x%3D%22200%22%20y%3D%22320%22%20width%3D%22100%22%20height%3D%2260%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22300%22%20y1%3D%22320%22%20x2%3D%22300%22%20y2%3D%22100%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22180%22%20y1%3D%22100%22%20x2%3D%22300%22%20y2%3D%22100%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Crect%20x%3D%22150%22%20y%3D%22100%22%20width%3D%2230%22%20height%3D%22200%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Cpath%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20fill%3A%20none%3B%22%20d%3D%22M%20200%20300%20C%20233.33%20300%20266.67%20233.33%20300%20100%22%2F%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%22%20x1%3D%22200%22%20y1%3D%2290%22%20x2%3D%22200%22%20y2%3D%22310%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%22%20x1%3D%22190%22%20y1%3D%22300%22%20x2%3D%22410%22%20y2%3D%22300%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%3C%2Fsvg%3E" style="width: 500px" />
    <figcaption style="text-align: center">(a) 1-bin histogram</figcaption>
  </figure>
  <figure style="display: inline-block; max-width: 33%; margin: 0;">
    <img alt="visualisation of change of variables formula with two histogram bins" src="data:image/svg+xml,%3C%3Fxml%20version%3D%221.0%22%20encoding%3D%22utf-8%22%3F%3E%0A%3Csvg%20viewBox%3D%220%200%20500%20500%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22200%22%20y1%3D%22320%22%20x2%3D%22200%22%20y2%3D%22300%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22180%22%20y1%3D%22300%22%20x2%3D%22200%22%20y2%3D%22300%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Crect%20x%3D%22200%22%20y%3D%22320%22%20width%3D%2250%22%20height%3D%2260%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22250%22%20y1%3D%22320%22%20x2%3D%22250%22%20y2%3D%22250%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22180%22%20y1%3D%22250%22%20x2%3D%22250%22%20y2%3D%22250%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Crect%20x%3D%22120%22%20y%3D%22250%22%20width%3D%2260%22%20height%3D%2250%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Crect%20x%3D%22249.96%22%20y%3D%22320%22%20width%3D%2250%22%20height%3D%2260%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22300%22%20y1%3D%22320%22%20x2%3D%22300%22%20y2%3D%22100%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22180%22%20y1%3D%22100%22%20x2%3D%22300%22%20y2%3D%22100%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Crect%20x%3D%22160%22%20y%3D%22100%22%20width%3D%2220%22%20height%3D%22150%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Cpath%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20fill%3A%20none%3B%22%20d%3D%22M%20200%20300%20C%20233.33%20300%20266.67%20233.33%20300%20100%22%2F%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%22%20x1%3D%22200%22%20y1%3D%22310%22%20x2%3D%22200%22%20y2%3D%2290%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%22%20x1%3D%22190%22%20y1%3D%22300%22%20x2%3D%22410%22%20y2%3D%22300%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%3C%2Fsvg%3E" style="width: 500px" />
    <figcaption style="text-align: center">(b) 2-bin histogram</figcaption>
  </figure>
  <figure style="display: inline-block; max-width: 33%; margin: 0;">
    <img alt="visualisation of change of variables formula with four histogram bins" src="data:image/svg+xml,%3C%3Fxml%20version%3D%221.0%22%20encoding%3D%22utf-8%22%3F%3E%0A%3Csvg%20viewBox%3D%220%200%20500%20500%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22200%22%20y1%3D%22320%22%20x2%3D%22200%22%20y2%3D%22300%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22180%22%20y1%3D%22300%22%20x2%3D%22200%22%20y2%3D%22300%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Crect%20x%3D%22200.04%22%20y%3D%22320%22%20width%3D%2225%22%20height%3D%2260%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22225%22%20y1%3D%22320%22%20x2%3D%22225%22%20y2%3D%22287.5%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22180%22%20y1%3D%22287.5%22%20x2%3D%22225%22%20y2%3D%22287.5%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Crect%20x%3D%2260%22%20y%3D%22287.5%22%20width%3D%22120%22%20height%3D%2212.5%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Crect%20x%3D%22225%22%20y%3D%22320%22%20width%3D%2225%22%20height%3D%2260%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22250%22%20y1%3D%22320%22%20x2%3D%22250%22%20y2%3D%22250%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22180%22%20y1%3D%22250%22%20x2%3D%22250%22%20y2%3D%22250%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Crect%20x%3D%22140%22%20y%3D%22250%22%20width%3D%2240%22%20height%3D%2237.5%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Crect%20x%3D%22250%22%20y%3D%22320%22%20width%3D%2225%22%20height%3D%2260%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22275%22%20y1%3D%22320%22%20x2%3D%22275%22%20y2%3D%22187.5%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22180%22%20y1%3D%22187.5%22%20x2%3D%22275%22%20y2%3D%22187.5%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Crect%20x%3D%22156%22%20y%3D%22187.5%22%20width%3D%2224%22%20height%3D%2262.5%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Crect%20x%3D%22275%22%20y%3D%22320%22%20width%3D%2225%22%20height%3D%2260%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22300%22%20y1%3D%22320%22%20x2%3D%22300%22%20y2%3D%22100%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20stroke-dasharray%3A%205px%3B%22%20x1%3D%22180%22%20y1%3D%22100%22%20x2%3D%22300%22%20y2%3D%22100%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Crect%20x%3D%22162.9%22%20y%3D%22100%22%20width%3D%2217.14%22%20height%3D%2287.5%22%20style%3D%22stroke%3A%20rgb%28255%2C%20255%2C%20255%29%3B%20fill%3A%20rgb%2831%2C%20119%2C%20180%29%3B%22%2F%3E%0A%20%20%3Cpath%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%20fill%3A%20none%3B%22%20d%3D%22M%20200%20300%20C%20233.33%20300%20266.67%20233.33%20300%20100%22%2F%3E%0A%20%20%3Cg%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%22%20x1%3D%22200%22%20y1%3D%22310%22%20x2%3D%22200%22%20y2%3D%2290%22%2F%3E%0A%20%20%20%20%3Cline%20style%3D%22stroke%3A%20rgb%280%2C%200%2C%200%29%3B%22%20x1%3D%22190%22%20y1%3D%22300%22%20x2%3D%22410%22%20y2%3D%22300%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%3C%2Fsvg%3E" style="width: 500px" />
    <figcaption style="text-align: center">(c) 4-bin histogram</figcaption>
  </figure>
</figure>

For multi-variable functions, we simply exchange the derivative
by the determinant of the Jacobian, which gives:

$$\begin{aligned}
  p_Y(\mathbf{y}) = p_X(g^{-1}(\mathbf{y})) \cdot \left|\det \mathcal{J}_g(\mathbf{y})\right|^{-1}.
\end{aligned}$$

Note that the determinant of the Jacobian corresponds to
the change of area due to function application.
To gain some intuition for this version of the formula,
the visualisations above can be generalised to e.g. 2d histograms.

### Exercise 1:  Jacobian Check 

To get some feeling for the change of variable formula,
an invertible function with easy-to-compute Jacobian determinant has been provided.
To make sure that there are no mistakes in the provided module,
you can use the `autograd` magic in pytorch to verify its implementation.

-----------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 2: Change of Variables 

With an invertible module like `AffineFunction`,
implementing the change of variables formula should not be too hard.

-----------------------------------------------------------------------------------------------------------------------------------------------------
# Normalising Flows

Given some input distribution and an invertible function,
the change of variables formula gives the distribution of the outputs.
Equivalently, it can be used to get the distribution of the inputs,
given the output distribution:

$$\begin{aligned}
  p_X(\mathbf{x}) = p_Y(g(\mathbf{x})) \cdot \left|\det \mathcal{J}_g(\mathbf{x})\right|.
\end{aligned}$$

Instead of fixing the invertible function in the formula,
it is also possible to fix the input and output distributions.
With this mindset, the change of variables formula
provides a way to learn a function that is able to
transform one distribution to some other distribution.

Normalising flows are parameterised invertible functions
that can be used to learn a function that can transform a distribution.
More concretely, they map a complex data distribution, $X$,
to some very simple target distribution, $Y$.
To make normalising flows a practical generative model,
two key entities should be easy to compute:
 1. the inverse:
    To generate new data, we sample from the simple distribution
    and use the inverse of the normalising flow to transform it
    to a sample in the data distribution.
 2. the (log-)determinant of the Jacobian:
    Normalising flows are trained to maximise the likelihood (see below),
    which can be computed using the change of variables formula.
    The logarithm stems from the fact that the log-likelihood
    has some practical advantages over the plain likelihood.

### Exercise 3: Invertible Networks 

Neural networks are in general not (easily) invertible,
and therefore not suited for normalising flows.
One solution is to make use of so-called *coupling layers*.
A coupling layer simply splits its inputs in two parts
and computes the two parts of the outputs separately.
One part of the output is simply one of the input parts.
This input part is also used as input to a *conditioner network*.
The conditioner network computes the parameters for an invertible transformation
that is used to compute the second part of the output from the other input part.

-----------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 4: Likelihood Maximisation 

Adding the parameters of the normalising flow
in the change of variable formula, gives the following expression

$$\begin{aligned}
  p_X(\mathbf{x} \mathbin{;} \mathbf{w}) = p_Y(g(\mathbf{x} \mathbin{;} \mathbf{w})) \cdot \left|\det \mathcal{J}_g(\mathbf{x} \mathbin{;} \mathbf{w})\right|.
\end{aligned}$$

This is exactly the probability of the data,
given the parameters, or the *likelihood*, of the normalising flow.
By maximising this likelihood, we obtain the function
that is most likely to be the true transformation to go from $X$ to $Y$.
Note that maximising the likelihood is equivalent to
minimising the negative logarithm of the likelihood.
This gives us the objective for training a normalising flow.

-----------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 5: Training a Normalising Flow 

With the invertible network layers and objective function,
we have everything we need to train a normalising flow.
Since multi-layer networks are more powerful,
you will find the `NormalisingFlow` class below useful.

