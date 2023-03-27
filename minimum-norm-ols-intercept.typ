#import "template.typ": *

#show: project.with(
  title: "Minimum-norm OLS estimator with intercept",
  authors: (
    "Olivier Grisel",
  ),
)

= Context and notation

Consider the minimum norm OLS estimator in the underdetermined case:

$ & &min_(w, w_0) 1/2 (w^T dot.op w + gamma w_0^2) \
  &"s.t." &X dot.op w + w_0 dot.op 1_n = y $  <problem>

- $X$ holds the input feature values and has shape $(n, p)$;
- $y$ is the column vector has shape $(n, 1)$;
- $n$ is the number of samples;
- $p$ is the number of features;
- $w$ is a column vector of trainable parameter shape $(p, 1)$;
- $w_0$ is an extra scalar trainable parameter ("the intercept");
- $1_n$, $1_p$ are column vectors of ones of shape $(n, 1)$ and $(p, 1)$;
- $overline(X)$ is the column vector of the mean of each column of $X$;
- $overline(y)$ is the mean of $y$;
- $X_c$ is the centered version of $X$ such that $X_c = X - 1_n dot.op overline(X)^T$;
- $y_c$ is the centered version of $y$ such that $y_c = y - overline(y) 1_n$;
- $gamma in {0, 1}$ makes it possible decide whether we want to include the
  intercept in the computation of the norm or not.

Setting $gamma = 1$ would yield the standard formulation which is equivalent
to concatenating a column of 1 to $X$ to avoid having to handle a separate
intercept coefficient. In this case we solve as in the standard presentations
that omit the intercept such as #cite("boyd-2007-slides-lecture-8").

However here are interested in the $gamma = 0$ to compute the minimum norm
OLS estimator where the magnitude of the intercept does not participate in
the computation of the norm, to be consistent with the choice to not penalize
the intercept in ridge regression for instance, and ensure the continuity of
the solutions when $alpha -> 0$.

= Solving for $gamma = 0$ with the method of Lagrange multipliers

Consider the centered data:

$ X = X_c +  1_n dot.op overline(X)^T $
$ y = y_c + overline(y) dot.op 1_n $

We can rewrite the generic formulation of the problem in @problem as:

$ & &min_(w, w_0) 1/2 (w^T dot.op w + gamma w_0^2) \
  &"s.t." &X_c dot.op w +  1_n dot.op overline(X)^T dot.op w
   + w_0 dot.op 1_n = y_c + overline(y) dot.op 1_n $

Let's introduce Langrange multipliers $lambda$ to define our unconstrained
objective function:

$ L(w, w_0, lambda) = &1/2 w^T dot.op w + gamma/2 w_0^2 \
  &+ lambda^T dot.op X_c dot.op w
   + (lambda^T dot.op 1_n) (overline(X)^T dot.op w)
   + w_0 lambda^T dot.op 1_n \
  &- lambda^T dot.op y_c - overline(y) lambda^T dot.op 1_n  $

The minimizer of this objective function is a critical point:

- $nabla L_w(w, w_0, lambda) = 0_p$ yields:

$ w + X_c^T dot.op lambda + (lambda^T dot.op 1_n) overline(X) = 0_p $ <grad_w>

- $nabla L_w_0(w, w_0, lambda) = 0$ yields:

$ gamma w_0 + lambda^T dot.op 1_n = 0 $ <grad_w_0>

- $nabla L_lambda(w, w_0, lambda) = 0_n$ yields:

$ X_c dot.op w + (overline(X)^T dot.op w) 1_n
   + w_0 1_n = y_c + overline(y) 1_n $ <grad_lambda>

Right-multiplying @grad_lambda by $1_n^T$ yields:

$ 1_n^T dot.op X_c dot.op w + (1_n^T dot.op 1_n) (overline(X)^T dot.op w)
   + w_0 (1_n^T dot.op 1_n) = 1_n^T dot.op y_c + overline(y)1_n^T dot.op 1_n $

Since $1_n^T dot.op 1_n = n$, $1_n^T dot.op X_c = 0_p$ and
$1_n^T dot.op y_c = 0$ we recover the usual:

$ w_0 = overline(y) - overline(X)^T dot.op w $ <w_0>

Note that @w_0 holds for any value of $gamma$.

For the case where $gamma = 0$, then @grad_w_0 becomes:

$ lambda^T dot.op 1_n = 0 $

and @grad_w yields:

$ w  = - X_c^T dot.op lambda $ <w_from_lambda>

and therefore:

$ w_0 = overline(y) + overline(X)^T dot.op X_c^T dot.op lambda $ <w_0_from_lambda>

Let's subtitute in $w_0$ and $w$ in @grad_lambda:

$ - X_c dot.op X_c^T dot.op lambda
  - (overline(X)^T dot.op X_c^T dot.op lambda) 1_n
  + (overline(y) + overline(X)^T dot.op X_c^T dot.op lambda) 1_n
  = y_c + overline(y) 1_n $

Hence, after simplification, and assuming $X_c dot.op X_c^T$ is invertible:

$ lambda = - (X_c dot.op X_c^T)^(-1) y_c $ <lambda>

and therefore the solution is:

$ &hat(w) &= - X_c^T dot.op (X_c dot.op X_c^T)^(-1) y_c \
  &hat(w)_0 &= overline(y) - overline(X)^T dot.op hat(w) $


The minimum norm solution for the centered problem without intercept
is also the minimum norm solution for the original problem (with intercept).

#bibliography("bibliography.bib")
