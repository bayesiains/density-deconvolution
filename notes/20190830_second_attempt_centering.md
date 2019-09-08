These are rough notes that hopefully make sense given our conversation earlier. They'll need polish for wider sharing.
The notation probably needs thinking about more, and I was too lazy to
add indices for which component I'm talking about.

\newcommand{\sumstat}[1]{\widetilde{#1}}
\newcommand{\avestat}[1]{\overline{#1}}
\newcommand{\new}{_\mathrm{new}}
\newcommand{\old}{_\mathrm{old}}
\newcommand{\batch}{_\mathrm{batch}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bb}{\mathbf{b}}
\newcommand{\bc}{\mathbf{c}}
\newcommand{\bm}{\mathbf{m}}

Let one of the exponential family sufficient statistics be $\phi$.

Let the sum of expected statistics over some set $\mathcal{D}$ be $\sumstat\phi_\mathcal{D} = \sum_{n \in \mathcal{D}} \mathbb{E}_\mathrm{latents}[\phi_n]$.

For a mixture of Gaussians, the statistics are $[q, q\bx, q\bx\bx^\top]$, where q is a
$\{0,1\}$ indicator variable for the component we're talking about.

For such mixture models, the expectation turns the q's into a *responsibility*
$r_n\in[0,1]$, the posterior probability the statistic is relevant for datapoint $n$.
In XD the expectation over latents also includes an integral over the unknown noise,
replacing the $\bx$'s with posterior means (and creating an extra term, see end).

For the mixture model case, let an estimate of the mean of a statistic for
relevant data points on this set be
$\avestat\phi_\mathcal{D} = \frac{1}{\sum_{n \in \mathcal{D}} r_n}\sumstat\phi_\mathcal{D}$.

So in particular:

$\sumstat{q} = \sum_n r_n$

$\sumstat{\bm} = \sumstat{q\bx} = \sum_n r_n \bx_n$

$\avestat{\bm} = \avestat{q\bx} = \frac{1}{\sum_n r_n} \sum_n r_n \bx_n = (1/\sumstat{q})\, \sumstat{\bm}$

$\sumstat{S} = \sumstat{q\bx\bx^\top} = \sum_n r_n \bx_n\bx_n^\top$

$\avestat{S} = \avestat{q\bx\bx^\top} = \frac{1}{\sum_n r_n} \sum_n r_n \bx_n \bx_n^\top = (1/\sumstat{q})\, \sumstat{S}$

**We want the exponential running averages of the summed statistics:**

$\sumstat{\phi}\new = 
(1-\lambda)\sumstat{\phi}\old +
\lambda\sumstat{\phi}\batch$

**From these statistics, we set parameters (the M-step):**

Mixing fraction $\alpha = \frac{1}{N} \sumstat{q}$, where $N$ is the
mini-batch size we're using (not the whole dataset).

[The papers largely talk about pure online updates, where $N=1$, so the mixing
fraction can be read straight off the running average without division by $N$.
George's MSc thesis does do the minibatch case, but isn't explicit that the $N$
in his equation (D.7) needs to be set to the minibatch size rather than the
dataset size as before (and all the minibatches need to be the same size). It
might be more natural to divide all the summed $\sumstat{\phi}$ stats by the minibatch size
(which would allow variable minibatch sizes), but the notation would be more cluttered.
Also these $1/N$ terms would cancel out in the ratios used in every other quantity we
compute apart from the mixing fraction estimate.]

The average statistics $\avestat\phi$ notation is useful for setting the remaining
parameters:

Mean of component $\mu = \avestat{\bm}$

Covariance of component $\Sigma = \avestat{S} - \avestat{\bm}\,\avestat{\bm}^\top$

Except computing $\Sigma$ like that fails with single floating point precision.

**Fixing the numerics**

Assume we have a current estimate of the covariance given our data so far, that
was obtained in a numerically stable way, but that is equal to:
$$\begin{aligned}
\Sigma\old &= \avestat{S}\old - \avestat{\bm}\old\avestat{\bm}\old^\top\\
           &= \frac{1}{\sumstat{q}\old}\sumstat{S}\old - \avestat{\bm}\old\avestat{\bm}\old^\top.
\end{aligned}$$
Now we want to update it with an estimate based on the current minibatch (computed using the first line for numerical stability):
$$\begin{aligned}
\Sigma\batch &= \frac{1}{\sumstat{q}\batch} \sum_n r_n (\bx_n - \avestat{\bm}\batch)(\bx_n - \avestat{\bm}\batch)^\top\\
             & \text{[expand out, pull $\bm$'s out of sum, cancel and group]}\\
             &= \avestat{S}\batch - \avestat{\bm}\batch\avestat{\bm}\batch^\top\\
             &= \frac{1}{\sumstat{q}\batch}\sumstat{S}\batch - \avestat{\bm}\batch\avestat{\bm}\batch^\top
\end{aligned}$$
A scale and shift adjustment function makes it easier to combine these estimates:
$$\begin{aligned}
\mathrm{adjust}(\Sigma, s, \bb, \bc)
    &= s\Sigma + \frac{1}{2}(\sqrt{s}\bb-\bc)(\sqrt{s}\bb+\bc)^\top + \frac{1}{2}(\sqrt{s}\bb+\bc)(\sqrt{s}\bb-\bc)^\top\\
    &= s(\Sigma + \bb\bb^\top) - \bc\bc^\top.
\end{aligned}
$$
It's computed as in the first line, but the second line shows that it
removes the centering of a covariance, scales the raw second moment, and then puts in a new centering.

Then
$$\begin{aligned}
\Sigma\new &= (1 - \lambda)\cdot \mathrm{adjust}\!\left(
\Sigma\old,\; \frac{\sumstat{q}\old}{\sumstat{q}\new},\; \avestat{\bm}\old,\; \avestat{\bm}\new\right)
+
\lambda\cdot \mathrm{adjust}\!\left(
\Sigma\batch,\; \frac{\sumstat{q}\batch}{\sumstat{q}\new},\; \avestat{\bm}\batch,\; \avestat{\bm}\new\right)\\
&= \frac{(1-\lambda)\sumstat{S}\old  + \lambda\sumstat{S}\batch}{\sumstat{q}\new} - \avestat{\bm}\new\avestat{\bm}\new^\top\\
&= \frac{\sumstat{S}\new}{\sumstat{q}\new} - \avestat{\bm}\new\avestat{\bm}\new^\top\\
&= \avestat{S}\new - \avestat{\bm}\new\avestat{\bm}\new^\top.
\end{aligned}$$

Again the first line is how the update could be implemented, while the remaining
lines show that it should be mathematically equivalent to tracking updates of
$\sumstat{S}$, and then computing the covariance from that.

In the limit of large mini-batches the scaling ratios will be close to 1, so I
think it's numerically ok(?).

For small mini-batches, there might not be any relevant datapoints, so
$\sumstat{q}\batch\approx0$, and the update will be small. Not seeing any
catastrophic cancellation.

I'm missing some detail. The XD covariance update doesn't just replace the
$\bx$'s with posterior means. There's an extra $B_{ij}$ term added on to the
covariance. Maybe that can be tracked separately and just added on. Or maybe it
causes its own problems. TODO.

The above is rather brute force, and there's probably a neater way to do it. But
the first thing is to get rid of the numerical problems at all. Compare updates
to naive code on well-behaved problems with 64-bit precision to check
correctness. Then see how it works. (Just running with 64-bit precision may be
also enough to see how well the stepwise EM works.)

