# Fitting a model while marginalizing noise

**Task:** fit underlying probabilistic model with parameters $\theta$, while
marginalizing out the unknown noise associated with each example.

If we observe noisy examples $\{x_n\}$ of underlying samples $\{z_n\}$ from a
density we want to learn, we want to fit:
$$
\mathrm{argmax}_\theta \sum_n \log p(x_n) = \sum_n \log \int p(x_n | z_n, \Sigma_n) p(z_n | \theta) \mathrm{d}z_n.
$$

We'll assume the noise parameters $\Sigma_n$ can be different for each example,
but are known.

When $p(x_n | z_n, \Sigma_n)$ is linear-Gaussian, and $p(z_n | \theta)$ is
a mixture of Gaussians (MoG) model, we can fit the MoG parameters $\theta$ with
an EM algorithm, presented in the
[*Extreme Deconvolution*](https://arxiv.org/abs/0905.2979) paper.

We'd like to scale to larger problems, be able to use neural density estimators,
and relax the noise-model assumptions.

Because there are astronomy problems with $10^9$ datapoints, in only a few
dimensions, we should be able to get a good fit for the underlying density
$p(z_n | \theta)$. It would be a shame to be biased by approximate inference on
the observation noise.


## Suggested research threads

**Baseline:** fit a mixture of Gaussians to the underlying density, using *extreme
deconvolution*. Extend to minibatch training so it can scale, and it's easier to
compare to neural methods that will also be fitted with online methods.

I think the approach to use is the "stepwise" EM approach of
[Cappe and Moulines (2009)](https://arxiv.org/abs/0712.4273).
Just keep smoothed running averages of the statistics from the E-step.
[Liang and Klein (2009)](https://www.aclweb.org/anthology/N09-1069) compared that approach
compared to the Hinton--Neal approach in the context of discrete models for
natural language processing.

Could also do any SGD-like method on a reparameterized model.

A baseline that can infer the underlying $z_n$ in closed form could be useful as
a proposal in approximate inference methods for more complicated models (below).

**Work out suitable toy problems and first steps with a flow:**
At first fit a toy problem without noise, then for small amount of noise (easy
because inference isn't complicated). Could try to make progress with a
variational approach and Monte Carlo approaches were we draw lots of samples to
get accurate estimates of the cost function. Is there a demo of what happens when
approximate inference introduces bias? Or is it not a problem in practice?

**First steps with unbiased estimation (final section of this document):** Set
up small toy problem that's actually tractable, and try to get unbiased
estimates of the gradients for learning, and then fit with SGD. Don't worry
about biting of whole real large-scale problem at first!


## Marginal MAP

The problem of maximizing model parameters while marginalizing out nuisance
parameters is sometimes called "Marginal-MAP". But most of the papers seem to be
about discrete problems(?); the sort of classical graphical model stuff that
appears at UAI.

There is an MCMC paper by [Robert et al (1999)](https://www.semanticscholar.org/paper/Marginal-MAP-estimation-using-Markov-chain-Monte-Robert-Doucet/81ba7e1f318cae40c98528e60cf5e21d76a0e3e9). But it looks expensive and, as presented, not scalable.

## Variational inference

If we can get a recognition network to represent an approximate posterior, we
could follow the variational autoencoder maths to fit the density. Unlike normal
VAE's we will fit a flow rather than an arbitrary generator, so the final
density is more convenient.

It might be hard to get a recognition network to take in and use the observation
noise parameters $\Sigma_n$? Probably easier to start with fixed noise level.
Might fit an extreme deconvolution model and use that to get posteriors? When
starting with toy problems, could even cheat and use the real MoG distribution
to get a posterior while testing other bits of the code.

## GANs

It's easy to forward simulate from a flow and to then add noise. So, in
principle, one could use a GAN-like algorithm to fit a flow to the underlying
density so that a discriminator can't tell synthetic noisy data from real noisy
data.

GANs probably don't have the right objective for science applications, and are
probably finicky to train. I wouldn't personally go this way early on, but it
could be an interesting point of comparison at some point.

## Approximate marginal likelihood

The marginal likelihood given a single observation,
$$
    p(x_n | \Sigma_n, \theta) = \int p(x_n | z_n, \Sigma_n) p(z_n | \theta) \mathrm{d}z_n,
$$
can be approximated in various ways, such as the Laplace approximation, or Monte
Carlo methods. Any of these methods could then be plugged into an SGD scheme for
fitting the flow parameters $\theta$. (I suggest trying a couple!)

If the noise level is small, importance sampling using $N(z_n; x_n, \Sigma_n)$
would probably work well (the posterior given a flat underlying density).
Otherwise inference on a preliminary *extreme deconvolution* fit could give a
more informed proposal.

The ideal proposal distribution is the posterior (given the current model):
$$
    p(z_n | x_n, \Sigma_n, \theta) = \frac{p(x_n | z_n, \Sigma_n) p(z_n | \theta)}{p(x_n | \Sigma_n, \theta)}.
$$
Simple importance sampling with this distribution gives exactly the correct
answer with a single sample -- but we can't do it (even if we got exact
posterior samples from some fancy MCMC variant), because we'd need to use the
marginal likelihood in the estimator. However, using a close approximation to
the posterior (that we can normalize) would work well.

Standard approximations will be biased estimates of the log marginal likelihood.
If a noisy estimate is unbiased in the marginal likelihood, the log of the
estimate will underestimate the log marginal likelihood on average (by Jensen's
inequality). In any case, arbitrary approximations will lead to us fitting the
wrong density $p(z | \theta)$ in the outer-loop SGD -- even if we have infinite
data.

## MCMC specifically

There are MCMC (and SMC) methods for estimating marginal likelihoods, for
example Annealed Importance Sampling (AIS). However, these are probably too
expensive to apply to each example in every training update. (We could be asking
for billions of AIS runs!)

I think (although I'm biased!) the method of
[Murray and Salakhutdinov (2009)](https://homepages.inf.ed.ac.uk/imurray2/pub/09eval_latents/)
looks like a good idea (if we're prepared to do anything more than the quickest
of importance sampling estimates). Although the paper is probably quite hard to
follow, the method will end up being fairly simple. We'd not use a version that
needs to evaluate a transition probability, instead we'd use a posterior
approximation $q(z_n)$ as in the final (terse) discussion and equation (19). In
our context that will be easier, and avoids the technical problems that are
squished into the (far-too-compressed) appendix.

To get low variance in this method, ideally the reference posterior
approximation would look like a variational approximation (inside the posterior)
rather than an importance sampling proposal (covering the posterior).

## Debiasing Monte Carlo

The Murray and Salakhutdinov method is unbiased in the marginal likelihood, and
therefore biased in the log marginal likelihood, which could mess up the
outer-loop SGD.

If we assume the error in the estimator is normally distributed, and can
estimate its variance, we could estimate the bias from the mean of a log-normal
distribution, and try to correct it. We still won't be unbiased though.

Alternatively we can notice that the gradient for SGD (dropping $\Sigma_n$ and
$\theta$ dependence to reduce clutter) is:
$$
    \nabla_\theta \log p(x_n) = \frac{1}{p(x_n)} \nabla p(x_n).
$$
So if we can also get an unbiased estimate for the reciprocal, $1/p(x_n)$, we
can get unbiased estimates of the gradient for SGD.

If we could draw exact posterior samples, we could get unbiased reciprocals. We
use the same $P()$ and $Q()$ extended distributions as in the Murray and
Salakhutdinov method, but assuming the sequence $Z$ is drawn from $P()$, a fully
burnt in Markov chain, not $Q()$, a chain initialized from $q()$. Then we have
an unbiased estimator:
$$
    \frac{1}{p(x_n)} \approx \frac{Q(Z)}{P^*(Z)} = \frac{1}{S} \sum_{s=1}^S \frac{q(z^{(s)})}{p(x, z^{(s)})}.
$$
Fortunately, we don't need exact samples, as discussed by
[Wei and Murray (2017)](https://homepages.inf.ed.ac.uk/imurray2/pub/17mtrunc/).
There have also been a bunch of recent developments in this area:
[Jacob et al. (2019)](https://arxiv.org/abs/1708.03625) will be a read paper for JRSS-B,
and [Jacob has other interesting
work](https://sites.google.com/site/pierrejacob/articles) including a method
specifically for use with HMC, and work for intractable distributions that might
apply. I'd read all the papers in the "Advances in Markov chain Monte Carlo" section.
[Beatson and Adams (2019)](https://arxiv.org/abs/1905.07006) might be useful too.

It may also be possible to attach a telescoping sum estimate to debias $\log p(x_n)$
before differentiating. And that may be less messy. Don't know.

It might be that attempting to debias things introduces a lot of variance, and
might be over-kill for many applications (including Hogg's). If it's possible to
get fairly low-variance marginal likelihood estimates, the bias on the gradients
might not turn out to matter in practice (a research question to be answered!).
However, the amount of activity in the above literature suggests that a
proof-of-concept learning algorithm that uses formally unbiased gradients to do
asymptotically correct marginal MAP could be an interesting stats paper anyway.

The way to succeed will be to follow multiple threads. One of them will be the
best for Hogg's given problem. But the others will be useful, and could lead to
more new ideas. And there's usually an appetite in the literature for
"gold-standard" approaches that will do the right thing given enough resources
-- even if a bit impractical. So whether or not it's worth it for Hogg's
problem, I have hope a new valid approach to marginal MAP is worth polishing.

