---
layout: post
title: "2. Data Translation: Diffusion Bridges and Stochastic Interpolants"
date: 2026-04-21 10:00:00
description: An overview of data-to-data translation methods grounded in diffusion bridges and stochastic interpolants, extending generative modeling frameworks to transport between arbitrary distributions.
tags: generative-models diffusion flow-matching optimal-transport deep-learning
categories: research
thumbnail: assets/img/related/distribution_matching.png
related_posts: false
toc:
  sidebar: left
---

## I. Introduction

<div id="fig-1" class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/related/distribution_matching.png" class="img-fluid rounded" zoomable=true %}
    </div>
</div>
<div class="caption">
    <strong>Figure 1.</strong> Distribution matching methods aim at building a bridge between two probability distributions. Samples \(x_0\) drawn from the source distribution \(p_0\) are mapped through a transport operator \(T\) to produce new samples \(x_1 = T(x_0)\), whose distribution matches the target distribution \(p_1\).
</div>

The problem of neural-network-based *transport* between probability measures is formulated as the identification of a transformation $$T : \mathcal{X}_0 \rightarrow \mathcal{X}_1$$ that maps samples drawn from an initial distribution $$p_0$$ to a target distribution $$p_1$$ ([Fig. 1](#fig-1)). Applications of distribution matching are broadly divided into two main settings:

1. *Generative modeling* aims to bridge a simple, well-understood, and easily sampled *prior distribution* (often Gaussian) to a complex data distribution. In practice, the data distribution is typically accessible only through samples.
2. *Data translation* seeks to bridge two complex distributions. Rather than having access to explicit models, one usually relies on empirical distributions, that is, samples from both the source and target domains.

Recent advances have revealed strong theoretical links between statistical physics and high-dimensional deep learning, motivating the development of a wide range of frameworks inspired by physical processes (diffusion, heat, and Poisson equations...). These approaches have been successfully applied to both generative modeling {% cite xu2022poisson teh2003energy rissanen2023generative song2021scorebased %} and data translation tasks {% cite Särkkä_Solin_2019 schrodinger1932theorie de2021diffusion %}.

**In this post, we focus on setting 2, data translation,** where both $$p_0$$ and $$p_1$$ are arbitrary complex distributions, and the goal is to transport samples from one to the other using *diffusion bridges* and *stochastic interpolants*. We assume access only to samples from both distributions, with no prior knowledge of their structure. Setting 1, generative modeling where $$p_0$$ is a simple Gaussian prior and $$p_1 = p_\text{data}$$, is covered in a [companion post](/blog/2026/generative-modeling/).

While many problems can be framed as data translation tasks, for instance, classification can be interpreted as transporting data from the input space to the label space, we restrict this post to translations that preserve dimensionality. Specifically, we consider mappings between data points that lie in the same space, such as image-to-image translation.

---

## II. Motivation

Our goal is to generate samples from $$p_1$$ given an initial observation $$x_0 \sim p_0$$, that is, to draw from the conditional distribution $$p(x_1 \mid x_0)$$. A natural approach is to train a generative model that outputs $$x_1$$ from noise while being conditioned on $$x_0$$. However, this strategy relies on a sufficiently expressive and robust conditioning mechanism to effectively steer the generation process. Furthermore, initializing from an uninformative prior, such as a Gaussian distribution, can be inefficient in scenarios where the source and target distributions are close in the data space, as in tasks like denoising or deblurring.

These observations motivate the development of *bridges* between data distributions. Early methods addressing this objective include adversarial approaches, with notable examples such as Pix2Pix {% cite isola2017image %} and CycleGAN {% cite CycleGAN2017 %}. Although successful in practice, these methods often face difficulties when modeling complex transport maps and are susceptible to training instabilities. By contrast, the strong generative performance of diffusion models provides a natural incentive to adapt them to data-to-data translation problems.

Initial attempts to employ diffusion models for data-to-data translation focused on combining two separately trained diffusion models, one for $$p_0$$ and another for $$p_1$$. The central idea was to identify a noise level $$t_0$$ at which the forward noising processes applied to both distributions coincide. *SDEdit* and *Dual Diffusion Implicit Bridges* use two-stage procedures: first corrupt the initial data up to time $$t_0$$, and then apply the reverse diffusion process learned on $$p_1$$ to guide the corrupted sample toward $$p_1$$ {% cite meng_sdedit_2022 su_dual_2022 %}. In practice, determining an appropriate value of $$t_0$$ such that the noised distributions align is challenging, and the required noise level may be so large as to remove most of the informative content of the initial sample $$x_0$$. This limitation underscores the necessity of end-to-end approaches that *explicitly* connect the source and target distributions.

More recently, successful applications of the *Doob $$h$$-transform* to *twist* diffusion processes to match the target distribution $$p_1$$ at the terminal time {% cite zhou2024denoising somnath2023aligned %} have inspired a shift toward simpler formulations of the *bridge matching* problem, notably through the use of *stochastic interpolants*.

---

## III. Mixture of Bridges

<div id="fig-2" class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/related/diffusion_bridge.png" class="img-fluid rounded" zoomable=true %}
    </div>
</div>
<div class="caption">
    <strong>Figure 2.</strong> Mixture of Bridges transports samples from \(p_0\) to \(p_1\) using stochastic differential equations. Unlike diffusion and Gaussian flow-matching models, both \(p_0\) and \(p_1\) are complex data distributions. A neural network approximation of the forward drift \(v_\theta\) enables the generation of new samples from \(p_1\) by integrating the forward SDE.
</div>

Motivated by the success of diffusion models, the transport between an initial distribution $$p_0$$ and a target distribution $$p_1$$ is modeled by a stochastic differential equation ([Fig. 2](#fig-2)) that defines a probabilistic bridge between the two:

$$dx_t = v(t, x_t)\,dt + \sigma\,dW_t, \qquad x_0 \sim p_0,\; x_1 \sim p_1.$$

This SDE induces a probability measure over continuous paths $$(x_t)_{t\in[0,1]}$$, whose marginals evolve from $$p_0$$ at time $$t=0$$ to $$p_1$$ at time $$t=1$$. The drift function $$v(t,\cdot): [0,1]\times\mathbb{R}^d \to \mathbb{R}^d$$ steers particles along the bridge, while the diffusion term $$\sigma\,dW_t$$ introduces stochasticity.

In contrast to the forward SDEs used in diffusion models, where the drift is fixed by design, the drift in the bridge SDE is data-dependent. It must be learned so that the induced path measure has prescribed endpoint marginals, ensuring in particular that $$x_1 \sim p_1$$. In practice, we parametrize the drift by a neural network $$v_\theta$$.

Rather than reasoning directly in terms of drift, it is often convenient to characterize the induced path measure itself. In this section, we focus on transports that can be expressed as *mixtures of bridges* {% cite zhou2024denoising tong_simulation-free_2023 %}. Specifically, we consider stochastic processes $$\mathbb{P}$$ whose law over paths admits the decomposition

$$\mathbb{P}((x_t)_{t\in[0,1]}) = \int_{\mathbb{R}^d \times \mathbb{R}^d} \mathbb{Q}\big((x_t)_t \mid x_0, x_1\big) \, d\mathbb{P}_{0,1}(x_0, x_1),$$

where $$\mathbb{P}_{0,1}$$ denotes a joint distribution over endpoints and $$\mathbb{Q}(\cdot \mid x_0, x_1)$$ is a reference bridge conditioned on $$(x_0, x_1)$$. Capital $$\mathbb{P}$$ denotes a stochastic process (distribution over continuous paths $$[0, 1] \rightarrow \mathbb{R}^d$$) of law $$p$$. Intermediate marginals are denoted $$p_t$$ as previously.

Here, $$\mathbb{Q}$$ is a scaled Brownian motion, i.e., $$\mathbb{Q} = \sigma \mathbb{W}$$ with $$\mathbb{W}$$ the Wiener process and $$\sigma$$ the constant diffusion rate introduced above. The global transport can be interpreted as a mixture of simple stochastic bridges, each connecting a pair of endpoints $$(x_0, x_1)$$, with mixing weights given by $$\mathbb{P}_{0,1}$$. It can be rewritten as:

$$\mathbb{P} = \underbrace{\mathbb{P}_{0, 1}}_{\text{coupling}} \, \underbrace{\mathbb{Q}_{\mid 0, 1}}_{\substack{\text{Brownian} \\ \text{bridge}}}$$

This formula casts light on the two main components of the transport, namely the *conditional Brownian bridge* $$\mathbb{Q}_{\mid 0, 1}(\cdot\mid x_0, x_1)$$ and the *coupling* $$\mathbb{P}_{0, 1}$$.

### Conditional Brownian Bridges

<div id="fig-3" class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/related/stochastic_interpolant.png" class="img-fluid rounded" zoomable=true %}
    </div>
</div>
<div class="caption">
    <strong>Figure 3.</strong> Illustration of scaled Brownian bridges \(\mathbb{Q}_{|0,1}\). Increasing the diffusion rate \(\sigma\) adds stochasticity while preserving the fixed endpoints \(x_0\) and \(x_1\). The case \(\sigma = 0\) corresponds to the linear interpolation used in flow matching.
</div>

*Mixture of bridges* uses a *Brownian bridge* as its reference process. Given two data points $$x_0 \sim p_0$$ and $$x_1 \sim p_1$$, the conditional Brownian bridge defines a stochastic path $$\mathbb{Q}_{\mid 0,1}(\cdot \mid x_0, x_1)$$ that starts at $$x_0$$ and ends at $$x_1$$ ([Fig. 3](#fig-3)). Intermediate points along this path can be generated using a *stochastic interpolant* {% cite albergo2023stochastic albergo2022building %}:

$$x_t = (1-t)x_0 + t x_1 + \sigma \sqrt{t(1-t)} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_d),$$

which provides a simple and exact way to sample from the Brownian bridge at any intermediate time $$t \in [0,1]$$. The *diffusion rate* $$\sigma$$ controls the stochasticity of the conditional path. In the case where $$\sigma=0$$, we retrieve flow matching's linear interpolation.

Following the same logic as flow matching and diffusion models, we are interested in the *conditional drift function* $$v_t(\cdot \mid x_0, x_1)$$, and later average conditional drifts to estimate the true drift function from the forward SDE. This conditional drift function takes the following form:

$$v_t(x_t \mid x_0, x_1) = \frac{x_1 - x_t}{1 -t}$$
with $$x_t$$ defined by the stochastic interpolant above.

### Couplings

<div id="fig-4" class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/related/couplings_final.png" class="img-fluid rounded" zoomable=true %}
    </div>
</div>
<div class="caption">
    <strong>Figure 4.</strong> Illustration of the different couplings \(\mathbb{P}_{0,1}\) used in mixtures of bridges. (Left) The <em>independent coupling</em> samples \(x_0\) and \(x_1\) independently. (Middle) The <em>data-dependent coupling</em> defines the joint distribution via a conditional \(p(x_1 \mid x_0)\), matching data points that share relevant information. (Right) The <em>minibatch-OT coupling</em> associates data points by minimizing the total distance between matched pairs.
</div>

The choice of the *coupling* $$\mathbb{P}_{0,1}$$ is crucial ([Fig. 4](#fig-4)). A naive approach is to sample $$(x_0, x_1)$$ independently from $$p(x_0)p(x_1)$$, which allows transporting $$p_0$$ to $$p_1$$ but fails to preserve high-level information from the original data points. For aligned datasets, where pairs of corresponding points $$(x_0, x_1)$$ are available, {% cite albergo2023couplings liu_i2sb_2023 %} suggest using *data-dependent couplings* $$p(x_0, x_1) = p(x_1 \mid x_0)p(x_0)$$, treating the dataset pairs as samples from this coupling. More recently, {% cite theodoropoulos2025feedback %} extended diffusion bridges to scenarios where only a small fraction of aligned data is available. These methods enforce desirable transport properties by explicitly providing pairs of points that share them. For example, in denoising, $$(x_0, x_1)$$ can be a noisy image and its clean counterpart, both preserving the same semantic content. In super-resolution, natural pairs are high-resolution images and their downsampled versions.

However, in many practical scenarios, paired samples are not available, leading to the problem of *unpaired data translation*. This challenge is often addressed using the Schrödinger bridge framework {% cite bortoli2024schrodinger peluchetti_diffusion_2023 %}, which relies on *optimal transport-based couplings* to define plausible correspondences between distributions. We expand on this approach in the **Schrödinger Diffusion Bridges** section.

### Learning the Drift

The drift function of the bridge, $$v(t, \cdot)$$, can be learned by matching it to the *conditional* drift of the Brownian bridge given the endpoints $$(x_0, x_1)$$. A neural network $$v_\theta(t, x_t)$$ is trained via a regression objective over these conditional drifts:

$$\mathcal{L}_\text{bridge}(\theta) = \mathbb{E}_{t \sim U(0,1)}\mathbb{E}_{(x_0, x_1) \sim p(x_0, x_1)} \Big\lVert v_\theta(t, x_t) - v_t(x_t \mid x_0, x_1) \Big\rVert^2,$$

where $$x_t$$ is sampled via the stochastic interpolant and $$v_t(x_t \mid x_0, x_1)$$ is the conditional drift of the Brownian bridge. This gives:

$$\mathcal{L}_\text{bridge}(\theta) = \mathbb{E}_{t \sim U(0,1)}\mathbb{E}_{(x_0, x_1) \sim p(x_0, x_1)} \Big\lVert v_\theta(t, x_t) - \frac{x_1 - x_t}{1-t} \Big\rVert^2.$$

By averaging over all endpoint pairs $$(x_0, x_1)$$, this objective enables the network to learn the marginal drift $$v_t(x_t)$$, which reflects the behavior of the ensemble of conditional drifts. In practice, the network is trained on minibatches of data pairs. The overall training procedure is summarized in the algorithm below. Once trained, the neural network $$v_\theta$$ can be used to estimate $$v_t(x_t)$$ in a numerical solver for the forward SDE.

**Training: Mixture of Bridges**

```
Require: Neural network v_θ, Dataset D = {X_0, X_1}, batch size B

While Training:
    Sample batches of size B: x_0 ∈ X_0, x_1 ∈ X_1
    Sample B timesteps: t ~ U([0, 1])
    Sample B noises: ε ~ N(0, I_d)
    Sample interpolant:
        x_t = (1-t)x_0 + t x_1 + σ √(t(1-t)) ε
    Compute loss:
        L_bridge(θ) ← ‖v_θ(t, x_t, x_0) - (x_1 - x_t)/(1-t)‖²

Return v_θ
```

---

## IV. Schrödinger Diffusion Bridges

The following section introduces a specific instance of mixture of bridges derived from optimal transport, providing complementary background on principled coupling strategies.

Although mixtures of bridges define a probabilistic transport between $$p_0$$ and $$p_1$$, they do not necessarily satisfy properties that are desirable for data-distribution transport, such as straight trajectories for efficient sampling or the preservation of high-level semantic structure. In practice, the choice of the coupling plays a central role, as it directly influences both the curvature of the transport trajectories and the semantic similarity between the original data point $$x_0$$ and its transported counterpart $$x_1$$.

In the unpaired data translation setting, suitable couplings cannot be inferred directly from data and must instead be constructed implicitly. Naive choices, such as independent couplings, often lead to inefficient or semantically inconsistent transports. This motivates the use of principled coupling strategies grounded in *optimal transport* (OT). Within this framework, the Schrödinger bridge problem provides an entropy-regularized formulation of OT on path space. Given a reference Brownian motion $$\mathbb{Q} = \sigma \mathbb{W}$$, the Schrödinger bridge is defined as the stochastic process that interpolates between prescribed marginals $$q_0$$ and $$q_1$$ while minimizing the relative entropy with respect to $$\mathbb{Q}$$:

$$\mathbb{P}^\star = \underset{\mathbb{P}:\, p_0 = q_0,\; p_1 = q_1}{\arg\min} \; D_{\mathrm{KL}}(\mathbb{P} \,\|\, \mathbb{Q}),$$

where $$\mathbb{P}$$ is the stochastic process with marginals $$p_t$$ that bridges $$q_0$$ and $$q_1$$.

Föllmer (1988) {% cite follmer_ecole_1988 %} proved that the Schrödinger problem admits a unique solution $$\mathbb{P}^*$$ that is written as a mixture of (scaled) Brownian bridges weighted by an Entropic Optimal Transport (EOT) plan $$\pi^*_{2\sigma^2}$$:

$$\mathbb{P^*}((x_t)_{t\in[0,1]}) = \int_{\mathbb{R}^d \times \mathbb{R}^d} \mathbb{Q}\big((x_t)_t \mid x_0, x_1\big) \, d\pi^*_{2\sigma^2}(x_0, x_1).$$

The induced endpoint coupling $$\pi^*_{2\sigma^2}$$ coincides with the entropic optimal transport plan with regularization parameter $$2\sigma^2$$:

$$\pi^*_{2\sigma^2}(q_0, q_1) = \underset{\pi: \pi_0 = q_0,\, \pi_1 = q_1}{\arg\min} \int_{\mathbb{R}^d\times \mathbb{R}^d}\lVert x_0 - x_1\rVert^2\,d\pi(x_0, x_1) + 2\sigma^2 D_\text{KL}(\pi \,\|\, q_0 \otimes q_1).$$

Sampling pairs $$(x_0, x_1) \sim \pi^*_{2\sigma^2}$$ has traditionally relied on iterative trajectory simulations combined with drift fine-tuning {% cite liu_flow_2022 bortoli2024schrodinger de2021diffusion %}, resulting in slow training and high computational cost. In contrast, Tong et al. {% cite tong_simulation-free_2023 %} introduce a simulation-free approach that approximates the Schrödinger bridge by estimating the EOT plan at the minibatch level. Notably, the EOT plan between empirical distributions can be efficiently computed using the Sinkhorn algorithm on a small number of samples of the batch.

---

## References

<div style="font-size: 0.85em;">
<div class="post-bibliography">
{% bibliography --cited --group_by none --template post_bib %}
</div>
</div>
