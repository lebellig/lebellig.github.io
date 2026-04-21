---
layout: post
title: "1. Generative Modeling: Score-Based Diffusion and Flow Matching"
date: 2026-04-21 10:00:00
description: An introduction to deep generative modeling, covering score-based diffusion models and flow matching as frameworks for learning transport maps between probability distributions.
tags: generative-models diffusion flow-matching deep-learning
categories: research
thumbnail: assets/img/related/generative_modeling.png
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

This post introduces the fundamental principles of *deep generative modeling* and neural-network-based *transport* between probability measures. The problem is formulated as the identification of a transformation $$T : \mathcal{X}_0 \rightarrow \mathcal{X}_1$$ that maps samples drawn from an initial distribution $$p_0$$ to a target distribution $$p_1$$ ([Fig. 1](#fig-1)). Applications of distribution matching are broadly divided into two main settings:

1. *Generative modeling* aims to bridge a simple, well-understood, and easily sampled *prior distribution* (often Gaussian) to a complex data distribution. In practice, the data distribution is typically accessible only through samples.
2. *Data translation* seeks to bridge two complex distributions. Rather than having access to explicit models, one usually relies on empirical distributions, that is, samples from both the source and target domains.

Recent advances have revealed strong theoretical links between statistical physics and high-dimensional deep learning, motivating the development of a wide range of frameworks inspired by physical processes (diffusion, heat, and Poisson equations...). These approaches have been successfully applied to both generative modeling {% cite xu2022poisson teh2003energy rissanen2023generative song2021scorebased %} and data translation tasks {% cite Särkkä_Solin_2019 schrodinger1932theorie de2021diffusion %}.

**In this post, we focus on setting 1, generative modeling,** presenting the foundations of *score-based diffusion models* and *flow matching*, where $$p_0 = p_\text{prior}$$ is a simple Gaussian prior and $$p_1 = p_\text{data}$$ is the unknown data distribution. The transport maps noise to data, enabling unconditional generation of new samples. Setting 2, data-to-data translation where both $$p_0$$ and $$p_1$$ are arbitrary complex distributions, is covered in a [companion post](/blog/2026/data-translation/).

Transport between probability distributions encompasses a wide range of tasks in deep learning. For instance, classification can be interpreted as a transport from a continuous data distribution to a discrete distribution over labels. In this post, however, we restrict our attention to transport operations between distributions supported on a common continuous space, with samples of identical dimensionality. Consequently, we do not review works on flows over discrete spaces {% cite shi_discrete dieleman2022continuous campbell2024generative gat2024discrete %} or recent advances on variable-length generative models {% cite billera2025branching campbell2023trans %}.

---

## II. Generative Modeling

<div id="fig-2" class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/related/generative_modeling.png" class="img-fluid rounded" zoomable=true %}
    </div>
</div>
<div class="caption">
    <strong>Figure 2.</strong> Generative models transport samples from a simple <em>prior</em> distribution \(p_0 = p_\text{prior}\) (here Gaussian) to the complex underlying distribution \(p_1 = p_\text{data}\).
</div>

In this section, we discuss the recent advances in generative modeling with a focus on *score-based models* (SBMs) and *flow matching* (FM) models.

Given a training dataset $$\mathcal{X}=\{x^{i}\}_{i \in [\![1, N]\!]}$$, generative approaches propose to learn the underlying distribution $$p_\text{data}(x)$$ with a parametric model $$p_\theta(x)$$ of parameters $$\theta$$, i.e. find:

$$p_\theta(x) \approx p_\text{data}(x)$$

A central challenge in generative modeling is the lack of an analytical expression for the data distribution $$p_\text{data}$$. Instead, one typically assumes access only to independent and identically distributed samples, while direct evaluation of the likelihood $$p_\text{data}(x)$$ remains intractable.

The generative modeling objective is therefore twofold: first, to generate new samples from the distribution $$p_\text{data}$$; and second, to evaluate the likelihood of data samples under the learned model.

In practice, the *curse of dimensionality* makes explicit modeling of its distribution and direct sampling intractable in high-dimensional spaces. Consequently, generative modeling is commonly framed as the problem of transporting a simple *prior* distribution $$p_\text{prior}$$ to the empirical data distribution $$p_\text{data}$$. The prior distribution is typically chosen to be well-known and easy to sample from, most often a Gaussian distribution. Under this formulation, new data samples are generated by first drawing samples $$x_0$$ from $$p_\text{prior}$$ and subsequently transporting them through a learned operator $$T_\theta$$.

Within this framework, the goal is to design a parametric transport $$T_\theta$$ bridging the *prior distribution* and the data distribution $$p_\text{data}$$ ([Fig. 2](#fig-2)).

*Generative Adversarial Networks* (GANs) {% cite goodfellow2014generative mao2017least arjovsky2017wasserstein %} introduced the idea of learning such a transport using a *one-step* generator $$T_\theta$$ via adversarial training. In particular, *Wasserstein GANs* aim to minimize the Wasserstein distance between the distribution of generated samples and the real data distribution, thereby encouraging convergence toward $$p_\theta = p_\text{data}$$. However, GANs are known to suffer from *mode collapse*, where only a subset of the modes of $$p_\text{data}$$ is captured {% cite thanh2020catastrophic miyato2018spectral %}. They are also often associated with unstable and difficult training dynamics {% cite wiatrak2019stabilizing mescheder2018training %}. To address these limitations, more recent generative modeling approaches propose learning *time-dependent* transports that progressively move samples from $$p_\text{prior}$$ to $$p_\text{data}$$, frequently relying on differential equations to connect the two distributions.

---

## III. Score-Based Diffusion Models

<div id="fig-3" class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/related/anisotropic_score-10.png" class="img-fluid rounded" zoomable=true %}
    </div>
</div>
<div class="caption">
    <strong>Figure 3.</strong> Visualization of a probability density and its associated score field. The underlying density \(p(x)\) is represented by the blue intensity, while the red arrows depict the score vector field \(\nabla_x \log p(x)\), pointing toward regions of higher likelihood. This illustrates how the score function guides samples toward high-density regions.
</div>

<div id="fig-4" class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/related/reverse_sde_image.png" class="img-fluid rounded" zoomable=true %}
    </div>
</div>
<div class="caption">
    <strong>Figure 4.</strong> Illustration of the forward and reverse diffusion processes. <strong>Top:</strong> the forward SDE progressively corrupts clean data samples \(x_0\) with Gaussian noise, transforming them into noise-like samples \(x_1\). <strong>Bottom:</strong> the reverse-time SDE leverages the learned score function \(\nabla_{x_t}\log p_t(x_t)\) to iteratively denoise samples, transporting noise back toward the data distribution and recovering data-like samples \(x_0'\). Due to the inherent stochasticity of both noising and denoising processes, the generated samples \(x_0'\) differ from the original inputs \(x_0\).
</div>

As a starting point for diffusion models, one constructs a reversible noising process that progressively corrupts data samples into pure noise. From a distributional viewpoint, this forward noising procedure maps the data distribution to a Gaussian prior. The central objective is then to learn the corresponding reverse process, which iteratively denoises samples to recover data-like observations.

While early diffusion models were primarily based on discrete-time noise schedules and *Markov chains* {% cite ho2020denoising pmlr-v37-sohl-dickstein15 %}, we focus here on their continuous-time formulations. *Score-based Diffusion Models* {% cite song2021scorebased JMLR:v6:hyvarinen05a song2020sliced %} (SBDMs) rely on *stochastic differential equations* (SDEs) to connect a Gaussian prior distribution to the data distribution.

In the following, we show that the *score function*

$$s(t, x_t) = \nabla_{x_t} \log p_t(x_t),$$

also referred to as the *Stein score function*, plays a fundamental role in diffusion models. This quantity is the gradient of the log-likelihood and can be viewed as a vector field over the data space $$\mathbb{R}^d$$ that, at each point $$x \in \mathbb{R}^d$$, points toward regions of high probability density, thereby guiding the sampling process toward the target distribution $$p_\text{data}$$ ([Fig. 3](#fig-3)).

### Stochastic Forward Process

The noising procedure, commonly referred to as the *forward process* or *diffusion process*, is modeled by a Stochastic Differential Equation (SDE) defined over the time interval $$t \in [0,1]$$:

$$dx_t = b^F_t(x_t)\,dt + g_t\,dw_t, \quad x_0 \sim p_\text{data}.$$

Here, $$b_t^F : \mathbb{R}^d \to \mathbb{R}^d$$ denotes the *forward drift function*, while $$g_t : \mathbb{R} \to \mathbb{R}^*_+$$ is the *diffusion coefficient* that controls the magnitude of the Wiener process $$w_t$$. The drift governs the evolution of the data points $$x_t$$ in the data space, whereas the diffusion coefficient determines the level of noise injected throughout the process.

The temporal evolution of the data distributions $$p_t$$ induced by the diffusion process is governed by the Fokker–Planck equation (FPE):

$$\partial_t p_t + \nabla\cdot(b^F_t p_t) = \frac{g_t^2}{2}\Delta p_t, \quad p_0 = p_\text{data}.$$

We now have two key tools for understanding diffusion models and the diffusion process. On the one hand, the forward SDE describes the evolution of individual data points throughout the diffusion. On the other hand, the Fokker–Planck equation characterizes the corresponding evolution of the data distribution.

The forward process corrupts the data with noise and, at $$t=1$$, the fully noised data distribution $$p_1$$ is close to the true Gaussian prior.

In Euclidean space, if the forward drift $$b^F$$ is an affine function, samples from an intermediate distribution $$p_t$$ can be obtained directly, without explicitly integrating the forward SDE. Note that on Riemannian manifolds, simulating the forward process typically requires time-discretized integration schemes {% cite de2022riemannian %}. Thus, we can generate $$x_t \sim p_t$$ by interpolating between a sample from the data distribution $$x_0 \sim p_\text{data}$$ and a noise $$x_1 \sim p_\text{prior}$$:

$$x_t = \alpha_t x_0 + \sigma_t x_1$$

where these interpolation coefficients (also called noise schedules) are directly linked to the drift and diffusion coefficients of the forward SDE:

$$b^F_t(x_t) = \frac{d\log \alpha_t}{dt}x_t, \quad g_t = \alpha_t \frac{d}{dt}\left[ \frac{\sigma_t^2}{\alpha_t^2} \right].$$

The choice of the drift function and diffusion coefficient has been extensively studied, leading to multiple noise schedules, the main ones being *variance-preserving* (VP) and *variance-exploding* (VE) {% cite song2021scorebased karras2022elucidating %}.

### Stochastic Backward Process

<div id="fig-5" class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/related/reverse_sde.png" class="img-fluid rounded" zoomable=true %}
    </div>
</div>
<div class="caption">
    <strong>Figure 5.</strong> The reverse diffusion process, i.e. the denoising process, is defined by the backward SDE and leverages the score function to guide pure noise \(x_1\) to the data distribution.
</div>

The backward process iteratively denoises samples, gradually transporting noisy data points $$x_1$$ back toward the original data distribution ([Fig. 4](#fig-4), bottom, and [Fig. 5](#fig-5)). Remarkably, the reverse of the diffusion process can be described by an explicit stochastic differential equation (SDE):

$$dx_t = \underbrace{b^F_t(x_t) - g_t^2 \nabla_{x_t} \log p_t(x_t)}_{b^B_t(x_t)}\, dt + g_t \, d\bar{w}_t, \quad x_1 \sim p_\text{prior}$$

Here, $$\bar{w}_t$$ denotes a time-reversed Wiener process, and the drift function of the backward process is given by $$b^B_t(x_t) = b^F_t(x_t) - g_t^2 \nabla_{x_t} \log p_t(x_t).$$ This backward process also admits an explicit Fokker–Planck equation, describing the evolution of the data distribution during the reverse process:

$$\underbrace{\partial_t p_t}_{\text{time derivative}} + \underbrace{\nabla \cdot (b^B_t p_t)}_{\text{drift term}} = \underbrace{-\frac{g_t^2}{2}\Delta p_t}_{\text{diffusion term}}, \quad p_1= p_\text{prior}.$$

A key insight of score-based diffusion models is that the reverse-time process depends explicitly on the *score function* $$s(t, x_t) = \nabla_{x_t} \log p_t(x_t)$$. In order to generate new data points from the data distribution $$p_\text{data}$$, we want to simulate the reverse process, i.e. use numerical solvers to solve the backward SDE. Since the score function is unknown, score-based diffusion models propose to use neural-based approximations. We detail in Section III.4 how to train neural networks to estimate the score function.

If this function is accurately approximated by a neural network, then the backward process can be simulated and we can draw new samples from $$p_\text{data}$$.

### Deterministic Processes

<div id="fig-6" class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/related/pfode.png" class="img-fluid rounded" zoomable=true %}
    </div>
</div>
<div class="caption">
    <strong>Figure 6.</strong> Visualizing the Probability Flow ODE trajectory. Unlike stochastic diffusion processes, the PF-ODE defines a deterministic path for samples, governed by the drift term and the score function \(\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)\), allowing for an invertible mapping between the prior and data distributions.
</div>

<div id="fig-7" class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/related/pfode_image.png" class="img-fluid rounded" zoomable=true %}
    </div>
</div>
<div class="caption">
    <strong>Figure 7.</strong> Probability Flow ODE (PF-ODE) forward and reverse processes. The forward PF-ODE deterministically transports data samples \(x_0\) toward Gaussian noise \(x_1\). In the reverse direction, the PF-ODE generates a deterministic trajectory from the prior back to the data distribution, guided by the learned score function. This deterministic property ensures that the original sample \(x_0\) can be exactly recovered from its corrupted version.
</div>

The stochastic denoising process enables sampling of new points from the underlying data distribution. However, its inherent stochasticity prevents exact computation of the likelihood for generated points and can produce multiple possible outputs from a single noise sample. In some applications, this stochasticity is undesirable, as we may require a deterministic process to reliably encode and decode data points into the prior distribution.

Interestingly, the forward and backward stochastic processes described above have a deterministic counterpart in the form of the *Probability Flow ODE* (PF-ODE) {% cite song2021scorebased maoutsa2020interacting %}:

$$dx_t = v_t(x_t)\, dt, \quad x_0 \sim p_0$$

where $$v_t: \mathbb{R}^d\rightarrow\mathbb{R}^d$$ is named the *velocity*. This ODE generates smooth, noise-free trajectories that transport particles from $$p_0$$ to $$p_1$$, while preserving the same marginal distributions $$p_t$$ at all times as those of the forward and backward stochastic processes ([Fig. 6](#fig-6)). This deterministic process satisfies the *continuity equation* (also called *transport equation*):

$$\underbrace{\partial_t p_t}_{\text{time derivative}} + \underbrace{\nabla \cdot (v_t p_t)}_{\text{advection term}} = \underbrace{0}_{\text{no diffusion}}, \quad p_0= p_\text{data}.$$

The velocity of the PF-ODE is directly determined by the drift and score functions of the forward SDE. Starting from the Fokker–Planck equation governing the forward process, we have:

$$\partial_t p_t + \nabla \cdot (b^F_t p_t) = \frac{g_t^2}{2}\,\Delta p_t$$

$$\partial_t p_t + \nabla \cdot (b^F_t p_t) = \frac{g_t^2}{2}\,\nabla \cdot (\nabla p_t)$$

$$\partial_t p_t + \nabla \cdot \left(b^F_t p_t - \frac{g_t^2}{2}\nabla p_t\right) = 0$$

$$\partial_t p_t + \nabla \cdot \left(b^F_t p_t - \frac{g_t^2}{2} p_t \nabla \log p_t\right) = 0$$

$$\partial_t p_t + \nabla \cdot \left(\left[\underbrace{b^F_t - \tfrac{g_t^2}{2}\nabla \log p_t}_{v_t}\right] p_t\right) = 0.$$

From this, we can identify the velocity function $$v_t$$ in the continuity equation and directly obtain the PF-ODE:

$$dx_t = \left[b^F_t(x_t) - \tfrac{1}{2} g_t^2 \nabla_{x_t}\log p_t(x_t)\right] dt.$$

Since the Probability Flow ODE (PF-ODE) is an ordinary differential equation, both the forward and backward processes are governed by the same equation. To simulate the forward process (noising), we integrate the ODE forward in time, starting from the initial distribution and progressing toward the prior distribution. Conversely, the backward process requires integrating the same ODE in reverse, starting from the noise distribution and moving backward in time toward the data distribution ([Fig. 6](#fig-6)).

Because it is deterministic, the PF-ODE has been widely used in contexts where we need to *encode* information. Specifically, the forward PF-ODE maps each data point $$x_0 \sim p_\text{data}$$ to a unique *latent noise* $$x_1$$. By construction, the backward PF-ODE can then *decode* this latent back to the original $$x_0$$ ([Fig. 7](#fig-7)). It is worth noting that, unlike VAEs {% cite vae-encoding_2014 %} and GANs, these "latent" representations have the same dimensionality as the original data; no compression is performed. The process simply maps the data distribution to a Gaussian noise distribution. As a result, this latent space may be less informative compared to the compressed representations learned by VAEs or GANs.

Interestingly, several studies have shown that the PF-ODE better preserves semantic information during the noising process than its stochastic SDE counterpart {% cite song2021denoising cao_2023_masactrl %}. Furthermore, the PF-ODE maps semantically similar data points to nearby regions in the prior distribution. This property makes it particularly well suited for editing tasks, often referred to as *DDIM inversion* in the discrete-time setting, for images {% cite cao_2023_masactrl wave %}, audio signals {% cite nils_demerle_2024_14877437 %}, and weather forecasting {% cite andrae2025continuous %}, as well as for performing smooth interpolations in the prior space {% cite zheng2024noisediffusion rouard2021crash guo2024smooth %}.

Although PF-ODE trajectories are smoother than those produced by the forward and backward SDEs, they are curved. As a result, the PF-ODE does not implement quadratic optimal transport between $$p_0$$ and $$p_1$$ in Euclidean space, whose geodesics, when a Monge map exists, are straight lines and optimal with respect to the $$L^2$$ norm {% cite LavSan22 kim2025simple liu2022rectified %}.

### Learning the Score

We have seen that the *score function* appears as an unknown term in both the stochastic and deterministic backward processes. The central idea underlying diffusion models is therefore to learn a neural approximation of this score function.

In practice, the score is intractable in most settings: it does not admit a closed-form expression, which prevents direct regression against the true score. Nevertheless, the score function admits an alternative interpretation as an expectation over conditional score functions:

$$\nabla_{x_t} \log p_t(x_t) = \mathbb{E}_{x_0 \sim p(x_0 \mid x_t)} \big[ \nabla_{x_t} \log p_t(x_t \mid x_0) \big],$$

which correspond to the gradients of the log-density of corrupted data conditioned on the original sample $$x_0$$.

This observation motivates the design of forward stochastic processes that gradually corrupt data into noise while remaining simple enough for the conditional distribution $$p(x_t \mid x_0)$$ to admit a closed-form expression. Under this assumption, the conditional score $$\nabla_{x_t} \log p_t(x_t \mid x_0)$$ has an analytical form, providing a tractable supervision signal for training neural networks.

Concretely, training proceeds by sampling a clean data point $$x_0 \sim p(x_0)$$, corrupting it with the forward process to obtain $$x_t \sim p(x_t \mid x_0)$$ at a randomly chosen time $$t \sim U(0,1)$$, and regressing a neural network $$s_\theta(t, x_t)$$ toward the corresponding conditional score. This leads to the following regression objective, namely the *denoising score matching* objective {% cite vincent %}:

$$\mathcal{L}(\theta) = \mathbb{E}_{\substack{t \sim U(0, 1) \\ (x_0, x_t) \sim p(x_t \mid x_0)p(x_0)}} \Big[ w(t)\, \lVert s_\theta(t, x_t) - \nabla_{x_t} \log p_t(x_t \mid x_0) \rVert_2^2 \Big].$$

As shown in the objective above, learning the true score function from conditional scores requires sampling corrupted data points $$x_t \sim p(x_t \mid x_0)$$, which in turn amounts to simulating the forward process from time $$0$$ to time $$t$$. The function $$w(t)$$ is a time-dependent weighting term that balances contributions across noise levels {% cite kingma2023understanding karras2022elucidating %}.

As discussed above, for appropriate choices of the scaling and noising schedules $$b_t^F$$ and $$g_t$$, the forward process admits a simple linear interpolation between data and noise,

$$x_t = \alpha_t x_0 + \sigma_t x_1,$$

where $$x_0$$ denotes the unknown clean data sample and $$x_1 \sim p_\text{prior}$$ the injected noise. Under this construction, the conditional score admits a closed-form expression. However, it depends explicitly on either $$x_0$$ or $$x_1$$, which are not available at inference time and must therefore be approximated by a neural network. For instance, in the $$x_0$$-prediction (denoising) parameterization, the conditional score is given by

$$\nabla_{x_t}\log p(x_t \mid x_0) = \frac{1}{\sigma_t^2}\bigl(\alpha_t \,x_0 - x_t\bigr),$$

where the clean sample $$x_0$$ is unknown and replaced in practice by a learned estimator $$x_0^\theta(t, \cdot)$$. Alternatively, in the noise-prediction formulation, the score can be written as

$$\nabla_{x_t}\log p(x_t \mid x_0) = -\frac{1}{\sigma_t}\,x_1,$$

where the noise realization $$x_1$$ must again be predicted by a time-conditional neural network $$x_1^\theta(t, \cdot)$$ solely from the noisy input $$x_t$$. As a consequence, several equivalent diffusion training objectives exist, differing only in their parameterization. Depending on the chosen formulation, the model is trained to predict the clean data sample $$x_0$$, a velocity-like variable, or the injected noise $$x_1 \sim p_\text{prior}$$. Among these alternatives, noise prediction is the most commonly adopted in practice due to its simplicity and numerical stability, and the neural-based noise estimator $$x_1^\theta$$ is trained with a simple regression objective:

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim U(0, 1)}\mathbb{E}_{x_0 \sim p_\text{data}}\mathbb{E}_{x_1 \sim p_\text{prior}} \Big[ \tilde{w}(t)\, \lVert x_1^\theta(t, x_t) - x_1 \rVert_2^2 \Big],$$

where $$\tilde{w}(t)$$ is a positive weighting function derived from the initial $$w(t)$$.

### Generation

The generation of novel data points is carried out by initially sampling $$x_1$$ from the Gaussian prior distribution $$p_\text{prior}$$. Starting from this initial condition, we apply numerical integration schemes to solve either the reverse-time stochastic differential equation or, alternatively, the probability flow ordinary differential equation, evolving the system backward from $$t = 1$$ to $$t = 0$$. In practice, this procedure can be implemented using a variety of sampling algorithms, including Euler–Maruyama discretizations, predictor–corrector samplers, and deterministic ODE solvers, as commonly employed in score-based diffusion models {% cite song2021scorebased song2020generative ho2020denoising %}.

### Likelihood Computation

We have seen that the probability-flow ODE (PF-ODE) provides a deterministic mapping that encodes an initial data point $$x_0$$ into a noise sample $$x_1$$. Since the prior distribution of $$x_1$$ is simple, and typically a standard Gaussian, its likelihood is straightforward to evaluate. A tempting, yet incorrect, idea would then be to treat the likelihood of the noise $$x_1$$ as a proxy for the likelihood of the original data point $$x_0$$.

The flaw in this reasoning lies in the fact that the transport induced by the PF-ODE continuously alters probability densities. In other words, while the mapping from $$x_0$$ to $$x_1$$ is deterministic, it does not preserve volume. By "volume," we mean the local density of probability mass in space: PF-ODE trajectories can compress or stretch regions, so a small neighborhood around $$x_0$$ may occupy a larger or smaller region around $$x_1$$. As a result, likelihoods are not invariant along the trajectory. This change in density can be exactly estimated using the *continuous change-of-variables* formula, allowing us to compute the exact likelihood of $$x_0$$.

Recall that the PF-ODE is governed by the velocity field

$$v_t(x_t) = b^F_t(x_t) - \frac{1}{2} g_t^2 \nabla_{x_t} \log p_t(x_t),$$

which defines how particles evolve over time. Importantly, the log-density $$\log p_t(x_t)$$ itself evolves according to a simple differential equation:

$$\frac{d}{dt} \log p_t(x_t) = - \nabla_x \cdot v_t(x_t).$$

As a consequence, the state variable $$x_t$$ and its log-density $$\log p_t(x_t)$$ can be evolved jointly by solving the following coupled system of ordinary differential equations:

$$\frac{d}{dt} \begin{bmatrix} x_t \\[6pt] \log p_t(x_t) \end{bmatrix} = \begin{bmatrix} v_t(x_t) \\[6pt] -\nabla_x \cdot v_t(x_t) \end{bmatrix}.$$

By integrating this system from $$t=0$$ to $$t=1$$, we simultaneously obtain the encoded noise sample $$x_1$$ and the exact log-likelihood of the original data point. In particular, the likelihood of $$x_0$$ can be expressed as

$$\log p_0(x_0) = \log p_1(x_1) + \int_0^1 \nabla_x \cdot v_t(x_t)\, dt,$$

where the integral term precisely accounts for the accumulated change in density along the deterministic PF-ODE trajectory.

---

## IV. Flow Matching

### Motivation

Diffusion models have demonstrated remarkable performance in high-quality data generation and have emerged as powerful generative modeling frameworks. However, they rely on stochastic differential equations to define both the forward and backward processes, which imposes several structural constraints. In particular, the terminal distribution $$p_1$$ is typically required to be Gaussian.

The flow matching framework {% cite lipman2022flow albergo2022building peluchetti2023non %} offers an alternative approach by directly learning a transport map between two arbitrary distributions. This is achieved by leveraging simple interpolations between pairs of samples $$x_0 \sim p_0$$ and $$x_1 \sim p_1$$ during training. We focus here on the generative setting, where $$p_\text{prior}$$ is a Gaussian distribution. This setting is commonly referred to as *Gaussian flow matching*.

Despite its conceptual similarity to diffusion models, an important distinction is that the direction of time is reversed: in flow matching for generation, samples are transported from noise to data, i.e., $$p_0 = p_\text{prior}$$ and $$p_1 = p_\text{data}$$.

### Flow

The goal of flow matching is to learn a function $$\phi : [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$$, called the *flow*, which transports data points sampled from the initial distribution $$p_0$$ to intermediate distributions $$p_t$$ over time. We denote by $$\phi_t(x_0)$$ the position at time $$t$$ of a particle that started at $$x_0$$. From a probabilistic perspective, the distribution $$p_t$$ corresponds to the distribution of transported particles and can be written as:

$$p_t = [\phi_t]_{\#} p_0,$$

where $$[\phi_t]_{\#}$$ denotes the push-forward operator acting on probability measures {% cite santambrogio2015optimal %}.

Rather than learning the flow $$\phi_t$$ directly, the flow matching framework parameterizes it through a time-dependent velocity field $$v : [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$$, which specifies, for every position $$x \in \mathbb{R}^d$$ and time $$t \in [0,1]$$, the instantaneous direction and speed of particles evolving under the flow. The flow $$\phi_t$$ is then induced by this velocity field through the following ordinary differential equation:

$$\begin{cases} \dfrac{d}{dt}\phi_t(x) = v_t(\phi_t(x)), \\[6pt] \phi_0(x) = x. \end{cases}$$

This ordinary differential equation defines how particles evolve under the velocity field $$v_t$$: starting from their initial position $$x$$ at time $$t=0$$, they follow the trajectories induced by $$v_t$$ and are transported toward the target distribution at $$t=1$$.

A natural training strategy is therefore to learn a neural network $$v_\theta$$, parameterized by $$\theta$$, that approximates the true (unknown) velocity field $$v_t$$. This leads to the following regression objective:

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim U(0,1)}\, \mathbb{E}_{x_t \sim p_t(x_t)} \Big[ \lVert v_\theta(t, x_t) - v_t(x_t) \rVert^2 \Big].$$

If this objective were minimized exactly, the flow induced by the learned velocity field $$v_\theta$$ would define a valid transport between the distributions $$p_0$$ and $$p_1$$. However, this objective is not directly tractable in practice. Indeed, the intermediate distributions $$p_t$$ are not explicitly defined; many choices of $$p_t$$ can satisfy the flow equation, and the true velocity field $$v_t$$ is unknown and does not admit a closed-form expression.

This difficulty mirrors a similar challenge encountered in diffusion models, where the score function is also intractable and is instead learned through regression on conditional scores. Analogously, flow matching replaces the regression on true velocities with a regression on *conditional velocities*, which can be computed analytically under suitable assumptions.

### Interpolants and Couplings

<div id="fig-8" class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/related/flow_matching_vectors.png" class="img-fluid rounded" zoomable=true %}
    </div>
</div>
<div class="caption">
    <strong>Figure 8.</strong> Illustration of the flow matching training process. Unlike stochastic diffusion processes, flow matching leverages interpolation between points of the initial and final distribution. The conditional velocity \(v_t(x_t \mid x_0, x_1)\) is then approximated with a neural network \(v_\theta(t, x_t)\).
</div>

Flow matching models construct the intermediate points $$x_t$$, and implicitly the distributions $$p_t$$, by interpolating between samples drawn from the endpoint distributions $$p_0$$ and $$p_1$$ ([Fig. 8](#fig-8)). A simple and commonly used choice is the linear interpolant $$I_t$$:

$$x_t = I_t(x_0, x_1) = (1 - t) x_0 + t x_1.$$

Sampling from the conditional distribution $$p(x_t \mid x_0, x_1)$$ then amounts to first sampling a pair $$(x_0, x_1)$$ from a joint distribution $$p(x_0, x_1)$$ referred to as a *coupling*, and subsequently computing the interpolated point. The simplest choice is the *independent coupling*, where $$x_0$$ and $$x_1$$ are sampled independently, i.e., $$p(x_0, x_1) = p(x_0)p(x_1)$$. More sophisticated couplings have been proposed to exploit additional structure or prior knowledge.

### Conditional Velocity

In the case of linear interpolants, the *conditional velocity* admits a closed-form expression. It is given by the time derivative of the interpolation path connecting the two endpoints $$x_0$$ and $$x_1$$:

$$v_t(x_t \mid x_0, x_1) = \frac{d}{dt} I_t(x_0, x_1) = \frac{d}{dt} \big[(1 - t) x_0 + t x_1\big] = x_1 - x_0.$$

This expression highlights that, conditionally on $$(x_0, x_1)$$, the induced trajectory is a straight line with constant velocity.

### Loss Function

As discussed above, directly regressing the marginal (or true) velocity field $$v_t(x_t)$$ is intractable, since neither the intermediate distribution $$p_t$$ nor the true velocity field are available in closed form. However, Lipman et al. {% cite lipman2022flow %} show that minimizing a regression objective on conditional velocities is sufficient: any minimizer of the conditional velocity regression problem is also a minimizer of the marginal velocity regression problem.

Consequently, the *conditional flow matching* objective used to train a neural network $$v_\theta$$ to approximate the marginal velocity field is defined as:

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t\sim U(0, 1)}\mathbb{E}_{x_t\sim p_t(x_t\mid x_0, x_1)}\Big\lVert v_\theta(t, x_t) - v_t(x_t\mid x_0, x_1) \Big\rVert^2.$$

For linear interpolants, this objective reduces to:

$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t\sim U(0, 1)}\mathbb{E}_{x_t\sim p_t(x_t\mid x_0, x_1)}\Big\lVert v_\theta(t, x_t) - (x_1 - x_0) \Big\rVert^2.$$

---

## V. Flow vs. Diffusion

Flow matching and diffusion models share strong conceptual connections and, under specific noise schedules and carefully chosen probability-flow ODE solvers, can even become mathematically equivalent {% cite gao2025diffusionmeetsflow %}.

Despite their close relationship, the two frameworks differ primarily in the nature of the process used to bridge the two distributions $$p_0$$ and $$p_1$$ at training time. Flow matching solely relies on interpolations between the two distributions, which provides greater flexibility in the formulation, whereas diffusion models explicitly rely on diffusion processes.

Moreover, the flow matching framework places explicit emphasis on the choice of the coupling $$p(x_0, x_1)$$ between the endpoint distributions and provides a geometric intuition about the transport with the velocity formulation. In contrast, diffusion models are fundamentally formulated in terms of probability densities, and the score function they learn is directly derived from the marginal distribution $$p_t(x_t)$$. This probabilistic formulation makes it particularly natural to incorporate additional conditioning information at generation time. Indeed, by applying Bayes' theorem, the conditional score can be decomposed as

$$\underbrace{\nabla_x \log p(x \mid y)}_{\text{conditional score}} = \underbrace{\nabla_x \log p(x)}_{\text{unconditional score}} + \underbrace{\nabla_x \log p(y \mid x)}_{\substack{\text{measurement matching term} \\ \text{or classifier guidance}}}.$$

This decomposition lies at the core of several conditioning strategies in diffusion models. In particular, it enables *classifier guidance* methods {% cite dhariwal2021diffusion %}, as well as a wide range of techniques for solving inverse problems using diffusion-based generative models {% cite daras2024survey rozet2024learning chung2023diffusion kawar2022denoising %}.

---

## VI. Are Flow Matching Sampling Trajectories Straight?

A common misconception is that flow matching models inherently produce straight sampling trajectories and are therefore more powerful than diffusion models, which are often associated with noisy and curved paths. While it is true that flow matching sampling is deterministic and noise-free, the same property also holds for probability-flow ODE (PF-ODE) sampling in diffusion models.

Regarding the straightness of the trajectories, the confusion arises from two technical considerations:

1. The training objective in flow matching is formulated as a regression on the *conditional velocity field* $$v_t(x_t \mid x_0, x_1)$$, which corresponds to straight paths between paired samples $$x_0$$ and $$x_1$$. However, the neural network does not directly represent these conditional velocities. Instead, it learns an approximation of the *marginal* or *true* velocity field $$v_t(x_t)$$, obtained by averaging over all possible couplings. As a result, the flow induced by this learned velocity field generally follows curved trajectories rather than straight lines.

2. The *Rectified Flow* framework {% cite liu2022rectified %} is often conflated with standard flow matching models using linear interpolants. While rectified flows do tend to produce straighter trajectories, this property does not stem solely from the use of linear interpolation. Rather, it emerges from an iterative refinement procedure that alternates between training flow matching models and generating new pairs $$(x_0, x_1)$$ for subsequent training iterations, similarly to the Iterative Markovian Fitting procedure {% cite shi2024diffusion %}. In the absence of these rectification steps, standard flow matching models still produce curved trajectories.

---

## References

<div style="font-size: 0.85em;">
<div class="post-bibliography">
{% bibliography --cited --group_by none --template post_bib %}
</div>
</div>
