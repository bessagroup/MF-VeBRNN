<!-- <p align="center">
  <a href=""><img alt="logo" src="https://avatars.githubusercontent.com/u/64279108?s=200&v=4" width="30%"></a>
</p> -->

<p align="center">
  <a href=""><img alt="logo" src="logo/logo.png" width="40%"></a>
</p>

# What is MF-VeBRNN?

| [**GitHub**](https://github.com/bessagroup/MF-VeBRNN.git)
| [**Paper**](https://www.sciencedirect.com/science/article/pii/S0045782525007510) |

## Summary

`MF-VeBRNN` provides the implementation for the paper [Single-to-multi-fidelity history-dependent learning with uncertainty quantification and disentanglement: Application to data-driven constitutive modeling](https://www.sciencedirect.com/science/article/pii/S0045782525007510).



## Statement of need

Data-driven learning is generalized to consider history-dependent multi-fidelity data, while quantifying epistemic uncertainty and disentangling it from data noise (aleatoric uncertainty). This generalization is hierarchical and adapts to different learning scenarios: from training the simplest single-fidelity deterministic neural networks up to the proposed multi-fidelity variance estimation Bayesian recurrent neural networks. The proposed methodology is demonstrated by applying it to different data-driven constitutive modeling scenarios for history-dependent plasticity of elastoplastic biphasic materials that include multiple fidelities with and without aleatoric uncertainty (noise). The method accurately predicts the response and quantifies model error while also discovering the noise distribution (when present). The versatility and generality of the proposed method open opportunities for future real-world applications in diverse scientific and engineering domains; especially, the most challenging cases involving design and analysis under uncertainty.

- a **Variance estimation network** (aleatoric uncertainty)
- a **Bayesian neural network** (epistemic uncertainty)

They iteratively refine each other, resulting in:

- disentangled **aleatoric & epistemic** uncertainty
- improved predictive accuracy
- stable training without ad-hoc tricks

> A visualization of the VeBNN training procedure is given as follows:


<div align="center">
    <img src="logo/illustration.png" alt="VeBNN" width="800"/>
</div>

---

**Authorship**:
- This repo is developed [Jiaxiang Yi](https://scholar.google.com/citations?user=LM6O83QAAAAJ&hl=en), a PhD candidate of Delft University of Technology, based on his research context.




## Community Support

If you find any **issues, bugs or problems** with this package, please use the [GitHub issue tracker](https://github.com/bessagroup/VeBNN/issues) to report them.

## License

Copyright (c) 2025, Jiaxiang Yi

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/bessagroup/VeBNN/blob/main/LICENSE) for the full license text.

