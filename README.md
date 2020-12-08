# DA_Infer
This repository contains the [PyTorch](https://pytorch.org) source code to
reproduce the experiments in our NeurIPS2020 paper [Domain Adaptation as a Problem of Inference on Graphical Models
](https://arxiv.org/abs/2002.03278).

# Main code
* ./code: DA on the learned augmented DAG. Experiments on the wifi localization data.
* ./code_digits: DA on the digit datasets using the causal graph y->x. 

# Citation

```
@article{zhang2020domain,
  title={Domain adaptation as a problem of inference on graphical models},
  author={Zhang, Kun and Gong, Mingming and Stojanov, Petar and Huang, Biwei and Liu, Qingsong and Glymour, Clark},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
