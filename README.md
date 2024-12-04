
# Bounds in Wasserstein Distance for Locally Stationary Processes
This repository includes the implementation experiments in the paper

> *Bounds in Wasserstein Distance for Locally Stationary Processes*
> 
> by Jan N. Tinio, Mokhtar Z. Alaya and Salim Bouzebda
> arXiv link: arXiv/6042737
> 
## Introduction
A brief introduction about the folders and files:
* `data/`: locally stationary real-world datasets.

* `src/`: methods and implementations.
    * `kernels.py`: 
    * `utils.py`: standard kernels in torch-mode calls.

* `models/`: python files contains all the used illustrated models.

* `notebooks/`: simulated and real data.

## Requirements
Python: > 3.10
Pytorch
Sckit-Learn

## Reproducibility
For synthetic data analysis in Section
* you can run notebooks to reproduce the results.

## Citation
If you use this toolbox in your research and find it useful, please cite:
```
@article{tinioetal2024,
  title={Bounds in Wasserstein Distance for Locally Stationary Processes},
  author={Jan N. Tinio, Mokhtar Z. Alaya and Salim Bouzebda},
  journal={arXiv preprint },
  year={2024}
}
```
