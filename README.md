# Synthetic Trajectory Generation Through Convolutional Neural Networks

Paper artifacts for PST'24 paper "Synthetic Trajectory Generation Through Convolutional Neural Networks".

## Table of Contents

<!-- TOC -->
- [Synthetic Trajectory Generation Through Convolutional Neural Networks](#synthetic-trajectory-generation-through-convolutional-neural-networks)
  - [Table of Contents](#table-of-contents)
  - [Citation](#citation)
  - [Abstract](#abstract)
  - [Quick Start](#quick-start)
  - [Setup](#setup)
    - [Environment](#environment)
      - [Conda](#conda)
      - [Pip venv](#pip-venv)
    - [Datasets](#datasets)
      - [Foursquare NYC](#foursquare-nyc)
      - [Geolife](#geolife)
  - [Usage](#usage)
    - [Run Evaluation Framework](#run-evaluation-framework)
      - [Evaluation Framework Syntax](#evaluation-framework-syntax)
      - [Configuration Files](#configuration-files)
      - [Concrete Cases](#concrete-cases)
    - [Plotting Results](#plotting-results)
      - [Paper Figures](#paper-figures)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)
  - [References](#references)
  - [Licence](#licence)
<!-- TOC -->


## Citation

[To be added] 

Accepted at the [21st Annual International Conference on Privacy, Security, and Trust (PST2024)](https://pstnet.ca/).

## Abstract

This repository contains the artifacts for the paper
"Synthetic Trajectory Generation Through Convolutional Neural Networks" with the following abstract:

> Location trajectories provide valuable insights for applications from urban planning to pandemic control.
> However, mobility data can also reveal sensitive information about individuals, such as political opinions, religious beliefs, or sexual orientations.
> Existing privacy-preserving approaches for publishing this data face a significant utility-privacy trade-off.
> Releasing synthetic trajectory data generated through deep learning offers a promising solution.
> Due to the trajectories' sequential nature, most existing models are based on recurrent neural networks (RNNs).
> However, research in generative adversarial networks (GANs) largely employs convolutional neural networks (CNNs) for image generation.
> This discrepancy raises the question of whether advances in computer vision can be applied to trajectory generation.
> In this work, we introduce a Reversible Trajectory-to-CNN Transformation (RTCT) that adapts trajectories into a format suitable for CNN-based models.
> We integrated this transformation with the well-known DCGAN in a proof-of-concept (PoC) and evaluated its performance against an RNN-based trajectory GAN using four metrics across two datasets.
> The PoC was superior in capturing spatial distributions compared to the RNN model but had difficulty replicating sequential and temporal properties.
> Although the PoC's utility is not sufficient for practical applications, the results demonstrate the transformation's potential to facilitate the use of CNNs for trajectory generation, opening up avenues for future research.
> To support continued research, all source code has been made available under an open-source license.

## Quick Start

The following section [Setup](#setup) describes how to set up the environment and datasets.
However, if you want to get started quickly, you can use the following script and skip straight to [Usage](#usage):

```shell
./setup.sh
```

This script creates a venv called 'conv_gan', and installs all the necessary packages.
To activate the created virtual environment, use the following command:

```shell
# Following command is necessary for cuda to be found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
${HOME}/.virtualenvs/conv_gan/bin/activate
```

If you want to use conda instead, you can use the following command:

```shell
./setup_conda.sh

# Activate the environment
conda activate conv_gan
```

## Setup

### Environment

#### Conda

To set up the environment with conda, the environment file `env.yml` can be used.

```shell
conda env create -f env.yml -n conv_gan
conda activate conv_gan
```

To activate the environment, use the following command:

```shell
conda activate conv_gan
```

And to deactivate the environment, use the following command:

```shell
conda deactivate
```

#### Pip venv
To set up the environment with pip, the requirements file `requirements.txt` can be used.

A virtual environment can be created like this:

```shell
VENV_NAME='venv'  # Change this to any name you prefer.
python -m venv $VENV_NAME
# Following command is necessary for cuda to be found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
source $VENV_NAME/bin/activate
pip install -r requirements.txt
```

To activate the virtual environment, use the following command:

```shell
source $VENV_NAME/bin/activate
# e.g., source venv/bin/activate for the above example
```

### Datasets

This repository makes use of the Geolife and the Foursquare NYC datasets.
Both datasets need to be present for the code to run.

#### Foursquare NYC
The FourSquare NYC dataset is included in this repository in its pre-processed form,
such that it can be used out of the box.
The Foursquare dataset is contained in the following file:
[restricted_forsquare.csv](data/fs_nyc/restricted_forsquare.csv).

The dataset is originally from the LSTM-TrajGAN repository [3].
Compared to that version, we have made the following changes:

- Restricted all trajectories to those within the following parameters:
  - max_lat = 40.9883
  - min_lat = 40.5508
  - max_lon = -73.6857
  - min_lon = -74.2696

#### Geolife

The Geolife dataset is also included in this repository in its pre-processed form, and can be found in [restriced_geolife.csv](data/geolife/restricted_geolife.csv).
Use `./download_geolife.sh` to download the original dataset and pre-process it yourself.


## Usage

### Run Evaluation Framework

#### Evaluation Framework Syntax

The evaluation framework is run using the following syntax,
```shell
python3 -m evaluation_framework.eval <model> <dataset> -c configs/<model>_<dataset>_<architecture>.json --gpu <gpu_num>
```

 Parameter Details
```shell
model: cnngan|ntg

Which model you are evaluating.

dataset: geolife|fs

Which dataset you are evaluating.

architecture: gan|wgan

If you are evaluating the normal GAN or Wasserstein GAN implementation.

gpu_num:0-9+

Which GPU you will use to evaluate.
```

#### Configuration Files

To keep experiments consistent, we used configuration files with the following parameters,
```python
{
  "file":"fs", # Which dataset to use
  "epochs":257, # How many epochs to run the experiment for.
  "batch_size":64, # The size of batches in each iteration.
  "lr":0.0002, # The learning rate of the model.
  "g_factor":1.0,  # The generator factor, our implementation for the Two Time Update Rule.
  "b1":0.5,  # First beta hyperparameter
  "evaluate":false, # Unused parameter - but have chosen not to omit it as it appears in results.
  "b2":0.999, # Second beta hyperparameter
  "n_cpu":8, # Number of cpu threads to use.
  "latent_dim":100,  # Latent Dimension hyperparameter.
  "img_size":24, # The height/width of the input images (the images are square so height and width are the same)
  "channels":4, # How many channels are in the input images.
  "sample_interval":1000, # How often to sample results to be recorded.
  "load_params":0,  # Which Batch to load parameters from (also unused in the current iteration but we have not ommited it)
  "schedule": 4000, # The batch at which the learning rate scheduler triggers.
  "dp_in_dis":false, # Used to change if differential privacy is implemented in the discriminator or generator.
  "plot_points":false, # Used to plot points during model testing.
  "dp":false, # Turn DP on or off.
  "max_grad_norm":1.0, # Maximum gradient norm.
  "wgan":false, # Turning WGAN on or off.
  "gradient_penalty":false, # Turning Gradient Penalty on or off.
  "lp":false, # Using the Lipschitz Penalty in WGAN or not.
  "n_critic":1, # Number of critics for DP
  "save_freq": 10 # How many batches between saving parameters.
}
```

#### Concrete Cases

These are the command we used in the paper for Figures 3 and 4 (Also see `gpuX.sh` for more examples):

```shell
# DCGAN on Foursquare
python3 -m evaluation_framework.eval cnngan fs -c configs/cnngan_fs_gan.json --gpu 0
# DCGAN on Geolife
python3 -m evaluation_framework.eval cnngan geolife -c configs/cnngan_geolife_wgan.json --gpu 1
# NTG on Foursquare
python3 -m evaluation_framework.eval ntg fs -c configs/noise_tg_fs.json --gpu 0
# NTG on Geolife
python3 -m evaluation_framework.eval ntg geolife -c configs/noise_tg_geolife.json --gpu 1
# DP-DCGAN on Foursquare
python3 -m evaluation_framework.eval cnngan fs -c configs/dp-cnngan_fs_iwgan.json --gpu 0
# DP-DCGAN on Geolife
python3 -m evaluation_framework.eval cnngan geolife -c configs/dp-cnngan_geolife_iwgan.json --gpu 1
# DP-NTG on Foursquare
python3 -m evaluation_framework.eval ntg fs -c configs/dp-noise_tg_fs_wgan.json --gpu 0
# DP-NTG on Geolife
python3 -m evaluation_framework.eval ntg geolife -c configs/dp-noise_tg_geolife_wgan.json --gpu 1
```

Here is a tabular overview of the configuration files

| ID  |  Model   | Dataset | DP Mode | WGAN | LP  | Done | In Paper |            Config             |
| :-: | :------: | :-----: | :-----: | :--: | :-: | :--: | :------: | :---------------------------: |
|  1  |  DCGAN   |   FS    | *None*  |  ❌   |  ❌  |  ✅   |    ✅     |      cnngan_fs_gan.json       |
|  2  |  DCGAN   | Geolife | *None*  |  ❌   |  ❌  |  ✅   |    ✅     |    cnngan_geolife_gan.json    |
|  3  |  DCGAN   |   FS    | *None*  |  ✅   |  ✅  |  ✅   |          |      cnngan_fs_wgan.json      |
|  4  |  DCGAN   | Geolife | *None*  |  ✅   |  ✅  |  ✅   |          |   cnngan_geolife_wgan.json    |
|  5  | DP-DCGAN |   FS    |   Gen   |  ❌   |  ❌  |  ✅   |          |     dp-cnngan_fs_gan.json     |
|  6  | DP-DCGAN | Geolife |   Gen   |  ❌   |  ❌  |  ✅   |          |  dp-cnngan_geolife_gan.json   |
|  7  | DP-DCGAN |   FS    |   Gen   |  ✅   |  ❌  |  ✅   |          |    dp-cnngan_fs_wgan.json     |
|  8  | DP-DCGAN | Geolife |   Gen   |  ✅   |  ❌  |  ✅   |          |  dp-cnngan_geolife_wgan.json  |
|  9  | DP-DCGAN |   FS    |   Gen   |  ✅   |  ✅  |  ✅   |    ✅     |    dp-cnngan_fs_iwgan.json    |
| 10  | DP-DCGAN | Geolife |   Gen   |  ✅   |  ✅  |  ✅   |    ✅     | dp-cnngan_geolife_iwgan.json  |
| 11  |   NTG    |   FS    | *None*  |  ✅   |  ✅  |  ✅   |    ✅     |       noise_tg_fs.json        |
| 12  |   NTG    | Geolife | *None*  |  ✅   |  ✅  |  ✅   |    ✅     |     noise_tg_geolife.json     |
| 13  |  DP-NTG  |   FS    |   Gen   |  ✅   |  ❌  |  ✅   |    ✅     |   dp-noise_tg_fs_wgan.json    |
| 14  |  DP-NTG  | Geolife |   Gen   |  ✅   |  ❌  |  ✅   |    ✅     | dp-noise_tg_geolife_wgan.json |


### Plotting Results

![metrics foursquare](/plots/metrics_fs.png)
![metrics geolife](/plots/metrics_geolife.png)
![progress](/plots/progress.png)
![point_clouds](/plots/point_clouds.png)



#### Paper Figures

The following exact commands were used to generate the figures in the paper:

1. Figure 3: `python -m figure_framework.plot -n results/NTG_geolife_1e-04_5xD1e-04_L256_N28_B64_WGAN-LP -c results/DCGAN_geolife_110_1xD_LR-2e-04_Sched-4000_GFac-1.0_GAN -dn results/DP-NTG_geolife_1e-04_5xD1e-04_L256_N28_B640_WGAN -dc results/DP-DCGAN_geolife_1100_5xD_LR-2e-04_Sched-4000_GFac-1.0_WGAN-LP -o metrics_geolife`
2. Figure 4: `python -m figure_framework.plot -n results/NTG_fs_1e-04_5xD1e-04_L256_N28_B64_WGAN-LP -c results/DCGAN_fs_257_1xD_LR-2e-04_Sched-4000_GFac-1.0_GAN -dn results/DP-NTG_fs_1e-04_5xD1e-04_L256_N28_B640_WGAN -dc results/DP-DCGAN_fs_2570_5xD_LR-2e-04_Sched-4000_GFac-1.0_WGAN-LP -o metrics_fs`
3. Figure 5: See Figure 3 (same command)
4. Figure 6: See Figure 3 (same command)

## Contact

**Authors:** [Jesse Merhi](https://jmerhi.mov) ([jesse.j.merhi@gmail.com](mailto:jesse.j.merhi@gmail.com)) and [Erik Buchholz](https://www.erikbuchholz.de) ([e.buchholz@unsw.edu.au](mailto:e.buchholz@unsw.edu.au))

**Supervision:**

- [Erik Buchholz](https://www.erikbuchholz.de) ([e.buchholz@unsw.edu.au](mailto:e.buchholz@unsw.edu.au))
- [Prof. Salil Kanhere](https://salilkanhere.net/)

**Maintainer E-mail:** [jesse.j.merhi@gmail.com](mailto:jesse.j.merhi@gmail.com) or ([e.buchholz@unsw.edu.au](mailto:e.buchholz@unsw.edu.au))

## Acknowledgements

The authors would like to thank the University of New South Wales,
the Commonwealth of Australia, and the Cybersecurity Cooperative Research Centre Limited, whose activities are partially
funded by the Australian Government’s Cooperative Research Centres Programme, for their support.

## References

This work is based on the following publications (especially [1] and [7]):

[1] Erik Linder-Norén, “Pytorch-GAN.” 2021. [Online]. Available: https://github.com/eriklindernoren/PyTorch-GAN

[2] J. Rao, S. Gao, Y. Kang, and Q. Huang, “LSTM-TrajGAN: A Deep Learning Approach to Trajectory Privacy Protection,” Leibniz International Proceedings in Informatics, vol. 177, no. GIScience, pp. 1–16, 2020, doi: 10.4230/LIPIcs.GIScience.2021.I.12.

_Their code is available at:_

[3] J. Rao, S. Gao, Y. Kang, and Q. Huang, “LSTM-TrajGAN.” GeoDS Lab @UW-Madison, 2020. Accessed: Sep. 25, 2023. [Online]. Available: https://github.com/GeoDS/LSTM-TrajGAN

[4] E. Buchholz, A. Abuadbba, S. Wang, S. Nepal, and S. S. Kanhere, “Reconstruction Attack on Differential Private Trajectory Protection Mechanisms,” in Proceedings of the 38th Annual Computer Security Applications Conference, in ACSAC ’22. New York, NY, USA: Association for Computing Machinery, December 2022, pp. 279–292. doi: 10.1145/3564625.3564628.

_Their code is available at:_

[5] E. Buchholz, S. Abuadbba, S. Wang, S. Nepal, and S. S. Kanhere, “Reconstruction Attack on Protected Trajectories (RAoPT).” [Online]. Available: https://github.com/erik-buchholz/RAoPT

[6] Erik Buchholz, Alsharif Abuadbba, Shuo Wang, Surya Nepal, and Salil S. Kanhere. SoK: Can Trajectory Generation Combine Privacy and Utility?. Proceedings on Privacy Enhancing Technologies, 2024(3), July 2024. 

_Their code is available at:_

[7] E. Buchholz, A. Abuadbba, S. Wang, S. Nepal, and S. S. Kanhere, “SoK: Can Trajectory Generation Combine Privacy and Utility?” [Online]. Available: https://github.com/erik-buchholz/SoK-TrajGen

## Licence

MIT License

Copyright © Cyber Security Research Centre Limited 2024.
This work has been supported by the Cyber Security Research Centre (CSCRC) Limited
whose activities are partially funded by the Australian Government’s Cooperative Research Centres Programme.
We are currently tracking the impact CSCRC funded research. If you have used this code/data in your project,
please contact us at contact@cybersecuritycrc.org.au to let us know.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
