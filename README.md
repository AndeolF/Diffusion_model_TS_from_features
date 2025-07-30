# CREATION_DATASET_CORRECTION_ARTEFACT

This is a deposit made by Andéol FOURNIER, undergraduate research trainee in the Brain Imaging Centre Laboratory - Neuroimaging and Neuroinformatics Unit at the Montreal Neurological Institute, Faculty of Medicine, McGill University under the supervision of Sylvain Baillet, Phd.

# 📦 Requirements

For a simplified and reproducible setup, it is recommended to use **[Poetry](https://python-poetry.org/)** to manage dependencies and the virtual environment. (files .toml and .lock are located at the root of the project)

---

# Diffusion Model Training for Time Series Generation from Catch22 Features

This repository contains the code for training a **diffusion model** that generates time series data from **Catch22 features**  
([Lubba et al., 2019](https://doi.org/10.1007/s10618-019-00647-x)).

The model is based on and adapted from the implementation of the paper:  
**T2S: High-resolution Time Series Generation with Text-to-Series Diffusion Models**  
Yunfeng Ge *et al.*, 2025 ([paper link](https://arxiv.org/pdf/2505.02417))

---

## 🗂 Repository Structure

### `checkpoints/`

Contains the saved weights for both the **VAE** and **diffusion models** for each training run and dataset.

### `Data/`

Contains the datasets generated using the `creation_dataset_correction_artefact` repository.

### `datafactory.py`

Includes utility functions to load and format datasets for training and evaluation.

### `evaluate/`

Original folder from the **Text2TimeSeries** repository.  
Not used or only minimally used in this project.

### `model/`

Contains architecture definitions for:

- The **VAE** (Variational Autoencoder),
- The **Diffusion Model**,
- Backbone networks used within the model.

---

## 🧪 Core Scripts

### `train.py`

Main training script.  
Training parameters can be adjusted either by modifying the script directly or using command-line arguments.

---

### `pretrained_lavae_unified.py`

Script to pretrain the **VAE** before training the diffusion model.

---

### `clean_the_dataset.py`

Cleans the dataset by removing data samples with extreme values (based on z-score threshold) to improve model training quality.

---

### `get_train_std_latent.py`

Computes and stores the **latent space standard deviation** needed by the diffusion model (used in backbone initialization).

---

### `t_giver.py`

Implements a class to dynamically provide values of **t** (diffusion time steps) that are more challenging for the model — this helps focus training on difficult cases.

---

### `see_denoising_t_importance.py`

Analyzes the loss across different values of **t** during training and generates a corresponding `t_giver` configuration to emphasize difficult steps.

---

### `test_best_filter.py`

Helps visualize and compare different signal filters in Python to select the most appropriate one for preprocessing.

---

### `infer.py`

Runs inference on a few test examples.  
Generates animated `.gif` visualizations of:

- The denoising process across time steps (`t`) for the time series,
- The evolution of the **Power Spectral Density (PSD)**.

---

### `evaluation.py`

Evaluates the trained model on a selected number of batches.

---

## 🔁 Recommended Script Execution Order

1. **Train the VAE** using `pretrained_lavae_unified.py`
2. **Clean the dataset** using `clean_the_dataset.py`
3. **Compute latent space std** with `get_train_std_latent.py`
4. **Train the diffusion model** using `train.py`
5. *(Optional)* Fine-tune the `t_giver` with `see_denoising_t_importance.py` and `t_giver.py` if needed.
