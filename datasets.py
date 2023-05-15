import tensorflow as tf
import time
import jax
import jax.numpy as np
import numpy as vnp
from pathlib import Path
import pandas as pd
from absl import logging

try:
    from jetnet.datasets import JetNet
except ImportError:
    logging.info("JetNet dataset not available.")

EPS = 1e-7


def make_dataloader(x, conditioning, mask, batch_size, seed, shuffle=True):
    n_train = len(x)

    train_ds = tf.data.Dataset.from_tensor_slices((x, conditioning, mask))
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat()

    batch_dims = [jax.local_device_count(), batch_size // jax.device_count()]

    for _batch_size in reversed(batch_dims):
        train_ds = train_ds.batch(_batch_size, drop_remainder=False)

    if shuffle:
        train_ds = train_ds.shuffle(n_train, seed=seed)
    return train_ds


def get_halo_data(data_dir, n_features, n_particles, split: str = "train"):
    x = np.load(data_dir / f"{split}_halos.npy")
    conditioning = np.array(pd.read_csv(data_dir / f"{split}_cosmology.csv").values)
    if n_features == 7:
        x = x.at[:, :, -1].set(np.log10(x[:, :, -1]))  # Use log10(mass)
    x = x[:, :n_particles, :n_features]
    return x, conditioning


def get_nbody_data(
    n_features,
    n_particles,
    split: str = "train",
    normalize: bool = False,
    standarize: bool = True,
):
    DATA_DIR = Path("/n/holyscratch01/iaifi_lab/ccuesta/data_for_sid/")
    x, conditioning = get_halo_data(
        data_dir=DATA_DIR, n_features=n_features, n_particles=n_particles, split=split
    )
    if split == "train":
        x_train = x
    else:
        x_train, _ = get_halo_data(
            data_dir=DATA_DIR,
            n_features=n_features,
            n_particles=n_particles,
            split="train",
        )
    if normalize and standarize:
        raise ValueError("Cannot normalize and standarize at the same time")
    if standarize:
        # Standardize per-feature (over datasets and particles)
        x_mean = x_train.mean(axis=(0, 1))
        x_std = x_train.std(axis=(0, 1))
        x = (x - x_mean + EPS) / (x_std + EPS)
        norm_dict = {"mean": x_mean, "std": x_std}
    elif normalize:
        x_min = x_train.min(axis=(0, 1))
        x_max = x_train.max(axis=(0, 1))
        x = (x - x_min) / (x_max - x_min)
        norm_dict = {"min": x_min, "max": x_max}
    # Finalize
    mask = np.ones((x.shape[0], n_particles))  # No mask
    conditioning = conditioning[:, [0, -1]]  # Select only omega_m and sigma_8
    return x, mask, conditioning, norm_dict


def nbody_dataset(
    n_features,
    n_particles,
    batch_size,
    seed,
    split: str = "train",
    shuffle: bool = True,
    normalize: bool = False,
    standarize: bool = True,
):
    x, mask, conditioning, norm_dict = get_nbody_data(
        n_features,
        n_particles,
        split=split,
        normalize=normalize,
        standarize=standarize,
    )
    ds = make_dataloader(
        x,
        conditioning,
        mask,
        batch_size,
        seed,
        shuffle=shuffle,
    )
    return ds, norm_dict


def jetnet_dataset(
    n_features,
    n_particles,
    batch_size,
    seed,
    jet_type=["q", "g", "t"],
    condition_on_jet_features=True,
    std_particle=(1.0, 1.0, 5.0),
):
    """Return training iterator for the JetNet dataset

    Args:
        n_features (int): Take first `n_features` features from the dataset
        n_particles (int): How many particles (typically 30 or 150)
        batch_size (int): Training batch size
        seed (int): PRNGKey seed
        jet_type (list, optional): List of particle types. Defaults to ["q", "g", "t"].
        condition_on_jet_features (bool, optional): Whether to condition on jet features. Defaults to True. If false, will only condition on jet class.

    Returns:
        train_ds, norm_dict: Training iterator and normalization dictionary (mean, std keys)
    """

    particle_data, jet_data = JetNet.getData(
        jet_type=jet_type, data_dir="./data/", num_particles=n_particles
    )

    # Normalize everything BUT the jet class and number of particles (first and last elements of `jet_data`)
    mean_jet = jet_data[:, 1:-1].mean(axis=(0,))
    std_jet = jet_data[:, 1:-1].std(axis=(0,))
    jet_data[:, 1:-1] = (jet_data[:, 1:-1] - mean_jet + EPS) / (std_jet + EPS)

    if not condition_on_jet_features:
        conditioning = jet_data[:, :1]  # Only keep jet class as conditioning feature
    else:
        conditioning = jet_data[:, :-1]  # Keep everything except number of particles

    # Get mask (to specify varying cardinality) and particle features to be modeled (eta, phi, pT)
    mask = particle_data[:, :, -1] > 0
    particle_data = particle_data[:, :, :n_features]

    # Create a masked array for the data excluding the last feature (mask)
    masked_data = vnp.ma.array(
        particle_data, mask=np.tile(~mask[:, :, None], (1, 1, n_features))
    )

    # Calculate the mean and std of valid particles (axis=(0, 1) to compute mean and std across batches and particles)
    mean_particle = masked_data.mean(axis=(0, 1))
    std_particle = masked_data.std(axis=(0, 1)) / std_particle

    # Normalize valid particles by subtracting the mean and dividing by the standard deviation
    # Fill the masked values with the original data
    normalized_data = (masked_data - mean_particle) / std_particle
    normalized_data = normalized_data.filled(particle_data)

    # Replace the original data with the normalized data, keeping the mask feature unchanged
    x = np.array(normalized_data.data)

    train_ds = make_dataloader(x, conditioning, mask.astype(np.int32), batch_size, seed)

    # Store normalization dictionary
    norm_dict = {
        "mean_jet": mean_jet,
        "std_jet": std_jet,
        "mean_particle": mean_particle.data,
        "std_particle": std_particle.data,
    }
    return train_ds, norm_dict


def load_data(dataset, n_features, n_particles, batch_size, seed, shuffle, split, **kwargs):
    if dataset == "nbody":
        train_ds, norm_dict = nbody_dataset(n_features, n_particles, batch_size, seed, shuffle=shuffle, split=split,)
    elif dataset == "jetnet":
        train_ds, norm_dict = jetnet_dataset(
            n_features, n_particles, batch_size, seed, **kwargs
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    return train_ds, norm_dict
