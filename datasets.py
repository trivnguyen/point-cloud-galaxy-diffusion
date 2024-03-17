
from typing import Optional

import os

import tensorflow as tf
import time
import jax
import jax.numpy as np
import numpy as vnp
from pathlib import Path
import pandas as pd
from absl import logging
from models.graph_utils import get_rotated_box

EPS = 1e-7


def make_dataloader(
    x, conditioning, mask, batch_size, seed=None,
    shuffle=True, repeat=True):
    n_train = len(x)

    train_ds = tf.data.Dataset.from_tensor_slices((x, conditioning, mask))
    train_ds = train_ds.cache()
    if repeat:
        train_ds = train_ds.repeat()

    batch_dims = [jax.local_device_count(), batch_size // jax.device_count()]

    for _batch_size in reversed(batch_dims):
        train_ds = train_ds.batch(_batch_size, drop_remainder=False)

    if shuffle:
        train_ds = train_ds.shuffle(n_train, seed=seed)
    return train_ds


def get_nbody_data(
    dataset_root,
    dataset_name,
    n_features,
    n_particles,
    norm_dict=None,
    conditioning_parameters: Optional[list] = None,
    norm_conditioning: bool = False,
):
    # read in the dataset and mask
    x = np.load(os.path.join(dataset_root, f"{dataset_name}_feat.npy"))
    x = x[:, :n_particles, :n_features]
    mask = np.load(os.path.join(dataset_root, f"{dataset_name}_mask.npy"))
    mask = mask[:, :n_particles]

    # read in the conditioning parameters
    if conditioning_parameters is not None:
        conditioning = pd.read_csv(
            os.path.join(dataset_root, f"{dataset_name}_cond.csv")
        )
        conditioning = np.array(conditioning[conditioning_parameters].values)

    # Standardize per-feature (over datasets and particles)
    if norm_dict is None:
        x_mean = x.mean(axis=(0, 1))
        x_std = x.std(axis=(0, 1))
        cond_mean = conditioning.mean(axis=0)
        cond_std = conditioning.std(axis=0)
        norm_dict = {
            "mean": x_mean, "std": x_std,
            "cond_mean": cond_mean, "cond_std": cond_std,
        }
    else:
        x_mean = norm_dict["mean"]
        x_std = norm_dict["std"]
        cond_mean = norm_dict.get("cond_mean", 0)
        cond_std = norm_dict.get("cond_std", 1)
    x = (x - x_mean + EPS) / (x_std + EPS)
    if norm_conditioning:
        conditioning = (conditioning - cond_mean + EPS) / (cond_std + EPS)

    # Finalize
    return x, mask, conditioning, norm_dict

def nbody_dataset(
    dataset_root,
    dataset_name,
    n_features,
    n_particles,
    batch_size,
    seed,
    split: str = "train",
    shuffle: bool = True,
    repeat: bool = True,
    conditioning_parameters: list = None,
    norm_conditioning: bool = False,
):
    x, mask, conditioning, norm_dict = get_nbody_data(
        dataset_root,
        dataset_name,
        n_features,
        n_particles,
        conditioning_parameters=conditioning_parameters,
        norm_conditioning=norm_conditioning,
    )
    ds = make_dataloader(
        x,
        conditioning,
        mask,
        batch_size,
        seed,
        shuffle=shuffle,
        repeat=repeat
    )
    return ds, norm_dict


def load_data(
    dataset_root, dataset_name, n_features, n_particles,
    batch_size, seed, shuffle, split, repeat=True, **kwargs
):
    train_ds, norm_dict = nbody_dataset(
        dataset_root,
        dataset_name,
        n_features,
        n_particles,
        batch_size,
        seed,
        shuffle=shuffle,
        repeat=repeat,
        split=split,
        **kwargs,
    )

    return train_ds, norm_dict


def augment_with_translations(
    x,
    conditioning,
    mask,
    rng,
    norm_dict,
    n_pos_dim=3,
    box_size: float = 1000.0,
):
    rng, _ = jax.random.split(rng)
    x = x * norm_dict["std"] + norm_dict["mean"]

    # Draw N random translations
    translations = jax.random.uniform(
        rng, minval=-box_size / 2, maxval=box_size / 2, shape=(*x.shape[:2], 3)
    )
    x = x.at[..., :n_pos_dim].set(
        (x[..., :n_pos_dim] + translations[..., None, :]) % box_size
    )
    x = (x - norm_dict["mean"]) / norm_dict["std"]
    return x, conditioning, mask


def random_symmetry_matrix(key):
    # 8 possible sign combinations for reflections
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )

    # 6 permutations for axis swapping
    perms = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])

    # Randomly select one sign combination and one permutation
    sign = signs[jax.random.randint(key, (), 0, 8)]
    perm = perms[jax.random.randint(key, (), 0, 6)]

    # Combine them to form the random symmetry matrix
    matrix = np.eye(3)[perm] * sign
    return matrix


def augment_with_symmetries(
    x,
    conditioning,
    mask,
    rng,
    norm_dict,
    n_pos_dim=3,
    box_size: float = 1000.0,
):
    rng, _ = jax.random.split(rng)
    # Rotations and reflections that respect boundary conditions
    matrix = random_symmetry_matrix(rng)
    x = x.at[..., :n_pos_dim].set(np.dot(x[..., :n_pos_dim], matrix.T))
    if x.shape[-1] > n_pos_dim:
        # Rotate velocities too
        x = x.at[..., n_pos_dim : n_pos_dim + 3].set(
            np.dot(x[..., n_pos_dim : n_pos_dim + 3], matrix.T)
        )
    return x, conditioning, mask


def augment_data(
    x,
    conditioning,
    mask,
    rng,
    norm_dict,
    rotations: bool = True,
    translations: bool = True,
    n_pos_dim=3,
    box_size: float = 1000.0,
):
    if rotations:
        x, conditioning, mask = augment_with_symmetries(
            x=x,
            mask=mask,
            conditioning=conditioning,
            rng=rng,
            norm_dict=norm_dict,
            n_pos_dim=n_pos_dim,
            box_size=box_size,
        )
    if translations:
        x, conditioning, mask = augment_with_translations(
            x=x,
            mask=mask,
            conditioning=conditioning,
            rng=rng,
            norm_dict=norm_dict,
            n_pos_dim=n_pos_dim,
            box_size=box_size,
        )
    return x, conditioning, mask
