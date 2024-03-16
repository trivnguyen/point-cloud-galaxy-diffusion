
import sys
import jax
from pathlib import Path
import yaml

from typing import List, Dict

from jax import random
import jax.numpy as np
import numpy as onp
from models.diffusion import VariationalDiffusionModel

import wandb
import matplotlib.pyplot as plt
from ml_collections.config_dict import ConfigDict
from pycorr import TwoPointCorrelationFunction
from models.diffusion_utils import generate
from models.train_utils import create_input_iter
from inference.likelihood import elbo
from datasets import nbody_dataset
from scipy.interpolate import interp1d
from scipy.stats import chi2

import time
from tqdm import tqdm

colors = [
    "lightseagreen",
    "mediumorchid",
    "salmon",
    "royalblue",
    "rosybrown",
]


def plot_pointclouds_2D(
    generated_samples: np.array, true_samples: np.array, idx_to_plot: int = 0
):
    """Plot pointcloud in two dimensions

    Args:
        generated_samples (np.array): samples generated by the model
        true_samples (np.array): true samples
        idx_to_plot (int, optional): idx to plot. Defaults to 0.

    Returns:
        plt.figure: figure
    """
    s = 4
    alpha = 0.5
    color = "firebrick"
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(20, 12),
    )
    ax1.scatter(
        generated_samples[idx_to_plot, :, 0],
        generated_samples[idx_to_plot, :, 1],
        alpha=alpha,
        s=s,
        color=color,
    )
    ax1.set_title("Gen")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2.scatter(
        true_samples[idx_to_plot, :, 0],
        true_samples[idx_to_plot, :, 1],
        alpha=alpha,
        s=s,
        color=color,
    )
    ax1.set_title("Gen")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    return fig


def plot_velocity_histograms(
    generated_velocities: np.array,
    true_velocities: np.array,
    idx_to_plot: List[int],
) -> plt.figure:
    """plot histograms of velocity modulus

    Args:
        generated_velocities (np.array): generated 3D velociteis
        true_velocities (np.array): true 3D velocities
        idx_to_plot (List[int]): idx to plot

    Returns:
        plt.Figure: figure vel hist
    """
    generated_mod = onp.sqrt(onp.sum(generated_velocities**2, axis=-1))
    true_mod = onp.sqrt(onp.sum(true_velocities**2, axis=-1))
    fig, _ = plt.subplots(figsize=(15, 5))
    offset = 0
    for i, idx in enumerate(idx_to_plot):
        true_hist, bin_edges = np.histogram(
            true_mod[idx],
            bins=50,
        )
        generated_hist, bin_edges = np.histogram(
            generated_mod[idx],
            bins=bin_edges,
        )
        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.plot(
            bin_centres + offset,
            true_hist,
            label="N-body" if i == 0 else None,
            color=colors[i],
        )
        plt.plot(
            bin_centres + offset,
            generated_hist,
            label="Diffusion" if i == 0 else None,
            linestyle="dashed",
            color=colors[i],
        )
        offset += onp.max(true_mod)
    plt.legend()
    plt.xlabel("|v| + offset [km/s]")
    plt.ylabel("PDF")
    return fig


def plot_hmf(
    generated_masses: np.array,
    true_masses: np.array,
    idx_to_plot: List[int],
) -> plt.figure:
    """plot halo mass functions

    Args:
        generated_masses (np.array): generated masses
        true_masses (np.array): true masses
        idx_to_plot (List[int]): idx to plot

    Returns:
        plt.Figure: hmf figure
    """
    fig, _ = plt.subplots()
    for i, idx in enumerate(idx_to_plot):
        true_hist, bin_edges = np.histogram(
            true_masses[idx],
            bins=50,
        )
        generated_hist, bin_edges = np.histogram(
            generated_masses[idx],
            bins=bin_edges,
        )
        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.semilogy(
            bin_centres,
            true_hist,
            label="N-body" if i == 0 else None,
            color=colors[i],
        )
        plt.semilogy(
            bin_centres,
            generated_hist,
            label="Diffusion" if i == 0 else None,
            color=colors[i],
            linestyle="dashed",
        )

    plt.legend()
    plt.xlabel("log Halo Mass")
    plt.ylabel("PDF")
    return fig


def eval_likelihood(
    vdm,
    pstate,
    rng,
    true_samples: np.array,
    conditioning: np.array,
    mask: np.array,
    log_wandb: bool = True,
):
    n_test = 16
    omega_m_ary = np.linspace(0.1, 0.5, 30)

    log_like_cov = []
    for idx in tqdm(range(n_test)):
        log_like = []
        x_test = true_samples[idx][None, ...]
        for omega_m in omega_m_ary:
            if len(conditioning[idx]) > 2:
                theta_test = np.array([omega_m] + list(conditioning[idx][1:]))[
                    None, ...
                ]
            else:
                theta_test = np.array([omega_m, conditioning[idx][1]])[None, ...]
            log_like.append(
                elbo(
                    vdm,
                    pstate.params,
                    rng,
                    x_test,
                    theta_test,
                    np.ones_like(x_test[..., 0]),
                    steps=20,
                    unroll_loop=True,
                )
            )
        log_like_cov.append(log_like)
    log_like_cov = np.array(log_like_cov)

    threshold_1sigma = -chi2.isf(1 - 0.68, 1)

    intervals1 = []
    true_values = []

    for ic, idx in enumerate(range(n_test)):
        likelihood_arr = 2 * (
            np.array(log_like_cov[idx]) - np.max(np.array(log_like_cov[idx]))
        )

        # Interpolate to find the 95% limits
        f_interp1 = interp1d(
            omega_m_ary,
            likelihood_arr - threshold_1sigma,
            kind="linear",
            fill_value="extrapolate",
        )
        x_vals = np.linspace(omega_m_ary[0], omega_m_ary[-1], 1000)
        diff_signs1 = np.sign(f_interp1(x_vals))

        # Find where the sign changes
        sign_changes1 = ((diff_signs1[:-1] * diff_signs1[1:]) < 0).nonzero()[0]

        if len(sign_changes1) >= 2:  # We need at least two crossings
            intervals1.append((x_vals[sign_changes1[0]], x_vals[sign_changes1[-1]]))
            true_values.append(conditioning[idx][0])
        else:
            # Optionally handle the case where no interval is found
            pass

    # Plotting true value vs. interval
    fig = plt.figure(figsize=(10, 4))

    for value, (low, high) in zip(true_values, intervals1):
        plt.errorbar(
            value,
            (low + high) / 2.0,
            yerr=[[(low + high) / 2.0 - low], [high - (low + high) / 2.0]],
            fmt="o",
            capsize=5,
            color="k",
        )

    plt.plot([0, 1], [0, 1], color="k", ls="--")

    plt.xlim(0.05, 0.5)
    plt.ylim(0.05, 0.5)

    plt.xlabel("True Value")
    plt.ylabel("Estimated Value and Interval")
    plt.grid(True)
    plt.tight_layout()

    if log_wandb:
        wandb.log({"eval/llprof_Om": wandb.Image(plt)})


def eval_generation(
    vdm,
    pstate,
    rng,
    n_samples: int,
    n_particles: int,
    true_samples: np.array,
    conditioning: np.array,
    mask: np.array,
    norm_dict: Dict,
    steps: int = 500,
    boxsize: float = 1000.0,
    log_wandb: bool = True,
):
    """Evaluate the model on a small subset and log figures and log figures and log figures and log figures

    Args:
        vdm (_type_): diffusion model
        pstate (_type_): model weights
        rng (_type_): random key
        n_samples (int): number of samples to generate
        n_particles (int): number of particles to sample
        true_samples (np.array): true samples
        conditioning (np.array): conditioning of the true samples
        mask (np.array): mask
        norm_dict (Dict): dictionariy with mean and std of the true samples, used to normalize the data
        steps (int, optional): number of steps to sample in diffusion. Defaults to 100.
        boxsize (float, optional): size of the simulation box. Defaults to 1000.0.
    """
    generated_samples = generate_samples(
        vdm=vdm,
        params=pstate.params,
        rng=rng,
        n_samples=n_samples,
        n_particles=n_particles,
        conditioning=conditioning,
        mask=mask,
        steps=steps,
        norm_dict=norm_dict,
        boxsize=boxsize,
    )
    true_samples = true_samples * norm_dict["std"] + norm_dict["mean"]
    true_positions = true_samples[..., :3]
    generated_positions = generated_samples[..., :3]
    if generated_samples.shape[-1] > 3:
        generated_velocities = generated_samples[..., 3:6]
        generated_masses = generated_samples[..., -1]
        true_velocities = true_samples[..., 3:6]
        if generated_samples.shape[-1] > 6:
            true_masses = true_samples[..., -1]
        else:
            generated_masses = None
            true_masses = None
    else:
        generated_velocities = None
        generated_masses = None
        true_velocities = None
        true_masses = None
    fig = plot_2pcf(
        generated_samples=generated_positions,
        true_samples=true_positions,
        boxsize=boxsize,
    )
    if log_wandb:
        wandb.log({"eval/2pcf": fig})
    plt.close()

    if generated_velocities is not None:
        fig = plot_velocity_histograms(
            generated_velocities=generated_velocities,
            true_velocities=true_velocities,
            idx_to_plot=[0, 1, 2],
        )
        if log_wandb:
            wandb.log({"eval/vels": fig})
        plt.close()
        fig = plot_2pcf_rsd(
            generated_positions=onp.array(generated_positions),
            true_positions=onp.array(true_positions),
            generated_velocities=onp.array(generated_velocities),
            true_velocities=onp.array(true_velocities),
            conditioning=onp.array(conditioning),
            boxsize=boxsize,
        )
        if log_wandb:
            wandb.log({"eval/2pcf_rsd": fig})
        plt.close()

    if generated_masses is not None:
        fig = plot_hmf(
            generated_masses=generated_masses,
            true_masses=true_masses,
            idx_to_plot=[
                0,
                1,
                2,
                3,
            ],
        )
        if log_wandb:
            wandb.log({"eval/mass": fig})
        plt.close()


def generate_samples(
    vdm,
    params,
    rng,
    n_samples,
    n_particles,
    conditioning,
    mask,
    steps,
    norm_dict,
    boxsize,
):
    generated_samples = generate(
        vdm,
        params,
        rng,
        (n_samples, n_particles),
        conditioning=conditioning,
        mask=mask,
        steps=steps,
    )
    generated_samples = generated_samples.mean()
    generated_samples = generated_samples * norm_dict["std"] + norm_dict["mean"]
    # make sure generated samples are inside boxsize
    generated_samples = generated_samples.at[..., :3].set(
        generated_samples[..., :3] % boxsize
    )
    return generated_samples


def generate_test_samples_from_model_folder(
    path_to_model: Path,
    steps: int = 500,
    batch_size: int = 20,
    boxsize: float = 1000.0,
    n_test: int = 200,
):
    with open(path_to_model / "config.yaml", "r") as file:
        config = yaml.safe_load(file)
    config = ConfigDict(config)
    # get conditioning for test set
    test_ds, norm_dict = nbody_dataset(
        n_features=config.data.n_features,
        n_particles=config.data.n_particles,
        batch_size=batch_size,
        seed=config.seed,
        shuffle=False,
        split="test",
    )
    return generate_samples_for_dataset(
        ds=test_ds,
        n_particles=config.data.n_particles,
        norm_dict=norm_dict,
        n_total_samples=n_test,
        path_to_model=path_to_model,
        steps=steps,
        batch_size=batch_size,
        boxsize=boxsize,
    )


def generate_samples_for_dataset(
    ds,
    norm_dict,
    n_particles: int,
    n_total_samples: int,
    path_to_model: Path,
    steps: int = 500,
    batch_size: int = 20,
    boxsize: float = 1000.0,
):
    batches = create_input_iter(ds)
    x_batch, conditioning_batch, mask_batch = next(batches)
    vdm, params = VariationalDiffusionModel.from_path_to_model(
        path_to_model=path_to_model
    )
    rng = jax.random.PRNGKey(42)
    n_batches = n_total_samples // batch_size
    true_samples, generated_samples, conditioning_samples = [], [], []
    for i in range(n_batches):
        t0 = time.time()
        x_batch, conditioning_batch, mask_batch = next(batches)
        true_samples.append(x_batch[0] * norm_dict["std"] + norm_dict["mean"])
        generated_samples.append(
            generate_samples(
                vdm=vdm,
                params=params,
                rng=rng,
                n_samples=batch_size,
                n_particles=n_particles,
                conditioning=conditioning_batch[0],
                mask=mask_batch[0],
                steps=steps,
                norm_dict=norm_dict,
                boxsize=boxsize,
            )
        )
        conditioning_samples.append(conditioning_batch[0])
        print(f"Iteration {i} takes {time.time() - t0} seconds")
    return (
        np.array(true_samples),
        np.array(generated_samples),
        np.array(conditioning_samples),
    )
