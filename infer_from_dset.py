
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List

import datasets
import eval
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
import models.diffusion
import numpy as np
import yaml
from absl import flags, logging
from ml_collections.config_dict import ConfigDict
from ml_collections import config_flags
from models.diffusion_utils import generate
from models.flows import maf, nsf
from models.train_utils import create_input_iter
from tqdm import tqdm


@partial(jax.vmap, in_axes=(0, None))
def create_mask(n, num_particles):
    # Create an array [0, 1, 2, ..., num_particles-1]
    indices = jnp.arange(num_particles)
    # Compare each index to n, resulting in True (1) if index < n, else False (0)
    mask = indices < n
    return mask.astype(jnp.float32)


def infer(config: ConfigDict):

    # set the random seed
    rng = jax.random.PRNGKey(config.seed)

    # load the dataset
    logging.info("Loading the dataset...")
    x, mask, conditioning, norm_dict = datasets.get_nbody_data(
        config.data.dataset_root,
        config.data.dataset_name,
        config.data.n_features,
        config.data.n_particles,
        conditioning_parameters=config.data.conditioning_parameters
    )
    x = x * norm_dict['std'] + norm_dict['mean']

    # load the VDM and the normalizing flows
    logging.info("Loading the VDM and the normalizing flows...")
    path_to_vdm = Path(os.path.join(config.logdir, config.vdm_name))
    vdm, vdm_params = models.diffusion.VariationalDiffusionModel.from_path_to_model(
        path_to_model=path_to_vdm, norm_dict=norm_dict)

    if config.flows_name is not None:
        path_to_flows = Path(os.path.join(config.logdir, config.flows_name))
        flows, flows_params = nsf.NeuralSplineFlow.from_path_to_model(
            path_to_model=path_to_flows)
    else:
        flows, flows_params = None, None

    # Create the sampling function based on the normalizing flows
    @partial(jax.vmap, in_axes=(0, None, 0))
    def sample_from_flow(context, n_samples=10_000, key=jax.random.PRNGKey(42)):
        """Helper function to sample from the flow model.
        """
        def sample_fn(flows):
            x_samples = flows.sample(
                num_samples=n_samples, rng=key,
                context=context * jnp.ones((n_samples, 1)))
            return x_samples

        x_samples = nn.apply(sample_fn, flows)(flows_params)
        return x_samples

    # Iterate over the entire dataset and start generation
    truth_samples = []
    truth_mask = []
    gen_samples = []
    gen_cond = []
    gen_mask = []

    dset = datasets.make_dataloader(
        x, conditioning, mask, batch_size=config.batch_size,
        shuffle=False, repeat=False)
    dset = create_input_iter(dset)

    logging.info("Starting generation...")
    for batch in tqdm(dset):
        x_batch, cond_batch, mask_batch = batch[0], batch[1], batch[2]
        x_batch = jnp.repeat(x_batch[0], config.n_repeats, axis=0)
        cond_batch = jnp.repeat(cond_batch[0], config.n_repeats, axis=0)
        truth_mask_batch = jnp.repeat(mask_batch[0], config.n_repeats, axis=0)

        rng, _ = jax.random.split(rng)

        # generate the number of particles using the flows
        if flows is not None:
            num_subhalos = 10**sample_from_flow(
                cond_batch, 1, jax.random.split(rng, len(cond_batch))).squeeze()
            num_subhalos = jnp.clip(num_subhalos, 1, config.data.n_particles)
            num_subhalos = jnp.round(num_subhalos).astype(jnp.int32)
            gen_mask_batch = create_mask(num_subhalos, config.data.n_particles)
        else:
            gen_mask_batch = truth_mask_batch

        # generate using the VDM
        gen_samples.append(
            eval.generate_samples(
                vdm=vdm,
                params=vdm_params,
                rng=rng,
                n_samples=len(cond_batch),
                n_particles=config.data.n_particles,
                conditioning=cond_batch,
                mask=gen_mask_batch,
                steps=config.steps,
                norm_dict=norm_dict,
                boxsize=config.data.box_size,  # doesn't matter
            )
        )
        gen_cond.append(cond_batch)
        gen_mask.append(gen_mask_batch)
        truth_samples.append(x_batch)
        truth_mask.append(truth_mask_batch)

    gen_samples = jnp.concatenate(gen_samples, axis=0)
    gen_cond = jnp.concatenate(gen_cond, axis=0)
    gen_mask = jnp.concatenate(gen_mask, axis=0)
    truth_samples = jnp.concatenate(truth_samples, axis=0)
    truth_mask = jnp.concatenate(truth_mask, axis=0)

    # Save the samples
    if config.output_name is None:
        vdm_base = os.path.basename(config.vdm_name)
        if config.flows_name is not None:
            flows_base = os.path.basename(config.flows_name)
            output_name = f'vdm-flows/{vdm_base}_{flows_base}.npz'
        else:
            output_name = f'vdm/{vdm_base}.npz'
    else:
        output_name = config.output_name
    output_path = os.path.join(config.workdir, output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info("Saving the generated samples to %s", output_path)
    np.savez(
        output_path, samples=gen_samples, cond=gen_cond, mask=gen_mask,
        truth=truth_samples, truth_mask=truth_mask
    )

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )

    # Parse flags
    FLAGS(sys.argv)

    # Ensure TF does not see GPU and grab all GPU memory
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())
    logging.info("JAX total visible devices: %r", jax.device_count())

    # Start training run
    infer(config=FLAGS.config)
