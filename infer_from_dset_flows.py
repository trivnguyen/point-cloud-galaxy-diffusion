
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
    _, _, flows_conditioning, flows_norm_dict = datasets.get_nbody_data(
        config.data.dataset_root,
        config.data.dataset_name,
        config.data.n_features,
        config.data.n_particles,
        conditioning_parameters=config.data.flows_conditioning_parameters + config.data.flows_labels
    )
    x = x * norm_dict['std'] + norm_dict['mean']

    num_conditioning = len(config.data.conditioning_parameters)
    num_flows_conditioning = len(config.data.flows_conditioning_parameters)
    num_flows_labels = len(config.data.flows_labels)


    # load the VDM and the normalizing flows
    logging.info("Loading the VDM and the normalizing flows...")
    path_to_vdm = Path(os.path.join(config.logdir, config.vdm_name))
    vdm, vdm_params = models.diffusion.VariationalDiffusionModel.from_path_to_model(
        path_to_model=path_to_vdm, norm_dict=norm_dict)

    path_to_flows = Path(os.path.join(config.logdir, config.flows_name))
    flows, flows_params = nsf.NeuralSplineFlow.from_path_to_model(
        path_to_model=path_to_flows)

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
    dset = datasets.make_dataloader(
        (x, conditioning, flows_conditioning, mask), batch_size=batch_size,
        seed=42, shuffle=False, repeat=False)
    dset = create_input_iter(dset)

    truth_samples = []
    truth_mask = []
    vdm_samples = []
    vdm_cond = []
    vdm_mask = []
    flows_samples = []
    flows_cond = []

    for batch in tqdm(dset):
        x_batch, cond_batch, flows_cond_batch, mask_batch = batch
        x_batch = jnp.repeat(x_batch[0], num_repeats, axis=0)
        cond_batch = jnp.repeat(cond_batch[0], num_repeats, axis=0)
        flows_cond_batch = jnp.repeat(flows_cond_batch[0], num_repeats, axis=0)
        truth_mask_batch = jnp.repeat(mask_batch[0], num_repeats, axis=0)

        # generate the flow samples
        flows_samples_batch = sample_from_flow(
            flows_cond_batch[:, :num_flows_conditioning], 1,
            jax.random.split(rng, len(cond_batch))).squeeze()

        flows_labels_std = flows_norm_dict['cond_std'][num_flows_conditioning:]
        flows_labels_mean = flows_norm_dict['cond_mean'][num_flows_conditioning:]
        flows_samples_batch = flows_samples_batch * flows_labels_std + flows_labels_mean

        # get the total number of particles
        num_subhalos = 10**flows_samples_batch[..., log_num_subhalos_idx]
        num_subhalos = jnp.clip(num_subhalos, 1, num_particles)
        num_subhalos = jnp.round(num_subhalos).astype(jnp.int32)
        vdm_mask_batch = create_mask(num_subhalos, num_particles)

        # get the new conditioning vector
        if len(flows_to_condition) > 0:
            # make sure that the normalization is working correctly
            cond_batch = cond_batch * norm_dict['cond_std'] + norm_dict['cond_mean']
            cond_batch = cond_batch.at[:, condition_to_flows].set(
                flows_samples_batch[:, flows_to_condition])
            cond_batch = (cond_batch - norm_dict['cond_mean']) / norm_dict['cond_std']

        vdm_samples.append(
            eval.generate_samples(
                vdm=vdm,
                params=vdm_params,
                rng=rng,
                n_samples=len(cond_batch),
                n_particles=num_particles,
                conditioning=cond_batch,
                mask=vdm_mask_batch,
                steps=steps,
                norm_dict=norm_dict,
                boxsize=1,  # doesn't matter
            )
        )
        vdm_mask.append(vdm_mask_batch)
        vdm_cond.append(cond_batch)
        truth_samples.append(x_batch)
        truth_mask.append(truth_mask_batch)
        flows_cond.append(flows_cond_batch)
        flows_samples.append(flows_samples_batch)

    vdm_samples = jnp.concatenate(vdm_samples, axis=0)
    vdm_mask = jnp.concatenate(vdm_mask, axis=0)
    vdm_cond = jnp.concatenate(vdm_cond, axis=0)
    truth_samples = jnp.concatenate(truth_samples, axis=0)
    truth_mask = jnp.concatenate(truth_mask, axis=0)
    flows_samples = jnp.concatenate(flows_samples, axis=0)
    flows_cond = jnp.concatenate(flows_cond, axis=0)

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
        output_path, samples=vdm_samples, cond=vdm_cond, mask=vdm_mask,
        flows_samples=flows_samples, flows_cond=flows_cond,
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
