
import os
import sys

os.environ["JAX_TRACEBACK_FILTERING"] = "off"  # more detailed stack traces

import flax
import jax
import jax.numpy as np
import ml_collections
import optax
import tensorflow as tf
import yaml
from absl import flags, logging
from clu import metric_writers
from flax.core import FrozenDict
from flax.training import checkpoints, common_utils, train_state
from ml_collections import config_flags
from tqdm import trange

import wandb
from datasets import load_data
from models.train_utils import create_input_iter, param_count
from models.train_utils import to_wandb_config
from models.flows import nsf, maf
from models.flows.train_utils import loss_flows, train_step

logging.set_verbosity(logging.INFO)

# check what devices jax is operating on
print("jax devices: ", jax.devices())

def build_flows(config):
    """Return the normalizing flow model."""
    if config.flows.flows == "neural_spline_flow":
        model = nsf.NeuralSplineFlow(
            n_dim=config.flows.n_dim,
            n_context=config.flows.n_context,
            hidden_dims=config.flows.hidden_dims,
            n_transforms=config.flows.n_transforms,
            activation=config.flows.activation,
            n_bins=config.flows.n_bins,
            range_min=config.flows.range_min,
            range_max=config.flows.range_max,
        )
    elif config.flows.flows == "maf":
        model = maf.MaskedAutoregressiveFlow(
            n_dim=config.flows.n_dim,
            n_context=config.flows.n_context,
            hidden_dims=config.flows.hidden_dims,
            n_transforms=config.flows.n_transforms,
            activation=config.flows.activation,
            unroll_loop=config.flows.unroll_loop,
            use_random_permutations=config.flows.use_random_permutations,
            inverse=config.flows.inverse,
            # rng_key=config.flows.rng_key,  # TODO: add rng_key to config
        )
    else:
        raise ValueError(f"Unknown flow type: {config.flows.flows}")
    return model


def train_flows(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
) -> train_state.TrainState:
    # Set up wandb run
    if config.wandb.log_train and jax.process_index() == 0:
        wandb_config = to_wandb_config(config)
        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            job_type=config.wandb.job_type,
            group=config.wandb.flow_group,
            config=wandb_config,
        )
        wandb.define_metric(
            "*", step_metric="train/step"
        )  # Set default x-axis as 'train/step'
        workdir = os.path.join(workdir, run.group, run.name)

        # Recursively create workdir
        os.makedirs(workdir, exist_ok=True)

    # Dump a yaml config file in the output directory
    with open(os.path.join(workdir, "config.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f)

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )
    # Load the dataset
    train_ds, norm_dict = load_data(
        config.data.dataset_root,
        config.data.dataset_name,
        config.data.n_features,
        config.data.n_particles,
        config.training.batch_size,
        config.seed,
        shuffle=True,
        split="train",
        conditioning_parameters=config.data.conditioning_parameters,
    )
    batches = create_input_iter(train_ds)

    logging.info("Loaded the %s dataset", config.data.dataset_name)

    # Model configuration
    model = build_flows(config)

    rng = jax.random.PRNGKey(config.seed)
    rng, rng_params = jax.random.split(rng)

    # Pass a test batch through to initialize model
    # TODO: Make so we don't have to pass an entire batch (slow)
    _, conditioning_batch, mask_batch = next(batches)
    theta_batch = np.log10(np.sum(mask_batch, axis=2)).reshape(1, -1, 1)
    x_batch = conditioning_batch
    params = model.init(rng, theta_batch[0], x_batch[0])

    logging.info("Instantiated the model")
    logging.info("Number of parameters: %d", param_count(params))

    # Training config and loop
    # Default schedule if not specified
    if not hasattr(config.optim, "lr_schedule"):
        config.optim.lr_schedule = "cosine"

    if config.optim.lr_schedule == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.optim.learning_rate,
            warmup_steps=config.training.warmup_steps,
            decay_steps=config.training.n_train_steps,
        )
    elif config.optim.lr_schedule == "constant":
        schedule = optax.constant_schedule(config.optim.learning_rate)
    else:
        raise ValueError(f"Invalid learning rate schedule: {config.optim.lr_schedule}")

    tx = optax.adamw(learning_rate=schedule, weight_decay=config.optim.weight_decay)

    # Check if config.optim.grad_clip exists, if so add gradient clipping
    if hasattr(config.optim, "grad_clip"):
        if config.optim.grad_clip is not None:
            tx = optax.chain(
                optax.clip(config.optim.grad_clip),
                tx,
            )

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    logging.info("Starting training...")

    train_metrics = []
    with trange(config.training.n_train_steps) as steps:
        for step in steps:
            _, conditioning, mask = next(batches)
            theta_batch = np.log10(np.sum(mask_batch, axis=2)).reshape(1, -1, 1)
            x_batch = conditioning_batch

            batch = (theta_batch[0], x_batch[0])
            state, metrics = train_step(state, batch, model, loss_flows)

            steps.set_postfix(val=metrics["loss"])
            train_metrics.append(metrics)

            # Log periodically
            if (
                (step % config.training.log_every_steps == 0)
                and (step != 0)
                and (jax.process_index() == 0)
            ):
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f"train/{k}": v
                    for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
                }

                writer.write_scalars(step, summary)
                train_metrics = []

                if config.wandb.log_train:
                    wandb.log({"train/step": step, **summary})

            # Eval periodically
            if (
                (step % config.training.eval_every_steps == 0)
                and (step != 0)
                and (jax.process_index() == 0)
                and (config.wandb.log_train)
            ):
                # pass for now because the eval code assumes Omega_m and sigma_8
                # as conditioning parameters
                pass

            # Save checkpoints periodically
            if (
                (step % config.training.save_every_steps == 0)
                and (step != 0)
                and (jax.process_index() == 0)
            ):
                checkpoints.save_checkpoint(
                    ckpt_dir=workdir,
                    target=state,
                    step=step,
                    overwrite=True,
                    keep=np.inf,
                )

    logging.info("All done! Have a great day.")

    return state


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
    train_flows(config=FLAGS.config, workdir=FLAGS.config.wandb.workdir)
