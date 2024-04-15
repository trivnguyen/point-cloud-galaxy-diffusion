import dataclasses
from typing import Union

from absl import logging
from pathlib import Path

import yaml
import jax
import flax.linen as nn
import jax.numpy as np
import tensorflow_probability.substrates.jax as tfp
from ml_collections.config_dict import ConfigDict
import optax
from flax.training import train_state, checkpoints
from flax.core import FrozenDict

from models.diffusion_utils import variance_preserving_map, alpha, sigma2
from models.diffusion_utils import (
    NoiseScheduleScalar,
    NoiseScheduleFixedLinear,
    NoiseScheduleNet,
)
from models.scores import (
    TransformerScoreNet,
    GraphScoreNet,
)
from models.mlp import MLPEncoder, MLPDecoder
from datasets import load_data

tfd = tfp.distributions


class VariationalDiffusionModel(nn.Module):
    """Variational Diffusion Model (VDM), adapted from https://github.com/google-research/vdm

    Attributes:
      d_feature: Number of features per set element.
      timesteps: Number of diffusion steps.
      gamma_min: Minimum log-SNR in the noise schedule (init if learned).
      gamma_max: Maximum log-SNR in the noise schedule (init if learned).
      antithetic_time_sampling: Antithetic time sampling to reduce variance.
      noise_schedule: Noise schedule; "learned_linear", "linear", or "learned_net"
      noise_scale: Std of Normal noise model.
      noise_mass_scale: Std of Normal noise model for mass.
      d_t_embedding: Dimensions the timesteps are embedded to.
      score: Score function; "transformer", "graph".
      score_dict: Dict of score arguments (see scores.py docstrings).
      n_classes: Number of classes in data. If >0, the first element of the conditioning vector is assumed to be integer class.
      embed_context: Whether to embed the conditioning context.
      use_encdec: Whether to use an encoder-decoder for latent diffusion.
      norm_dict: Dict of normalization arguments (see datasets.py docstrings).
      n_pos_features: Number of positional features, for graph-building etc.
      n_mass_features: Number of mass features.
      scale_non_linear_init: Whether to scale the initialization of the non-linear layers in the noise model.
    """

    d_feature: int = 3
    timesteps: int = 1000
    gamma_min: float = -8.0
    gamma_max: float = 14.0
    antithetic_time_sampling: bool = True
    noise_schedule: str = "linear"  # "linear", "learned_linear", or "learned_net"
    noise_scale: float = 1.0e-3
    noise_mass_scale: float = 1.0e-3
    d_t_embedding: int = 32
    score: str = "transformer"  # "transformer", "graph"
    score_dict: dict = dataclasses.field(
        default_factory=lambda: {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 4,
            "n_heads": 4,
        }
    )
    encoder_dict: dict = dataclasses.field(
        default_factory=lambda: {"d_embedding": 12, "d_hidden": 256, "n_layers": 4}
    )
    decoder_dict: dict = dataclasses.field(
        default_factory=lambda: {"d_hidden": 256, "n_layers": 4}
    )
    n_classes: int = 0
    embed_context: bool = False
    d_context_embedding: int = 32
    use_encdec: bool = True
    norm_dict: dict = dataclasses.field(
        default_factory=lambda: {"x_mean": 0.0, "x_std": 1.0, "box_size": 1000.0}
    )
    n_pos_features: int = 3
    n_vel_features: int = 3
    n_mass_features: int = 1
    scale_non_linear_init: bool = False

    @classmethod
    def from_path_to_model(
        cls,
        path_to_model: Union[str, Path],
        checkpoint_step: int = None,
        norm_dict: dict = None,
    ) -> "VariationalDiffusionModel":
        """load model from path where it is stored

        Args:
            path_to_model (Union[str, Path]): path to model

        Returns:
            Tuple[VariationalDiffusionModel, np.array]: model, params
        """
        with open(path_to_model / "config.yaml", "r") as file:
            config = yaml.safe_load(file)
        config = ConfigDict(config)
        score_dict = FrozenDict(config.score)
        encoder_dict = FrozenDict(config.encoder)
        decoder_dict = FrozenDict(config.decoder)
        if norm_dict is None:
            _, norm_dict = load_data(
                config.data.dataset,
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
        x_mean = tuple(map(float, norm_dict["mean"]))
        x_std = tuple(map(float, norm_dict["std"]))
        norm_dict_input = FrozenDict(
            {"x_mean": x_mean, "x_std": x_std, "box_size": config.data.box_size}
        )
        n_pos_features = config.score.get("n_pos_features", 3)
        n_vel_features = config.score.get("n_vel_features", 3)
        n_mass_features = config.score.get("n_mass_features", 1)
        vdm = VariationalDiffusionModel(
            d_feature=config.data.n_features,
            timesteps=config.vdm.timesteps,
            gamma_min=config.vdm.gamma_min,
            gamma_max=config.vdm.gamma_max,
            noise_schedule=config.vdm.noise_schedule,
            noise_scale=config.vdm.noise_scale,
            noise_mass_scale=config.vdm.noise_mass_scale,
            d_t_embedding=config.vdm.d_t_embedding,
            score=config.score.score,
            score_dict=score_dict,
            encoder_dict=encoder_dict,
            decoder_dict=decoder_dict,
            n_classes=config.vdm.n_classes,
            embed_context=config.vdm.embed_context,
            d_context_embedding=config.vdm.d_context_embedding,
            use_encdec=config.vdm.use_encdec,
            norm_dict=norm_dict_input,
            n_pos_features=n_pos_features,
            n_vel_features=n_vel_features,
            n_mass_features=n_mass_features,
            # scale_non_linear_init = config.vdm.scale_non_linear_init,
        )
        rng = jax.random.PRNGKey(42)
        x_dummy = jax.random.normal(
            rng,
            (
                config.training.batch_size,
                config.data.n_particles,
                config.data.n_features,
            ),
        )
        conditioning_dummy = jax.random.normal(
            rng, (config.training.batch_size, len(config.data.conditioning_parameters)))
        mask_dummy = np.ones((config.training.batch_size, config.data.n_particles))
        _, params = vdm.init_with_output(
            {"sample": rng, "params": rng}, x_dummy, conditioning_dummy, mask_dummy
        )
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.optim.learning_rate,
            warmup_steps=config.training.warmup_steps,
            decay_steps=config.training.n_train_steps,
        )
        tx = optax.adamw(learning_rate=schedule, weight_decay=config.optim.weight_decay)
        if hasattr(config.optim, "grad_clip"):
            if config.optim.grad_clip is not None:
                tx = optax.chain(
                    optax.clip(config.optim.grad_clip),
                    tx,
                )

        state = train_state.TrainState.create(apply_fn=vdm.apply, params=params, tx=tx)
        # Training config and state
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=path_to_model,
            target=state,
            step=checkpoint_step,
        )
        if state is restored_state:
            raise FileNotFoundError(f"Did not load checkpoint correctly")
        return vdm, restored_state.params

    def setup(self):
        # Noise schedule for diffusion
        if self.noise_schedule == "linear":
            self.gamma = NoiseScheduleFixedLinear(
                gamma_min=self.gamma_min, gamma_max=self.gamma_max
            )
        elif self.noise_schedule == "learned_linear":
            self.gamma = NoiseScheduleScalar(
                gamma_min=self.gamma_min, gamma_max=self.gamma_max
            )
        elif self.noise_schedule == "learned_net":
            self.gamma = NoiseScheduleNet(
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
                scale_non_linear_init=self.scale_non_linear_init,
            )
        else:
            raise NotImplementedError(f"Unknown noise schedule {self.noise_schedule}")

        # Score model specification
        if self.score == "transformer":
            self.score_model = TransformerScoreNet(
                d_t_embedding=self.d_t_embedding,
                score_dict=self.score_dict,
                adanorm=False,
            )
        elif self.score == "transformer_adanorm":
            self.score_model = TransformerScoreNet(
                d_t_embedding=self.d_t_embedding,
                score_dict=self.score_dict,
                adanorm=True,
            )
        elif self.score in ["graph", "chebconv", "edgeconv"]:
            self.score_model = GraphScoreNet(
                d_t_embedding=self.d_t_embedding,
                score_dict=self.score_dict,
                norm_dict=self.norm_dict,
                gnn_type=self.score,
            )
        else:
            raise NotImplementedError(f"Unknown score model {self.score}")

        # Optional encoder/decoder for latent diffusion
        if self.use_encdec:
            self.encoder = MLPEncoder(**self.encoder_dict)

            self.decoder = MLPDecoder(
                d_output=self.d_feature,
                noise_scale=self.noise_scale,
                **self.decoder_dict,
            )

        # Embedding for class and context
        if self.n_classes > 0:
            self.embedding_class = nn.Embed(self.n_classes, self.d_context_embedding)
        self.embedding_context = nn.Dense(self.d_context_embedding)

    def score_eval(self, z, t, conditioning, mask):
        """Evaluate the score model."""
        cond = self.embed(conditioning)
        return self.score_model(
            z=z,
            t=t,
            conditioning=cond,
            mask=mask,
        )

    def gammat(self, t):
        return self.gamma(t)

    def recon_loss(self, x, f, cond):
        """The reconstruction loss measures the gap in the first step.
        We measure the gap from encoding the image to z_0 and back again.
        """
        g_0 = self.gamma(0.0)
        eps_0 = jax.random.normal(self.make_rng("sample"), shape=f.shape)
        z_0 = variance_preserving_map(f, g_0, eps_0)
        z_0_rescaled = z_0 / alpha(g_0)
        loss_recon = -self.decode(z_0_rescaled, cond).log_prob(x)

        return loss_recon

    def recon_mass_loss(self, x, f, cond):
        """ Additional term to the recon_loss that enforces mass conservation. """
        g_0 = self.gamma(0.0)
        eps_0 = jax.random.normal(self.make_rng("sample"), shape=f.shape)
        z_0 = variance_preserving_map(f, g_0, eps_0)
        z_0_rescaled = z_0 / alpha(g_0)

        # calculate the mass of the input and the output
        # get the mass index
        m_idx_start = self.n_pos_features + self.n_vel_features
        m_idx_end = m_idx_start + self.n_mass_features
        logm_scale = np.array(self.norm_dict['x_std'][m_idx_start:m_idx_end])
        logm_x = x[..., m_idx_start:m_idx_end] * logm_scale
        logm_z0 = z_0_rescaled[..., m_idx_start:m_idx_end] * logm_scale

        # calculate the total mass
        logm_x_total = np.log10((10**logm_x).sum((-2)))
        logm_z0_total = np.log10((10**logm_z0).sum((-2)))

        loss_recon_mass = -tfd.Normal(
            loc=logm_z0_total, scale=self.noise_mass_scale).log_prob(logm_x_total)

        return loss_recon_mass

    def latent_loss(self, f):
        """The latent loss measures the gap in the last step, this is the KL
        divergence between the final sample from the forward process and starting
        distribution for the reverse process, here taken to be a N(0,1).
        """
        g_1 = self.gamma(1.0)
        var_1 = sigma2(g_1)
        mean1_sqr = (1.0 - var_1) * np.square(f)
        loss_klz = 0.5 * (mean1_sqr + var_1 - np.log(var_1) - 1.0)
        return loss_klz

    def diffusion_loss(self, t, f, cond, mask):
        """The diffusion loss measures the gap in the intermediate steps."""
        # Sample z_t
        g_t = self.gamma(t)
        eps = jax.random.normal(self.make_rng("sample"), shape=f.shape)
        z_t = variance_preserving_map(f, g_t[:, None], eps)
        # Compute predicted noise
        eps_hat = self.score_model(
            z_t,
            g_t,
            cond,
            mask,
        )
        deps = eps - eps_hat
        loss_diff_mse = np.square(deps)  # Compute MSE of predicted noise
        T = self.timesteps
        # NOTE: retain dimension here so that mask can be applied later (hence dummy dims)
        # NOTE: opposite sign convention to official VDM repo!
        if T == 0:
            # Loss for infinite depth T, i.e. continuous time
            _, g_t_grad = jax.jvp(self.gamma, (t,), (np.ones_like(t),))
            loss_diff = -0.5 * g_t_grad[:, None, None] * loss_diff_mse
        else:
            # Loss for finite depth T, i.e. discrete time
            s = t - (1.0 / T)
            g_s = self.gamma(s)
            loss_diff = 0.5 * T * np.expm1(g_s - g_t)[:, None, None] * loss_diff_mse

        return loss_diff

    def __call__(self, x, conditioning=None, mask=None):
        d_batch = x.shape[0]

        # 1. Reconstruction loss
        # Add noise and reconstruct
        f = self.encode(x, conditioning)
        loss_recon = self.recon_loss(x, f, conditioning)

        # 2. Latent loss
        # KL z1 with N(0,1) prior
        loss_klz = self.latent_loss(f)

        # 3. Diffusion loss
        # Sample time steps
        rng1 = self.make_rng("sample")
        if self.antithetic_time_sampling:
            t0 = jax.random.uniform(rng1)
            t = np.mod(t0 + np.arange(0.0, 1.0, step=1.0 / d_batch), 1.0)
        else:
            t = jax.random.uniform(rng1, shape=(d_batch,))
        # Discretize time steps if we're working with discrete time
        T = self.timesteps
        if T > 0:
            t = np.ceil(t * T) / T
        cond = self.embed(conditioning)
        loss_diff = self.diffusion_loss(t, f, cond, mask)

        # 4. Mass reconstruction loss
        f = self.encode(x, conditioning)
        loss_recon_mass = self.recon_mass_loss(x, f, conditioning)

        return (loss_diff, loss_klz, loss_recon, loss_recon_mass)

    def embed(self, conditioning):
        """Embed the conditioning vector, optionally including embedding a class assumed to be the first element of the context vector."""
        if not self.embed_context:
            return conditioning
        else:
            if (
                self.n_classes > 0 and conditioning.shape[-1] > 1
            ):  # If both classes and conditioning
                classes, conditioning = (
                    conditioning[..., 0].astype(np.int32),
                    conditioning[..., 1:],
                )
                class_embedding, context_embedding = self.embedding_class(
                    classes
                ), self.embedding_context(conditioning)
                return class_embedding + context_embedding
            elif (
                self.n_classes > 0 and conditioning.shape[-1] == 1
            ):  # If no conditioning but classes
                classes = conditioning[..., 0].astype(np.int32)
                class_embedding = self.embedding_class(classes)
                return class_embedding
            elif (
                self.n_classes == 0 and conditioning is not None
            ):  # If no classes but conditioning
                context_embedding = self.embedding_context(conditioning)
                return context_embedding
            else:  # If no conditioning
                return None

    def encode(self, x, conditioning=None, mask=None):
        """Encode an image x."""

        # Encode if using encoder-decoder; otherwise just return data sample
        if self.use_encdec:
            if conditioning is not None:
                cond = self.embed(conditioning)
            else:
                cond = None
            return self.encoder(x, cond, mask)
        else:
            return x

    def decode(
        self,
        z0,
        conditioning=None,
        mask=None,
    ):
        """Decode a latent sample z0."""

        # Decode if using encoder-decoder; otherwise just return last latent distribution
        if self.use_encdec:
            if conditioning is not None:
                cond = self.embed(conditioning)
            else:
                cond = None
            return self.decoder(z0, cond, mask)
        else:
            return tfd.Normal(loc=z0, scale=self.noise_scale)

    def sample_step(self, rng, i, T, z_t, conditioning=None, mask=None):
        """Sample a single step of the diffusion process."""
        rng_body = jax.random.fold_in(rng, i)
        eps = jax.random.normal(rng_body, z_t.shape)
        t = (T - i) / T
        s = (T - i - 1) / T

        g_s = self.gamma(s)
        g_t = self.gamma(t)
        cond = self.embed(conditioning)
        eps_hat_cond = self.score_model(
            z_t,
            g_t * np.ones((z_t.shape[0],), z_t.dtype),
            cond,
            mask,
        )

        a = nn.sigmoid(g_s)
        b = nn.sigmoid(g_t)
        c = -np.expm1(g_t - g_s)
        sigma_t = np.sqrt(sigma2(g_t))
        z_s = (
            np.sqrt(a / b) * (z_t - sigma_t * c * eps_hat_cond)
            + np.sqrt((1.0 - a) * c) * eps
        )

        return z_s

    def evaluate_score(
        self,
        z_t,
        g_t,
        cond,
        mask,
    ):
        return self.score_model(
            z=z_t,
            t=g_t,
            conditioning=cond,
            mask=mask,
        )
