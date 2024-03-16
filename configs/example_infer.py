
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 42

    # inference hyperparameters
    config.logdir = "/mnt/ceph/users/tnguyen/dark_camels/point-cloud-diffusion-logging/"
    config.vdm_name = "vdm/expert-elevator-112"
    config.flows_name = "flows/tough-plasma-113"
    config.steps = 500
    config.batch_size = 256
    config.n_repeats = 100

    config.workdir = "/mnt/ceph/users/tnguyen/dark_camels/point-cloud-diffusion-outputs"
    config.output_name = None

    # data inference
    config.data = data = ml_collections.ConfigDict()
    data.dataset_root = "/mnt/ceph/users/tnguyen/dark_camels/" \
        "point-cloud-diffusion-datasets/processed_datasets/"
    data.dataset_name = "mw_zooms-wdm-dmprop/nmax100-vmaxtilde-pad-v2"
    data.n_particles = 100  # Select the first n_particles particles
    data.n_features = 8  # Select the first n_features features
    data.n_pos_features = 3
    data.box_size = 1.0  # Need to know the box size for augmentations
    data.conditioning_parameters = [
        "center_subhalo_mvir", "center_subhalo_vmax_tilde",
        "inv_wdm_mass", "log_sn1", "log_sn2", "log_agn1"
    ]

    return config
