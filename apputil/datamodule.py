import pytorch_lightning as pl
from omegaconf import DictConfig
from pl_bolts.datamodules import MNISTDataModule


def get_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Create DataModule according to the user configuration.

    Currently, only ImageNet and CIFAR10 dataset is supported.

    Args:
        cfg: The top-level user configuration object.

    Returns:
        A LightningDataModule, which will be consumed by the neural network model.

    Raises:
        ValueError: An unsupported dataset is specified.
    """
    dataset_cfg = cfg.dataset
    if dataset_cfg.name == 'MNIST':
        dm = MNISTDataModule(dataset_cfg.data_dir,
                                num_workers=dataset_cfg.workers,
                                batch_size=dataset_cfg.batch_size,
                                seed=cfg.seed)
    else:
        raise ValueError(f'get_datamodule does not support dataset {dataset_cfg.name}')
    return dm
