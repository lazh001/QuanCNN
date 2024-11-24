import sys
from pathlib import Path
from typing import Optional, Union

from omegaconf import OmegaConf, DictConfig


def get_config(base_conf_filepath: Optional[Union[Path, str]] = None) -> DictConfig:
    """Get user configurations from CLI parameters and overwrite the default ones.

    Args:
        base_conf_filepath: The path to the template configuration file, which provides default
            values of all the configuration items.

    Returns:
        A DictConfig object, which should include all the settings needed by NeuralZip.
    """
    conf = OmegaConf.load(base_conf_filepath) if base_conf_filepath else OmegaConf.create()
    conf_files = [i[len('conf_filepath='):] for i in sys.argv if i.startswith('conf_filepath=')]        # 这个就是在命令行参数中找一下有没有conf_filepath=开头的内容，
                                                                                                        # 然后用字符串切片把后面的字符内容提取出来
                                                                                                        # 比较奇怪的是允许命令行参数中有好几个以conf_filepath=开头的内容，也就是允许多条配置文件
                                                                                                        # 多个配置文件在merge中应该是后面的配置文件可以覆盖前面的内容，但是保留前面独占的内容
                                                                                                        # 简单理解就是命令行输入可以覆盖默认配置文件中的内容
    for conf_file in conf_files:
        cfg_load = OmegaConf.load(conf_file)
        conf.merge_with(cfg_load)

    conf_cli_override = [i for i in sys.argv[1:] if not i.startswith('conf_filepath=')]
    conf.merge_with_dotlist(conf_cli_override)

    return conf
