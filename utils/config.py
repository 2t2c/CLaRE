import argparse
from typing import Any

import yaml
from yacs.config import CfgNode


def construct_config(args: argparse.Namespace) -> CfgNode:
    """Loads YAML config file and overrides with CLI args.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Merged CfgNode config object.

    Raises:
        FileNotFoundError: If the config file path is invalid.
        yaml.YAMLError: If the YAML file is invalid or cannot be parsed.
    """
    try:
        with open(args.config, "r") as f:
            config_file = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {args.config}") from e
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing YAML config file: {args.config}") from e

    cfg = dict_to_yacs(config_file)
    args_dict = vars(args)

    for key, value in args_dict.items():
        cfg[key] = value

    return cfg


def dict_to_yacs(config: dict[str, Any]) -> CfgNode:
    """Recursively converts a dictionary into a yacs CfgNode.

    Args:
        config: A dictionary (possibly nested) representing the configuration.

    Returns:
        A CfgNode object with the same structure as the input dictionary,
        or the original value if it is not a dictionary.
    """
    if isinstance(config, dict):
        node = CfgNode()
        for k, v in config.items():
            node[k] = dict_to_yacs(v)
        return node
    return config
