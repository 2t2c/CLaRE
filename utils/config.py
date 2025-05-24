from yacs.config import CfgNode


def load_config(config):
    """
    Convert a dictionary to a CfgNode object.
    :param:
        config: yaml file
    :return:
        cfg: CfgNode object
    """
    if isinstance(config, dict):
        node = CfgNode()
        for k, v in config.items():
            node[k] = load_config(v)
        return node

    return config
