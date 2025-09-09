from .src.network import MoxiFederatedNetwork
from .src.node import WORKER_MAP
from .src.util import random_network_dict, star_network_dict
from .src.errors import AssemblyException
from typing import Literal
from torch.utils.data import DataLoader
from flwr_datasets.partitioner import DirichletPartitioner
import torch.nn as nn
from .src.data import datamaker


# create network configs
def get_network_config(config):
    to_keep = ["network_name", "network_type", "federated_rounds"]
    return {k: config[k] for k in to_keep if k in config}


# create node configs
def get_node_config(config):
    return config["node_base_config"]


# Create Network
def assemble_network(config, model, data, adj_map: dict[str, list[str]] = None):
    network_config = get_network_config(config)
    node_config = get_node_config(config)
    node_config["model"] = model
    flnw = MoxiFederatedNetwork(network_config)
    worker = WORKER_MAP[config.get("comms").lower()][config.get("ml_framework").lower()]

    if adj_map:
        conn_config = {"nsm": adj_map}
    elif "de" in config.get("network_type").lower():
        conn_config = {"nsm": random_network_dict(config.get("network_size"))}
    elif "centralised" in config.get("network_type").lower():
        conn_config = {"nsm": star_network_dict(config.get("network_size"))}
    else:
        raise AssemblyException("Failed to parse network adj map")

    flnw.populate_network(conn_config, node_config, worker, data)
    return flnw


DIRCILET_TARGETS = {
    "binary_classification": "diagnosis",
    "regression": "species",
    "image_classification": "label",
}


def test_data_builder(config: dict):
    if isinstance(config["experiment_config"]["partitioner"], DirichletPartitioner):
        partitioner = DirichletPartitioner(
            alpha=config["experiment_config"]["alpha"],
            num_partitions=config["network_size"],
            partition_by=DIRCILET_TARGETS[config["experiment_config"]["task"]],
            seed=44,
        )
    else:
        partitioner = config["experiment_config"]["partitioner"](
            **config["experiment_config"]["partitioner_params"]
        )
        
    train_loaders, test_loader = datamaker(
        task=config["experiment_config"]["task"],
        partitioner=partitioner,
        batch_size=config["node_base_config"]["batch_size"],
        num_workers=config["experiment_config"]["num_worker"],
        max_samples=config["experiment_config"]["max_samples"],
    )
    data = [(train_loader, test_loader) for train_loader in train_loaders]
    return data


def create_experiment(config: dict, model: nn.Module):
    data = test_data_builder(config)
    flnw = assemble_network(config, model, data)
    return flnw
