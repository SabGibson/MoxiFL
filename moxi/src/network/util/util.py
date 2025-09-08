from moxi.src.common.interfaces import (
    MoxiNetworkProtocol,
    NetworkArchitecture,
    MoxiNodeConfig,
    MoxiWorkerProtocol,
)
import matplotlib.pyplot as plt
import networkx as nx
import torch


def add_nodes(
    network: MoxiNetworkProtocol,
    connect: NetworkArchitecture,
    config: MoxiNodeConfig,
    worker_constructor: MoxiWorkerProtocol,
):

    for device_id in list(connect["nsm"].keys()):
        # Create worker instance
        worker = worker_constructor(
            nw_ref=network,
            device_id=device_id,
            model=config.get("model"),
            lr=config.get("learning_rate"),
            n_epochs=config.get("n_epochs"),
            optimizer=config.get("optimizer"),
            criterion=config.get("criterion"),
            random_sampling=config.get("random_sampling"),
        )

        # Add node and worker in the same scope
        network._graph.add_node(device_id)
        network._graph.nodes[device_id]["node"] = worker


def add_edges(network: MoxiNetworkProtocol, connect: NetworkArchitecture):
    for node, neighbours in connect["nsm"].items():
        for neighbour in neighbours:
            network._graph.add_edge(node, neighbour)


def add_datasets(network: MoxiNetworkProtocol, datasets: list[dict[object]]):
    for device_id, data in zip(network._graph.nodes(), datasets):
        # Get worker with safety check
        worker = network._graph.nodes[device_id].get("node")
        if worker is None:
            raise ValueError(f"No worker found for node {device_id}")

        # Set training data
        worker.train_data = data[0]
        # Set validation data if available
        if len(data) > 1:
            worker.validation_data = data[1]


def plot_network(network, title):
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(network._graph)
    nx.draw(
        network._graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=10,
        font_weight="bold",
    )
    plt.title(title)
    plt.show()


async def calculate_validation_loss(worker: MoxiWorkerProtocol) -> float:
    """Calculate MSE loss on validation data"""
    total_loss = 0.0
    with torch.no_grad():
        for X, y in worker.validation_data:
            pred = worker.model(X)
            total_loss += worker.criterion(pred, y).item()
    return total_loss / len(worker.validation_data)
