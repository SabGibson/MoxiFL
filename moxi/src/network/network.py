from moxi.src.common.interfaces import (
    MoxiMLModelType,
    MoxiNetworkConfiguration,
    MoxiNetworkType,
)
from .util import add_nodes, add_edges, add_datasets, calculate_validation_loss
import networkx as nx
import asyncio
from collections import defaultdict
import mlflow

NAME_MIN_LIMIT = 5
MIN_TRAINING_ROUNDS = 5
import random


class MoxiFederatedNetwork:
    def __init__(self, config: MoxiNetworkConfiguration):
        self.config = config
        self._graph = nx.Graph()

    @property
    def network_id(self):
        return self.config.get("network_name", "Blank")

    @network_id.setter
    def network_id(self, new_value):
        if isinstance(new_value, str) & len(new_value > NAME_MIN_LIMIT):
            self.config["network_name"] = new_value

    @property
    def network_type(self):
        return self.config.get("model_type")

    @network_type.setter
    def model_type(self, new_value):
        try:
            new_model_type = MoxiMLModelType[new_value]
            self.config["model_type"] = new_model_type

        except Exception as e:
            print(
                "Attempting to set invalid 'MoxiMLModelType', please select : 'centralized' or 'decentralized'"
            )

    @property
    def network_type(self):
        return self.config.get("network_type")

    @network_type.setter
    def network_type(self, new_value):
        try:
            new_network_type = MoxiNetworkType[new_value]
            self.config["network_type"] = new_network_type

        except Exception as e:
            print(
                "Attempting to set invalid 'MoxiNetworkType', please select : 'parametric' or 'nonparametric'"
            )

    @property
    def max_rounds(self):
        return self.config.get("federated_rounds")

    @max_rounds.setter
    def max_rounds(self, new_value: int):
        if (new_value > MIN_TRAINING_ROUNDS) & isinstance(new_value, int):
            self.config["federated_rounds"] = new_value

    #
    def __repr__(self):
        return (
            f"<MoxiFederatedNetwork(network_id='{self.network_id}', "
            f"network_type='{self.config.get('network_type')}', "
            f"model_type='{self.config.get('model_type')}', "
            f"num_nodes={self._graph.number_of_nodes()}, "
            f"num_edges={self._graph.number_of_edges()}, "
            f"max_rounds={self.config.get('federated_rounds')})>"
        )

    def populate_network(
        self, network_conn_config, worker_config, worker_constructor, datasets
    ):
        # add nodes
        add_nodes(self, network_conn_config, worker_config, worker_constructor)
        # add edges
        add_edges(self, network_conn_config)
        # add data
        add_datasets(self, datasets)

    async def train(
        self, num_rounds=5, epochs_per_round=1, experiment_name="Federated_Learning"
    ):
        """
        Federated training with MLflow logging.
        """
        mlflow.set_experiment(experiment_name)

        # Network-level run
        with mlflow.start_run(run_name="network_aggregate") as parent_run:
            parent_run_id = parent_run.info.run_id

            for round_id in range(num_rounds):
                device_train_losses = {}
                device_val_losses = {}

                tasks = [
                    asyncio.create_task(
                        self._run_device(
                            node=self._graph.nodes[node_id]["node"],
                            round_id=round_id,
                            epochs_per_round=epochs_per_round,
                            device_train_losses=device_train_losses,
                            device_val_losses=device_val_losses,
                        )
                    )
                    for node_id in self._graph.nodes()
                ]

                await asyncio.gather(*tasks)

                # Aggregate metrics across devices
                avg_train_loss = sum(device_train_losses.values()) / len(
                    device_train_losses
                )
                avg_val_loss = sum(device_val_losses.values()) / len(device_val_losses)

                mlflow.log_metric("avg_train_loss_round", avg_train_loss, step=round_id)
                mlflow.log_metric("avg_val_loss_round", avg_val_loss, step=round_id)

        print("Training Complete!")

    async def _run_device(
        self, node, round_id, epochs_per_round, device_train_losses, device_val_losses
    ):

        with mlflow.start_run(run_name=f"device_{node.device_id}", nested=True):
            mlflow.log_param("device_id", node.device_id)

            local_update, epoch_losses = await node.train_local(epochs_per_round)
            for step, loss in enumerate(epoch_losses):
                mlflow.log_metric(
                    "train_loss_epoch", loss, step=round_id * epochs_per_round + step
                )

            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            mlflow.log_metric("train_loss_round", avg_train_loss, step=round_id)
            device_train_losses[node.device_id] = avg_train_loss

            val_loss, val_acc = await node.validate_local()
            if val_loss is not None:
                mlflow.log_metric("val_loss_round", val_loss, step=round_id)
            device_val_losses[node.device_id] = val_loss

            penalized_update = await node.compute_updates(local_update)
            node.model.load_state_dict(penalized_update)

            peers = [d for d in node.peers if d.device_id != node.device_id]
            if peers:
                k = max(1, len(peers) // 2)
                k = min(k, len(peers))
                for peer in random.sample(peers, k=k):
                    await node.send_updates(peer.device_id, penalized_update)
