import pytest
import networkx as nx
from moxi.src.core.interface import NodeType
from moxi.src.core.node.moxinode import MoxiClientNode
from moxi.src.trainers import DummyTrainer


class DummyNetwork:
    def __init__(self):
        self.graph = nx.Graph()


@pytest.fixture
def create_dummy_network():
    return DummyNetwork()


@pytest.fixture
def create_dummy_client_node():

    node_id = "client_1"
    node_type = NodeType.CLIENT
    parent_network = DummyNetwork()
    client_node = MoxiClientNode(node_id, node_type, parent_network)

    # setup network

    client_node.parent_network.graph.add_node(client_node.node_id)
    client_node.parent_network.graph.add_node("client_2")
    client_node.parent_network.graph.add_node("client_3")
    client_node.parent_network.graph.add_node("orchestrator_1")

    client_node.parent_network.graph.add_edges_from(
        [
            (client_node.node_id, "client_2"),
            (client_node.node_id, "client_3"),
            (client_node.node_id, "orchestrator_1"),
        ]
    )

    return client_node
