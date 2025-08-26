import networkx as nx
from typing import Literal
from moxi.src.core.node import create_node
from moxi.src.core.interface import NodeType

class MoxiFederatedNetwork:
    def __init__(
        self,
        network_alias: str,
        nw_architecture: Literal["centralized", "decentralized"]= "decentralized",
        config: dict = None,
    ) -> None:
        """
        A class to represent federated network and form the basis for calculations
        """
        self.graph = nx.Graph()
        self._arch = nw_architecture 
        self.network_name = network_alias  # dir name base
        self.config = config  # Store the config parameter
        self.has_orch = False

    @property
    def architecture(self):
        return self._arch
    
    def add_client_node(self, node_id: str):
        """Add client node that uses network"""
        if self.config is not None:
            node = create_node(self, node_id, NodeType.CLIENT, self, self.config)
            self.graph.add_node(node_id.node_id, node=node)

    def add_orch_node(self, node_id: str):
        """Add admin node that acts as orchestrator in centralised architecture"""
        if (self.config is not None) and (not self.has_orch):
            node = create_node(self, node_id, NodeType.ORCHESTRATOR, self, self.config)
            self.graph.add_node(node_id, node=node)
            self.has_orch = True

    def remove_node(self, node_id: str):
        """Remove node from network"""
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
            # If removing orchestrator node, update the flag
            if node_id in self.graph.nodes and self.graph.nodes[node_id].get('node', {}).get('node_type') == 'orchestrator':
                self.has_orch = False

    
    def setup(self):
        """Build the network of nodes based on the configured architecture."""
        n_size = self.config["network_size"]

        # Decide node creation function for device_0
        first_node = (
            lambda: self.add_orch_node("device_0")
            if self.architecture == "centralized"
            else lambda: self.add_client_node("device_0")
        )
        first_node()

        # Add the rest as clients
        for i in range(1, n_size):
            self.add_client_node(f"device_{i}")