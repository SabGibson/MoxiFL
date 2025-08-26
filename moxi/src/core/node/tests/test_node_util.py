from moxi.src.core.interface import NodeType
from moxi.src.core.node.moxinode import MoxiClientNode
from moxi.src.trainers import DummyTrainer, PytorchTrainer, ScikitLearnTrainer


def test_create_moxi_client_node(create_dummy_network):

    # Given
    node_id = "client_1"
    node_type = NodeType.CLIENT
    parent_network = create_dummy_network
    # When
    client_node = MoxiClientNode(node_id, node_type, parent_network)

    # Then
    assert isinstance(client_node, MoxiClientNode)
    assert client_node.node_id == node_id
    assert client_node.nodetype == node_type.value
    assert client_node.parent_network == parent_network
    assert client_node.current_round == 0
    assert client_node.model_version == 0


def test_moxi_client_neighbour_size(create_dummy_client_node):
    # Given
    client_node = create_dummy_client_node

    # When
    neighbours = list(client_node.neighbours)
    # Then
    assert len(neighbours) == 3


def test_moxi_client_nodetype(create_dummy_client_node):
    # Given
    client_node = create_dummy_client_node
    # When
    node_type = client_node.nodetype
    # Then
    assert node_type == NodeType.CLIENT.value


from dataclasses import dataclass


def test_moxi_client_init(create_dummy_client_node):
    # Given
    client_node = create_dummy_client_node
    # When
    test_config = {
        "framework": "dummy",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "local_rounds": 1,
        "min_participants": 1,
        "max_participants": 3,
        "timeout": 300.0,
    }
    trainer = PytorchTrainer
    client_node.initialize(model=None, data=None, config=test_config)
    # Then
    #
    assert client_node.trainer is not None


def test_moxi_client_train_local_model(create_dummy_client_node):
    # Given
    client_node = create_dummy_client_node

    # When
    test_config = {
        "framework": "dummy",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "local_rounds": 1,
        "min_participants": 1,
        "max_participants": 3,
        "timeout": 300.0,
    }
    
    client_node.initialize(model=None, data=None, config=test_config)
    client_node.train_local_model()

    # Then
    assert client_node.current_round == 1
    assert client_node.model_version == 1

# def test_moxi_client_update_model():
    