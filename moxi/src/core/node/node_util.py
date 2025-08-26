# create a node 
from moxi.src.core.node.moxinode import MoxiClientNode
from moxi.src.trainers.util import convert_str_to_framework , convert_str_to_node_type

def create_node(parent_network, node_id, node_type, config, data=None, model=None) -> MoxiClientNode: 

    node_type = convert_str_to_node_type(node_type)
    node = MoxiClientNode(node_id, node_type, parent_network)
    # process config
    config["framework"] = convert_str_to_framework(config["framework"])
    node.initialize(model=model, data=data, config=config)
    return node