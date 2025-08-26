from moxi.src.trainers.interfaces import MoxiTrainerFrameworkType
from moxi.src.core.interface import NodeType

def convert_str_to_framework(framework_str: str) -> MoxiTrainerFrameworkType:
    """Convert a string to a MoxiTrainerFrameworkType enum.

    Args:
        framework_str (str): The framework string.

    Returns:
        MoxiTrainerFrameworkType: The corresponding enum value.

    Raises:
        ValueError: If the string does not match any enum value.
    """
    try:
        return MoxiTrainerFrameworkType(framework_str.lower())
    except ValueError as e:
        raise ValueError(f"Invalid framework type: {framework_str}") from e


def convert_str_to_node_type(node_type_str: str) -> MoxiTrainerFrameworkType:
    """ convert string to NodeType enum"""
    try:
        return NodeType(node_type_str.lower())
    except ValueError as e:
        raise ValueError(f"Invalid node type: {node_type_str}") from e
