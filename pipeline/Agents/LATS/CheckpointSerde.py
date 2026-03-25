from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from Agents.LATS.Reflection import Node

_NODE_MARKER = "__lats_node__"


def _node_to_payload(node: Node) -> dict:
    return {
        _NODE_MARKER: True,
        "messages": node.messages,
        "reflection": node.reflection,
        "value": node.value,
        "visits": node.visits,
        "depth": node.depth,
        "is_solved": node._is_solved,
        "children": [_node_to_payload(child) for child in node.children],
    }

def _node_to_metadata(node: Node) -> dict:
    return {
        "type": "LATSNode",
        "value": node.value,
        "visits": node.visits,
        "depth": node.depth,
        "is_solved": node._is_solved,
    }


def _payload_to_node(payload: dict, parent: Node | None = None) -> Node:
    node = Node.__new__(Node)
    node.messages = payload.get("messages", [])
    node.reflection = payload.get("reflection")
    node.parent = parent
    node.children = []
    node.value = payload.get("value", 0)
    node.visits = payload.get("visits", 0)
    node.depth = payload.get("depth", parent.depth + 1 if parent else 1)
    node._is_solved = payload.get("is_solved", False)
    for child_payload in payload.get("children", []):
        child = _payload_to_node(child_payload, parent=node)
        node.children.append(child)
    return node


def _serialize_obj(obj):
    if isinstance(obj, Node):
        return _node_to_payload(obj)
    if isinstance(obj, dict):
        return {key: _serialize_obj(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_serialize_obj(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_serialize_obj(value) for value in obj)
    if isinstance(obj, set):
        return {_serialize_obj(value) for value in obj}
    return obj


def _deserialize_obj(obj):
    if isinstance(obj, dict):
        if obj.get(_NODE_MARKER):
            return _payload_to_node(obj)
        return {key: _deserialize_obj(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_deserialize_obj(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_deserialize_obj(value) for value in obj)
    if isinstance(obj, set):
        return {_deserialize_obj(value) for value in obj}
    return obj


class LatsJsonPlusSerializer(JsonPlusSerializer):
    def _default(self, obj):
        if isinstance(obj, Node):
            return _node_to_metadata(obj)
        return super()._default(obj)

    def dumps_typed(self, obj):
        return super().dumps_typed(_serialize_obj(obj))

    def loads_typed(self, data):
        obj = super().loads_typed(data)
        return _deserialize_obj(obj)
