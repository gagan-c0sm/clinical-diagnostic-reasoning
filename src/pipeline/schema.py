"""Canonical tree: final/current diagnosis root, supporting evidence below it."""


def validate_tree(nodes, inputs=None):
    if not nodes:
        return ["empty_tree"]
    errors = []
    roots = [key for key,n in nodes.items() if n.get("parent") is None]
    if len(roots) != 1:
        errors.append("expected_one_root")
    for key, node in nodes.items():
        if node.get("type") not in {"Input", "Cause", "Intermedia"}:
            errors.append(f"{key}:invalid_type")
        if not isinstance(node.get("content"), str) or not node["content"].strip():
            errors.append(f"{key}:empty_content")
        parent = node.get("parent")
        if parent is not None and (parent not in nodes or key not in nodes[parent].get("children", [])):
            errors.append(f"{key}:invalid_parent")
        for child in node.get("children", []):
            if child not in nodes or nodes[child].get("parent") != key:
                errors.append(f"{key}:invalid_child")
        if node.get("type") == "Input":
            field = node.get("source_field")
            if type(field) is not int or not 1 <= field <= 6:
                errors.append(f"{key}:invalid_field")
            elif inputs and node["content"] not in inputs.get_field(field):
                errors.append(f"{key}:ungrounded_quote")
            if node.get("children"):
                errors.append(f"{key}:observation_has_children")
    visited, active = set(), set()
    def visit(key):
        if key in active:
            errors.append("cycle")
            return
        if key in visited or key not in nodes:
            return
        visited.add(key)
        active.add(key)
        for child in nodes[key].get("children", []):
            visit(child)
        active.remove(key)
    for root in roots:
        visit(root)
    if visited != set(nodes):
        errors.append("unreachable_nodes")
    return errors


def to_nested(nodes):
    """Convert canonical nodes to the dataset's nested annotation encoding."""
    errors = validate_tree(nodes)
    if errors:
        raise ValueError(errors)
    def encode(key):
        n = nodes[key]
        suffix = f"Input{n['source_field']}" if n['type'] == 'Input' else f"{n['type']}_{list(nodes).index(key)+1}"
        children = {}
        for child in n["children"]:
            part = encode(child)
            if set(children) & set(part):
                raise ValueError("Nested format would collapse duplicate observations")
            children.update(part)
        return {f"{n['content']}${suffix}":children}
    return encode(next(k for k,n in nodes.items() if n["parent"] is None))
