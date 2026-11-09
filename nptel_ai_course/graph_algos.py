def __check_path_exists_util(cur_node, target_node, seen, path):
    """Internal method to form a path from inital node to target node if it exists"""
    seen.add(cur_node)
    if cur_node == target_node:
        return True
    for conn in cur_node.connections:
        if conn not in seen:
            result = __check_path_exists_util(conn, target_node, seen, path)
            if result:
                path.append(conn)
                return True
    return False


def get_path_in_graph(initial_node, target_node):
    """Returns a path from the initial node to the target node"""
    path = []
    seen = set()
    result = __check_path_exists_util(initial_node, target_node, seen, path)
    if result:
        # This is because we never add the inital node in the path, we start
        # adding from connections
        path.append(initial_node)
    path.reverse()
    return result, path
