import os
from functools import wraps


def prevent_on_server(server_name: str="alblas"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.uname().nodename.split(".")[0] == server_name:
                raise EnvironmentError(
                    f"Do not run on the node {server_name}. "
                    f"This node has not enough memory."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator