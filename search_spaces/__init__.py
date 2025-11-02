import importlib

class Node:
    
    def __init__(self):
        pass

    @property
    def is_terminal(self):
        pass

    @property
    def state(self):
        pass

    def get_actions_tuples(self):
        pass

    def play_action(self, action):
        pass

def make_search_space(cfg):
    """
    Instantiate a search space by name.

    Args:
        name: The name of the search space, e.g., "SS1".
        **kwargs: Arguments to pass to the search space constructor.

    Returns:
        An instance of the requested search space.
    """
    # Normalize to module and class names
    name = cfg.problem.name

    module_path = f".{name}.{name}"
    class_name = name.upper()
    
    try:
        module = importlib.import_module(module_path, package=__name__)
        cls = getattr(module, class_name)
        return cls(cfg)
    except ModuleNotFoundError:
        raise ValueError(f"Search space module '{module_path}' not found.")
    except AttributeError:
        raise ValueError(f"Class '{class_name}' not found in module '{module_path}'.")
