import json
def load_config(path):
    from types import SimpleNamespace
    with open(path) as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    return config