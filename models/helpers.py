import importlib

def get_metric(metric):
    module = importlib.import_module('sklearn.metrics')
    f = getattr(module, metric)
    return f