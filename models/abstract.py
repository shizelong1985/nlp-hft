import os
from abc import ABC, abstractmethod

from joblib import dump, load
from models.helpers import get_metric


class AbstractModel(ABC):

    def __init__(self):
        '''
        This is an abstract base class (ABC). That means that it cannot be instantiated unless we define a subclass that
        implements all the abstract methods of the base class.

        The @abstractmethod decorator is used to define the abstract methods of the base class that need to be
        implemented by the subclass.

        The purpose of this particular abstract base class is to serve as a template for machine learning models.

        In addition to the abstract methods "fit" and "predict", this class also implements methods "save" and "load" to
        save/load models in JSON format.
        '''

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Fit model."""

    @abstractmethod
    def predict(self, X):
        """Predict using model"""

    def evaluate(self, X, y, metrics=None):
        if metrics is None:
            raise ValueError('Metrics to evaluate is None.')
        y_pred = self.predict(X)
        res = dict()
        for metric_key in metrics:
            metric = metrics[metric_key]
            f = get_metric(metric)
            res[metric_key] = f(y, y_pred)
        return res

    def save(self, name, dir):
        if os.path.isdir(dir) is False:
            os.makedirs(dir)
        dump(self, '{}/{}'.format(dir, name))
        return

    def load(self, name, dir):
        model = load('{}/{}'.format(dir, name))
        return model
