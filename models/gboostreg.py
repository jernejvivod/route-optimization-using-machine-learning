import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


class GradientBoostingRegressor(BaseEstimator, ClassifierMixin):
    """
    Gradient boosting regressor implementation

    Args:
        params (dict): Model parameters. If None, use default pre-set values.
        n_rounds (int): Number of training rounds.
        objective (str): Prediction objecitve.
        f_to_name (dict): Dictionary mapping feature enumerations in the form 'f0', 'f1', ...
        to their names.

    Attributes:
        params (dict): Model parameters. If None, use default pre-set values.
        objective (str): Prediction objecitve.
        n_rounds (int): Number of training rounds.
    """

    def __init__(self, params=None, n_rounds=2000, objective='reg:linear', f_to_name=None):

        # Set parameters.
        if params is None:
            self.params = {
                'booster':            'gbtree',
                'objective':          objective,
                'learning_rate':      0.05,
                'max_depth':          14,
                'subsample':          0.9,
                'colsample_bytree':   0.7,
                'colsample_bylevel':  0.7,
                'silent':             1,
                'feval':              'rmsle'
            }
        else:
            self.params = params
        
        # Set prediction objective.
        self.objective = objective

        # Set number of rounds.
        self.n_rounds = n_rounds


    def fit(self, X, y):
        """
        Fit classifier to training data.

        Args:
            X (numpy.ndarray): Training data samples
            y (numpy.ndarray): Training data labels

        Returns:
            (obj): Reference to self
        """
        
        # Split training data into training and validation sets.
        data_train, data_val, target_train, target_val = train_test_split(X, y, test_size=0.2)

        # Define train and validation sets in required format.
        dtrain = xgb.DMatrix(data_train, np.log(target_train + 1))
        dval = xgb.DMatrix(data_val, np.log(target_val+1))
        watchlist = [(dval, 'eval'), (dtrain, 'train')]

        # Train model.
        gbm = xgb.train(self.params,
                        dtrain,
                        num_boost_round = self.n_rounds,
                        evals = watchlist,
                        verbose_eval = True
                        )
        self._gbm = gbm

        # Return reference to self.
        return self

  
    def predict(self, X, y=None):
        """
        Predict labels of new data.
        
        Args:
            X (numpy.ndarray): Data for which to predict target values

        Returns:
            (numpy.ndarray): Predicted target values
        """

        # Return labels with highest probability.
        return np.exp(self._gbm.predict(xgb.DMatrix(X))) - 1


    def score_features(self, f_to_name):
        """
        Score feature importances.

        Args:
            f_to_name (dict): Dictionary mapping feature enumerations such as 'f0', 'f1', ...
            to feature names.

        Returns:
            (dict): Dictionary mapping feature names as defined in f_to_name parameter to
            their estimated importances.
        """

        f_scores = self._gbm.get_fscore()
        sum_f_scores = sum(f_scores.values())
        return {f_to_name[key] : val/sum_f_scores for key, val in f_scores.items()}

