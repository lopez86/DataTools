from enum import Enum
from typing import List

import funcy


class Task(Enum):
    TRAIN = 'train'
    PREDICT = 'predict'
    CONVERT_MODEL = 'convert_model'
    REFIT = 'refit'


class Objective(Enum):
    REGRESSION = 'regression'
    REGRESSION_L1 = 'regression_l1'
    HUBER = 'huber'
    FAIR = 'fair'
    POISSON = 'poisson'
    QUANTILE = 'quantile'
    MAPE = 'mape'
    GAMMA = 'gammma'
    TWEEDIE = 'tweedie'
    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    MULTICLASSOVA = 'multiclassova'
    XENTROPY = 'xentropy'
    XENTLAMBDA = 'xentlambda'
    LAMBDARANK = 'lambdarank'


class Boosting(Enum):
    GBDT = 'gbdt'
    RF = 'rf'
    DART = 'dart'
    GOSS = 'goss'


class TreeLearner(Enum):
    SERIAL = 'serial'
    FEATURE = 'feature'
    DATA = 'data'
    VOTING = 'voting'


class DeviceType(Enum):
    CPU = 'cpu'
    GPU = 'gpu'


class Metric(Enum):
    DEFAULT = ''
    NONE = 'None'
    MAE = 'mae'
    MSE = 'mse'
    RMSE = 'rmse'
    QUANTILE = 'quantile'
    MAPE = 'mape'
    HUBER = 'huber'
    POISSON = 'poisson'
    GAMMA = 'gamma'
    GAMMA_DEVIANCE = 'gamma_deviance'
    TWEEDIE = 'tweedie'
    NDCG = 'ndcg'
    MAP = 'map'
    AUC = 'auc'
    BINARY_LOGLOSS = 'binary_logloss'
    BINARY_ERROR = 'binary_error'
    MULTI_LOGLOSS = 'multi_logloss'
    MULTI_ERROR = 'multi_error'
    KLDIV = 'kldiv'


class LGBMParameterBuilder:
    def __init__(self):
        self._params = {}

    def set_config(self, config: str):
        self._params['config'] = config
        return self

    def set_task(self, task: Task):
        self._params['task'] = task.value
        return self

    def set_objective(self, objective: Objective):
        self._params['objective'] = objective.value
        return self

    def set_boosting(self, value: Boosting):
        self._params['boosting'] = value.value
        return self

    def set_learning_rate(self, value: float):
        self._params['learning_rate'] = value
        return self

    def set_num_leaves(self, value: int):
        self._params['num_leaves'] = value
        return self

    def set_tree_learner(self, value: TreeLearner):
        self._params['tree_learner'] = value
        return self

    def set_num_threads(self, value: int):
        self._params['num_threads'] = value
        return self

    def set_device_type(self, value: DeviceType):
        self._params['device_type'] = value.value
        return self

    def set_seed(self, value: int):
        self._params['seed'] = value
        return self

    def set_max_depth(self, value: int):
        self._params['max_depth'] = value
        return self

    def set_min_data_in_leaf(self, value: int):
        self._params['min_data_in_leaf'] = value
        return self

    def set_min_sum_hessian_in_leaf(self, value: float):
        self._params['min_sum_hessian_in_leaf'] = value
        return self

    def set_bagging_fraction(self, value: float):
        self._params['bagging_fraction'] = value
        return self

    def set_bagging_freq(self, value: int):
        self._params['bagging_freq'] = value
        return self

    def set_bagging_seed(self, value: int):
        self._params['bagging_seed'] = value
        return self

    def set_feature_fraction(self, value: float):
        self._params['feature_fraction'] = value
        return self

    def set_feature_fraction_seed(self, value: int):
        self._params['feature_fraction_seed'] = value
        return self

    def set_early_stopping_round(self, value: int):
        self._params['early_stopping_round'] = value
        return self

    def set_max_delta_step(self, value: float):
        self._params['max_delta_step'] = value
        return self

    def set_lambda_l1(self, value: float):
        self._params['lambda_l1'] = value
        return self

    def set_lambda_l2(self, value: float):
        self._params['lambda_l2'] = value
        return self

    def set_min_gain_to_split(self, value: float):
        self._params['min_gain_to_split'] = value
        return self

    def set_verbosity(self, value: int):
        self._params['verbosity'] = value
        return self

    def set_num_class(self, value: int):
        self._params['num_class'] = value
        return self

    def set_is_unbalance(self, value: bool):
        if value is True and 'scale_pos_weight' in self._params:
            raise ParameterError(
                'is_unbalance and scale_pos_weight cannot both be used.'
            )
        self._params['is_unbalance'] = value
        return self

    def set_scale_pos_weight(self, value: float):
        if self._params.get('is_unbalance', False) is True:
            raise ParameterError(
                'is_unbalance and scale_pos_weight cannot both be used.'
            )
        self._params['scale_pos_weight'] = value
        return self

    def add_metric(self, value: Metric):
        self._params['metric'] = funcy.ldistinct(
            self._params.get('metric', []).append(value.value)
        )
        return self

    def add_metrics(self, values: List[Metric]):
        self._params['metric'] = funcy.ldistinct(
            self._params.get('metric', []) + [v.value for v in values]
        )
        return self

    def set_is_provide_training_metric(self, value: bool):
        self._params['is_provide_training_metric'] = value
        return self

    def construct(self) -> dict:
        return self._params


class ParameterError(Exception):
    pass
