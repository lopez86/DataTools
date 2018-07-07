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
        self._config = ''
        self._task = Task.TRAIN
        self._objective = Objective.REGRESSION
        self._boosting = Boosting.GBDT
        self._learning_rate = 0.1
        self._num_leaves = 31
        self._tree_learner = TreeLearner.SERIAL
        self._num_threads = 0
        self._device_type = DeviceType.CPU
        self._seed = 0
        self._max_depth = -1
        self._min_data_in_leaf = 20
        self._min_sum_hessian_in_leaf = 1e-3
        self._bagging_fraction = 1.0
        self._bagging_freq = 0
        self._bagging_seed = 3
        self._feature_fraction = 1.0
        self._feature_fraction_seed = 2
        self._early_stopping_round = 0
        self._max_delta_step = 0.0
        self._lambda_l1 = 0.0
        self._lambda_l2 = 0.0
        self._min_gain_to_split = 0.0
        # I/O Params
        self._verbosity = 1
        # Objective stuff
        self._num_class = 1
        self._is_unbalance = False
        self._scale_pos_weight = 1.0
        # Metric
        self._metric = []
        self._is_provide_training_metric = False

    def set_config(self, config: str):
        self._config = config
        return self

    def set_task(self, task: Task):
        self._task = task
        return self

    def set_objective(self, objective: Objective):
        self._objective = objective
        return self

    def set_boosting(self, value: Boosting):
        self._boosting = value
        return self

    def set_learning_rate(self, value: float):
        self._learning_rate = value
        return self

    def set_num_leaves(self, value: int):
        self._num_leaves = value
        return self

    def set_tree_learner(self, value: TreeLearner):
        self._tree_learner = value
        return self

    def set_num_threads(self, value: int):
        self._num_threads = value
        return self

    def set_device_type(self, value: DeviceType):
        self._device_type = value
        return self

    def set_seed(self, value: int):
        self._seed = value
        return self

    def set_max_depth(self, value: int):
        self._max_depth = value
        return self

    def set_min_data_in_leaf(self, value: int):
        self._min_data_in_leaf = value
        return self

    def set_min_sum_hessian_in_leaf(self, value: float):
        self._min_sum_hessian_in_leaf = value
        return self

    def set_bagging_fraction(self, value: float):
        self._bagging_fraction = value
        return self

    def set_bagging_freq(self, value: int):
        self._bagging_freq = value
        return self

    def set_bagging_seed(self, value: int):
        self._bagging_seed = value
        return self

    def set_feature_fraction(self, value: float):
        self._feature_fraction = value
        return self

    def set_feature_fraction_seed(self, value: int):
        self._feature_fraction_seed = value
        return self

    def set_early_stopping_round(self, value: int):
        self._early_stopping_round = value
        return self

    def set_max_delta_step(self, value: float):
        self._max_delta_step = value
        return self

    def set_lambda_l1(self, value: float):
        self._lambda_l1 = value
        return self

    def set_lambda_l2(self, value: float):
        self._lambda_l2 = value
        return self

    def set_min_gain_to_split(self, value: float):
        self._min_gain_to_split = value
        return self

    def set_verbosity(self, value: int):
        self._verbosity = value
        return self

    def set_num_class(self, value: int):
        self._num_class = value
        return self

    def set_is_unbalance(self, value: bool):
        self._is_unbalance = value
        return self

    def set_scale_pos_weight(self, value: float):
        self._scale_pos_weight = value
        return self

    def add_metric(self, value: Metric):
        self._metric.append(value)
        return self

    def add_metrics(self, values: List[Metric]):
        self._metric = self._metric + values
        return self

    def set_is_provide_training_meric(self, value: bool):
        self._is_provide_training_metric = value
        return self

    def construct(self) -> dict:
        param_dict = {}
        if self._config:
            param_dict['config'] = self._config
        param_dict['task'] = self._task.value
        param_dict['objective'] = self._objective.value
        param_dict['boosting'] = self._boosting.value
        param_dict['learning_rate'] = self._learning_rate
        param_dict['num_leaves'] = self._num_leaves
        param_dict['tree_learner'] = self._tree_learner.value
        param_dict['num_threads'] = self._num_threads
        param_dict['device_type'] = self._device_type.value
        param_dict['seed'] = self._seed
        param_dict['max_depth'] = self._max_depth
        param_dict['min_data_in_leaf'] = self._min_data_in_leaf
        param_dict['min_sum_hessian_in_leaf'] = self._min_sum_hessian_in_leaf
        param_dict['bagging_fraction'] = self._bagging_fraction
        param_dict['bagging_freq'] = self._bagging_freq
        param_dict['bagging_seed'] = self._bagging_seed
        param_dict['feature_fraction'] = self._feature_fraction
        param_dict['feature_fraction_seed'] = self._feature_fraction_seed
        param_dict['early_stopping_round'] = self._early_stopping_round
        param_dict['max_delta_step'] = self._max_delta_step
        param_dict['lambda_l1'] = self._lambda_l1
        param_dict['lambda_l2'] = self._lambda_l2
        param_dict['min_gain_to_split'] = self._min_gain_to_split
        param_dict['verbosity'] = self._verbosity
        param_dict['num_class'] = self._num_class
        param_dict['is_unbalance'] = self._is_unbalance
        if not self._is_unbalance:
            param_dict['scale_pos_weight'] = self._scale_pos_weight
        if self._metric:
            param_dict['metric'] = [
                metric.value for metric in funcy.ldistinct(self._metric)
            ]
        param_dict['is_provide_training_metric'] = self._is_provide_training_metric

        return param_dict
