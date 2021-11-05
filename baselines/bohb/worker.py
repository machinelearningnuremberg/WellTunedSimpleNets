from copy import deepcopy
from functools import partial
import os
from typing import Dict, Tuple, Union

from catboost import CatBoostClassifier, metrics
import ConfigSpace as cs
from hpbandster.core.worker import Worker
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import balanced_accuracy_score
import torch
import xgboost as xgb

from data.loader import Loader


def balanced_error(
    threshold_predictions: bool,
    predt: np.ndarray,
    dtrain: xgb.DMatrix,
) -> Tuple[str, float]:
    """Calculate the balanced error for the predictions.

    Calculate the balanced error. Used as an evaluation metric for
    the xgboost algorithm.

    Parameters:
    -----------
    threshold_predictions: bool
        If the predictions should be threshold to 0 or 1. Should only be used for
        binary classification.
    predt: np.ndarray
        The predictions of the algorithm.
    dtrain: float
        The real values for the set.

    Returns:
    --------
    str, float - The name of the evaluation metric and its value on the arguments.
    """

    if threshold_predictions:
        predt = np.array(predt)
        predt = predt > 0.5
        predt = predt.astype(int)
    else:
        predt = np.argmax(predt, axis=1)

    y_train = dtrain.get_label()
    accuracy_score = balanced_accuracy_score(y_train, predt)

    return 'Balanced_error', 1 - accuracy_score


class XGBoostWorker(Worker):

    def __init__(self, *args, param=None, **kwargs):

        super().__init__(*args, **kwargs)
        self.param = deepcopy(param)
        self.task_id = self.param['task_id']
        self.output_directory = self.param['output_directory']

        del self.param['task_id']
        del self.param['output_directory']

        if self.param['objective'] == 'binary:logistic':
            self.threshold_predictions = True
        else:
            self.threshold_predictions = False

    def compute(self, config, budget, **kwargs):
        """What should be computed for one XGBoost worker.

        The function takes a configuration and a budget, it
        then uses the xgboost algorithm to generate a loss
        and other information.

        Parameters:
        -----------
            config: dict
                dictionary containing the sampled configurations by the optimizer
            budget: float
                amount of time/epochs/etc. the model can use to train
        Returns:
        --------
            dict:
                With the following mandatory arguments:
                'loss' (scalar)
                'info' (dict)
        """
        xgboost_config = deepcopy(self.param)
        xgboost_config.update(config)

        if 'num_round' in xgboost_config:
            num_rounds = xgboost_config['num_round']
            del xgboost_config['num_round']
            early_stopping_iterations = None
        else:
            num_rounds = 4000
            early_stopping_iterations = \
                xgboost_config['early_stopping_rounds']
            del xgboost_config['early_stopping_rounds']

        if 'use_imputation' in xgboost_config:
            apply_imputation = xgboost_config['use_imputation']
            del xgboost_config['use_imputation']
        else:
            # if no conditional imputation, always use it
            apply_imputation = True

        if xgboost_config['use_ohe'] == 'True':
            use_ohe = True
        else:
            use_ohe = False

        del xgboost_config['use_ohe']

        loader = Loader(
            task_id=self.task_id,
            seed=xgboost_config['seed'],
            apply_one_hot_encoding=use_ohe,
            apply_imputation=apply_imputation,
        )
        splits = loader.get_splits()

        # not used at the moment
        # categorical_information = loader.categorical_information
        X_train = splits['X_train']
        X_val = splits['X_val']
        X_test = splits['X_test']
        y_train = splits['y_train']
        y_val = splits['y_val']
        y_test = splits['y_test']

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_val = xgb.DMatrix(X_val, label=y_val)
        d_test = xgb.DMatrix(X_test, label=y_test)

        eval_results = {}
        gb_model = xgb.train(
            xgboost_config,
            d_train,
            num_rounds,
            feval=partial(balanced_error, self.threshold_predictions),
            evals=[(d_train, 'd_train'), (d_val, 'd_val')],
            evals_result=eval_results,
            early_stopping_rounds=early_stopping_iterations,
        )

        # TODO Do something with eval_results in the future
        # print(eval_results)

        # Default value if early stopping is not activated
        best_iteration = None

        n_tree_limit = None
        # early stopping activated and triggered
        if hasattr(gb_model, 'best_score'):
            n_tree_limit = gb_model.best_ntree_limit
            best_iteration = gb_model.best_iteration
            print(f'Best iteration for xgboost: {best_iteration}')

        y_train_preds = gb_model.predict(
            d_train,
            ntree_limit=n_tree_limit,
        )
        y_val_preds = gb_model.predict(
            d_val,
            ntree_limit=n_tree_limit,
        )
        y_test_preds = gb_model.predict(
            d_test,
            ntree_limit=n_tree_limit,
        )

        if self.threshold_predictions:
            y_train_preds = np.array(y_train_preds)
            y_train_preds = y_train_preds > 0.5
            y_train_preds = y_train_preds.astype(int)

            y_val_preds = np.array(y_val_preds)
            y_val_preds = y_val_preds > 0.5
            y_val_preds = y_val_preds.astype(int)

            y_test_preds = np.array(y_test_preds)
            y_test_preds = y_test_preds > 0.5
            y_test_preds = y_test_preds.astype(int)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        val_performance = balanced_accuracy_score(y_val, y_val_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if val_performance is None or val_performance is np.inf:
            val_error_rate = 1
        else:
            val_error_rate = 1 - val_performance

        res = {
            'train_accuracy': float(train_performance),
            'val_accuracy': float(val_performance),
            'test_accuracy': float(test_performance),
            'best_round': best_iteration,
        }

        return ({
            'loss': float(val_error_rate),  # this is the a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information - also mandatory
        })

    def refit(self, config):
        """Runs refit on the best configuration.

        The function refits on the best configuration. It then
        proceeds to train and test the network, this time combining
        the train and validation set together for training. Probably,
        in the future, a budget should be added too as an argument to
        the parameter.

        Parameters:
        -----------
            config: dict
                dictionary containing the sampled configurations by the optimizer
        Returns:
        --------
            res: dict
                Dictionary with the train and test accuracy.
        """
        xgboost_config = deepcopy(self.param)
        xgboost_config.update(config)

        if 'num_round' in xgboost_config:
            num_rounds = xgboost_config['num_round']
            del xgboost_config['num_round']
        else:
            num_rounds = 4000

        if xgboost_config['use_ohe'] == 'True':
            use_ohe = True
        else:
            use_ohe = False

        del xgboost_config['use_ohe']

        if 'use_imputation' in xgboost_config:
            apply_imputation = xgboost_config['use_imputation']
            del xgboost_config['use_imputation']
        else:
            # if no conditional imputation, always use it
            apply_imputation = True

        loader = Loader(
            task_id=self.task_id,
            val_fraction=0,
            seed=xgboost_config['seed'],
            apply_one_hot_encoding=use_ohe,
            apply_imputation=apply_imputation,
        )
        splits = loader.get_splits()

        X_train = splits['X_train']
        X_test = splits['X_test']
        y_train = splits['y_train']
        y_test = splits['y_test']

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_test = xgb.DMatrix(X_test, label=y_test)

        eval_results = {}
        gb_model = xgb.train(
            xgboost_config,
            d_train,
            num_rounds,
            feval=partial(balanced_error, self.threshold_predictions),
            evals=[(d_train, 'd_train'), (d_test, 'd_test')],
            evals_result=eval_results,
        )

        gb_model.save_model(
            os.path.join(
                self.output_directory,
                'xgboost_refit_model_dump.json',
            )
        )

        # TODO do something with eval_results
        # print(eval_results)

        # make prediction
        y_train_preds = gb_model.predict(d_train)
        y_test_preds = gb_model.predict(d_test)

        if self.threshold_predictions:
            y_train_preds = np.array(y_train_preds)
            y_train_preds = y_train_preds > 0.5
            y_train_preds = y_train_preds.astype(int)

            y_test_preds = np.array(y_test_preds)
            y_test_preds = y_test_preds > 0.5
            y_test_preds = y_test_preds.astype(int)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if test_performance is None or test_performance is np.inf:
            test_performance = 0

        res = {
            'train_accuracy': float(train_performance),
            'test_accuracy': float(test_performance),
        }

        return res

    @staticmethod
    def get_default_configspace(
        seed: int = 11,
        early_stopping: bool = False,
        conditional_imputation: bool = False,
    ) -> cs.ConfigurationSpace:
        """Get the hyperparameter search space.

        The function provides the configuration space that is
        used to generate the algorithm specific hyperparameter
        search space.

        Parameters:
        -----------
        seed: int
            The seed used to build the configuration space.
        Returns:
        --------
        config_space: cs.ConfigurationSpace
            Configuration space for XGBoost.
        """
        config_space = cs.ConfigurationSpace(seed=seed)
        # learning rate
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'eta',
                lower=0.001,
                upper=1,
                log=True,
            )
        )
        # l2 regularization
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'lambda',
                lower=1E-10,
                upper=1,
                log=True,
            )
        )
        # l1 regularization
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'alpha',
                lower=1E-10,
                upper=1,
                log=True,
            )
        )

        # not added directly because condition
        # has to be applied.
        booster = cs.CategoricalHyperparameter(
            'booster',
            choices=['gbtree', 'dart'],
        )
        config_space.add_hyperparameter(
            booster,
        )
        rate_drop = cs.UniformFloatHyperparameter(
            'rate_drop',
            lower=1e-10,
            upper=1-(1e-10),
            default_value=0.5,
        )
        config_space.add_hyperparameter(
            rate_drop,
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'gamma',
                lower=0.1,
                upper=1,
                log=True,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'colsample_bylevel',
                lower=0.1,
                upper=1,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'colsample_bynode',
                lower=0.1,
                upper=1,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'colsample_bytree',
                lower=0.5,
                upper=1,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformIntegerHyperparameter(
                'max_depth',
                lower=1,
                upper=20,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformIntegerHyperparameter(
                'max_delta_step',
                lower=0,
                upper=10,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'min_child_weight',
                lower=0.1,
                upper=20,
                log=True,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'subsample',
                lower=0.01,
                upper=1,
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'use_ohe',
                choices=['True', 'False'],
            )
        )

        if conditional_imputation:
            config_space.add_hyperparameter(
                cs.CategoricalHyperparameter(
                    'use_imputation',
                    choices=['True', 'False'],
                )
            )

        # if early stopping is activated, add the
        # number of stopping rounds as a hyperparameter.
        # Number of rounds is fixed at 4000.
        if early_stopping:
            config_space.add_hyperparameter(
                cs.UniformIntegerHyperparameter(
                    'early_stopping_rounds',
                    lower=1,
                    upper=20,
                )
            )
        else:
            # no early stopping activated, number of rounds
            # is a hyperparameter.
            config_space.add_hyperparameter(
                cs.UniformIntegerHyperparameter(
                    'num_round',
                    lower=1,
                    upper=1000,
                )
            )

        config_space.add_condition(
            cs.EqualsCondition(
                rate_drop,
                booster,
                'dart',
            )
        )

        return config_space

    @staticmethod
    def get_parameters(
        nr_classes: int,
        seed: int = 11,
        nr_threads: int = 1,
        task_id: int = 233088,
        output_directory: str = 'path_to_output',
    ) -> Dict[str, Union[int, str]]:
        """Get the parameters of the method.

        Get a dictionary based on the arguments given to the
        function, which will be used to as the initial configuration
        for the algorithm.

        Parameters:
        -----------
        nr_classes: int
            The number of classes in the dataset that will be used
            to train the model.
        seed: int
            The seed that will be used for the model.
        nr_threads: int
            The number of parallel threads that will be used for
            the model.
        task_id: int
            The id of the task that is used for the experiment.
        output_directory: str
            The path to the output directory where the results and
            model can be stored.

        Returns:
        --------
        param: dict
            A dictionary that will be used as a configuration for the
            algorithm.
        """
        param = {
            'disable_default_eval_metric': 1,
            'seed': seed,
            'nthread': nr_threads,
            'task_id': task_id,
            'output_directory': output_directory,
        }

        if nr_classes != 2:
            param.update(
                {
                    'objective': 'multi:softmax',
                    'num_class': nr_classes + 1,
                }
            )
        else:
            param.update(
                {
                    'objective': 'binary:logistic',

                }
            )

        return param


class TabNetWorker(Worker):

    def __init__(
        self,
        *args,
        param: dict,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.param = deepcopy(param)
        self.task_id = self.param['task_id']
        del self.param['task_id']

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.param['seed'])
        os.environ['OMP_NUM_THREADS'] = '1'

    def compute(self, config: dict, budget: float, **kwargs) -> Dict:
        """What should be computed for one TabNet worker.

        The function takes a configuration and a budget, it
        then uses the tabnet algorithm to generate a loss
        and other information.

        Parameters:
        -----------
            config: dict
                dictionary containing the sampled configurations by the optimizer
            budget: float
                amount of time/epochs/etc. the model can use to train

        Returns:
        --------
            dict:
                With the following mandatory arguments:
                'loss' (scalar)
                'info' (dict)
        """
        # Always activate imputation for TabNet.
        # No encoding needed, TabNet makes it's own embeddings.
        loader = Loader(
            task_id=self.task_id,
            seed=self.param['seed'],
            apply_one_hot_encoding=False,
            apply_imputation=True,
        )
        splits = loader.get_splits()

        X_train = splits['X_train']
        X_val = splits['X_val']
        X_test = splits['X_test']
        y_train = splits['y_train']
        y_val = splits['y_val']
        y_test = splits['y_test']

        categorical_information = loader.categorical_information
        assert categorical_information is not None
        _ = categorical_information['categorical_ind']
        categorical_columns = categorical_information['categorical_columns']
        categorical_dimensions = categorical_information['categorical_dimensions']

        # Default value if early stopping is not activated
        best_iteration = None

        clf = TabNetClassifier(
            n_a=config['na'],
            n_d=config['na'],
            n_steps=config['nsteps'],
            gamma=config['gamma'],
            lambda_sparse=config['lambda_sparse'],
            momentum=config['mb'],
            cat_idxs=categorical_columns,
            cat_dims=categorical_dimensions,
            seed=self.param['seed'],
            optimizer_params={
                'lr': config['learning_rate'],
            },
            scheduler_params={
                'step_size': config['decay_iterations'],
                'gamma': config['decay_rate'],
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
        )
        batch_size = config['batch_size']
        if batch_size == 32768:
            vbatch_size = config['vbatch_size1']
        elif batch_size == 16384:
            vbatch_size = config['vbatch_size2']
        elif batch_size == 8192:
            vbatch_size = config['vbatch_size3']
        elif batch_size == 4096:
            vbatch_size = config['vbatch_size4']
        elif batch_size == 2048:
            vbatch_size = config['vbatch_size5']
        elif batch_size == 1024:
            vbatch_size = config['vbatch_size6']
        elif batch_size == 512:
            vbatch_size = config['vbatch_size7']
        elif batch_size == 256:
            vbatch_size = config['vbatch_size8']
        else:
            raise ValueError('Illegal batch size given')

        early_stopping_activated = True if 'early_stopping_rounds' in config else False

        clf.fit(
            X_train=X_train,
            y_train=y_train,
            batch_size=batch_size,
            virtual_batch_size=vbatch_size,
            eval_set=[(X_val, y_val)],
            eval_name=['Validation'],
            eval_metric=['balanced_accuracy'],
            max_epochs=200,
            patience=config['early_stopping_rounds'] if early_stopping_activated else 0,
        )

        if early_stopping_activated:
            best_iteration = clf.best_epoch

        y_train_preds = clf.predict(X_train)
        y_val_preds = clf.predict(X_val)
        y_test_preds = clf.predict(X_test)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        val_performance = balanced_accuracy_score(y_val, y_val_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if val_performance is None or val_performance is np.inf:
            val_error_rate = 1
        else:
            val_error_rate = 1 - val_performance

        res = {
            'train_accuracy': float(train_performance),
            'val_accuracy': float(val_performance),
            'test_accuracy': float(test_performance),
            'best_round': best_iteration,
        }

        return ({
            'loss': float(val_error_rate),  # this is the a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information - also mandatory
        })

    def refit(self, config: dict) -> Dict:
        """Runs refit on the best configuration.

        The function refits on the best configuration. It then
        proceeds to train and test the network, this time combining
        the train and validation set together for training. Probably,
        in the future, a budget should be added too as an argument to
        the parameter.

        Parameters:
        -----------
            config: dict
                dictionary containing the sampled configurations by the optimizer
        Returns:
        --------
            res: dict
                Dictionary with the train and test accuracy.
        """
        # early stopping was activated in this experiment
        if 'max_epochs' in config:
            max_epochs = config['max_epochs']
        else:
            max_epochs = 200

        # Always activate imputation for TabNet.
        # No encoding needed, TabNet makes it's own embeddings
        loader = Loader(
            task_id=self.task_id,
            val_fraction=0,
            seed=self.param['seed'],
            apply_one_hot_encoding=False,
            apply_imputation=True,
        )

        splits = loader.get_splits()
        categorical_information = loader.categorical_information
        assert categorical_information is not None
        _ = categorical_information['categorical_ind']
        categorical_columns = categorical_information['categorical_columns']
        categorical_dimensions = categorical_information['categorical_dimensions']

        X_train = splits['X_train']
        X_test = splits['X_test']
        y_train = splits['y_train']
        y_test = splits['y_test']

        clf = TabNetClassifier(
            n_a=config['na'],
            n_d=config['na'],
            n_steps=config['nsteps'],
            gamma=config['gamma'],
            lambda_sparse=config['lambda_sparse'],
            momentum=config['mb'],
            cat_idxs=categorical_columns,
            cat_dims=categorical_dimensions,
            seed=self.param['seed'],
            optimizer_params={
                'lr': config['learning_rate'],
            },
            scheduler_params={
                'step_size': config['decay_iterations'],
                'gamma': config['decay_rate'],
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
        )
        batch_size = config['batch_size']
        if batch_size == 32768:
            vbatch_size = config['vbatch_size1']
        elif batch_size == 16384:
            vbatch_size = config['vbatch_size2']
        elif batch_size == 8192:
            vbatch_size = config['vbatch_size3']
        elif batch_size == 4096:
            vbatch_size = config['vbatch_size4']
        elif batch_size == 2048:
            vbatch_size = config['vbatch_size5']
        elif batch_size == 1024:
            vbatch_size = config['vbatch_size6']
        elif batch_size == 512:
            vbatch_size = config['vbatch_size7']
        elif batch_size == 256:
            vbatch_size = config['vbatch_size8']
        else:
            raise ValueError('Illegal batch size given')

        clf.fit(
            X_train=X_train, y_train=y_train,
            batch_size=batch_size,
            virtual_batch_size=vbatch_size,
            eval_metric=['balanced_accuracy'],
            max_epochs=max_epochs,
            patience=0,
        )

        y_train_preds = clf.predict(X_train)
        y_test_preds = clf.predict(X_test)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if test_performance is None or test_performance is np.inf:
            test_performance = 0

        res = {
            'train_accuracy': float(train_performance),
            'test_accuracy': float(test_performance),
        }

        return res

    @staticmethod
    def get_default_configspace(
        seed: int = 11,
    ) -> cs.ConfigurationSpace:
        """Get the hyperparameter search space.

        The function provides the configuration space that is
        used to generate the algorithm specific hyperparameter
        search space.

        Parameters:
        -----------
        seed: int
            The seed used to build the configuration space.
        Returns:
        --------
        config_space: cs.ConfigurationSpace
            Configuration space for TabNet.
        """
        config_space = cs.ConfigurationSpace(seed=seed)
        # learning rate
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'na',
                choices=[8, 16, 24, 32, 64, 128],
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'learning_rate',
                choices=[0.005, 0.01, 0.02, 0.025],
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'gamma',
                choices=[1.0, 1.2, 1.5, 2.0],
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'nsteps',
                choices=[3, 4, 5, 6, 7, 8, 9, 10],
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'lambda_sparse',
                choices=[0, 0.000001, 0.0001, 0.001, 0.01, 0.1],
            )
        )
        batch_size = cs.CategoricalHyperparameter(
            'batch_size',
            choices=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        )
        vbatch_size1 = cs.CategoricalHyperparameter(
            'vbatch_size1',
            choices=[256, 512, 1024, 2048, 4096],
        )
        vbatch_size2 = cs.CategoricalHyperparameter(
            'vbatch_size2',
            choices=[256, 512, 1024, 2048, 4096],
        )
        vbatch_size3 = cs.CategoricalHyperparameter(
            'vbatch_size3',
            choices=[256, 512, 1024, 2048, 4096],
        )
        vbatch_size4 = cs.CategoricalHyperparameter(
            'vbatch_size4',
            choices=[256, 512, 1024, 2048],
        )
        vbatch_size5 = cs.CategoricalHyperparameter(
            'vbatch_size5',
            choices=[256, 512, 1024],
        )
        vbatch_size6 = cs.CategoricalHyperparameter(
            'vbatch_size6',
            choices=[256, 512],
        )
        vbatch_size7 = cs.Constant(
            'vbatch_size7',
            256
        )
        vbatch_size8 = cs.Constant(
            'vbatch_size8',
            256
        )
        config_space.add_hyperparameter(
            batch_size
        )
        config_space.add_hyperparameters(
            [
                vbatch_size1,
                vbatch_size2,
                vbatch_size3,
                vbatch_size4,
                vbatch_size5,
                vbatch_size6,
                vbatch_size7,
                vbatch_size8,
            ]
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'decay_rate',
                choices=[0.4, 0.8, 0.9, 0.95],
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'decay_iterations',
                choices=[500, 2000, 8000, 10000, 20000],
            )
        )
        config_space.add_hyperparameter(
            cs.UniformIntegerHyperparameter(
                'early_stopping_rounds',
                lower=1,
                upper=20,
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'mb',
                choices=[0.6, 0.7, 0.8, 0.9, 0.95, 0.98],
            )
        )

        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size1,
                batch_size,
                32768,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size2,
                batch_size,
                16384,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size3,
                batch_size,
                8192,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size4,
                batch_size,
                4096,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size5,
                batch_size,
                2048,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size6,
                batch_size,
                1024,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size7,
                batch_size,
                512,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size8,
                batch_size,
                256,
            )
        )

        return config_space

    @staticmethod
    def get_parameters(
        seed: int = 11,
        task_id: int = 233088,
    ) -> Dict[str, Union[int, str]]:
        """Get the parameters of the method.

        Get a dictionary based on the arguments given to the
        function, which will be used to as the initial configuration
        for the algorithm.

        Parameters:
        -----------
        seed: int
            The seed that will be used for the model.
        task_id: int
            The id of the task that will be used for the experiment.

        Returns:
        --------
        param: dict
            A dictionary that will be used as a configuration for the
            algorithm.
        """
        param = {
            'task_id': task_id,
            'seed': seed,
        }

        return param


class CatBoostWorker(Worker):

    def __init__(
        self,
        *args,
        param: dict,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.param = deepcopy(param)
        self.task_id = self.param['task_id']
        self.seed = self.param['seed']
        self.output_directory = self.param['output_directory']
        del self.param['task_id']
        del self.param['seed']
        del self.param['output_directory']

    def compute(self, config: dict, budget: float, **kwargs) -> Dict:
        """What should be computed for one CatBoost worker.

        The function takes a configuration and a budget, it
        then uses the CatBoost algorithm to generate a loss
        and other information.

        Parameters:
        -----------
        config: dict
            dictionary containing the sampled configurations by the optimizer
        budget: float
            amount of time/epochs/etc. the model can use to train

        Returns:
        --------
        dict:
            With the following mandatory arguments:
            'loss' (scalar)
            'info' (dict)
        """
        # budget at the moment is not used because we do
        # not make use of multi-fidelity

        # Always activate imputation for CatBoost.
        # No encoding needed, CatBoost deals with it
        # natively.
        loader = Loader(
            task_id=self.task_id,
            seed=self.seed,
            apply_one_hot_encoding=False,
            apply_imputation=True,
        )
        splits = loader.get_splits()

        X_train = splits['X_train']
        X_val = splits['X_val']
        X_test = splits['X_test']
        y_train = splits['y_train']
        y_val = splits['y_val']
        y_test = splits['y_test']

        categorical_information = loader.categorical_information
        assert categorical_information is not None
        categorical_feature_indices = loader.categorical_information['categorical_columns']

        # Default value if early stopping is not activated
        best_iteration = None

        params = {
            'iterations': config['iterations'],
            'learning_rate': config['learning_rate'],
            'random_strength': config['random_strength'],
            'one_hot_max_size': config['one_hot_max_size'],
            'random_seed': self.seed,
            'l2_leaf_reg': config['l2_leaf_reg'],
            'bagging_temperature': config['bagging_temperature'],
            'leaf_estimation_iterations': config['leaf_estimation_iterations'],
        }

        model = CatBoostClassifier(
            **params,
        )

        model.fit(
            X_train,
            y_train,
            cat_features=categorical_feature_indices,
            eval_set=(X_val, y_val),
            plot=False,
        )

        y_train_preds = model.predict(X_train)
        y_val_preds = model.predict(X_val)
        y_test_preds = model.predict(X_test)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        val_performance = balanced_accuracy_score(y_val, y_val_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if val_performance is None or val_performance is np.inf:
            val_error_rate = 1
        else:
            val_error_rate = 1 - val_performance

        res = {
            'train_accuracy': float(train_performance),
            'val_accuracy': float(val_performance),
            'test_accuracy': float(test_performance),
            'best_round': best_iteration,
        }

        return ({
            'loss': float(val_error_rate),  # this is the a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information - also mandatory
        })

    def refit(self, config: dict) -> Dict:
        """Runs refit on the best configuration.

        The function refits on the best configuration. It then
        proceeds to train and test bohb, this time combining
        the train and validation set together for training. Probably,
        in the future, a budget should be added too as an argument to
        the parameter.

        Parameters:
        -----------
            config: dict
                dictionary containing the sampled configurations by the optimizer
        Returns:
        --------
            res: dict
                Dictionary with the train and test accuracy.
        """
        # Always activate imputation for CatBoost.
        loader = Loader(
            task_id=self.task_id,
            val_fraction=0,
            seed=self.seed,
            apply_one_hot_encoding=False,
            apply_imputation=True,
        )

        splits = loader.get_splits()
        X_train = splits['X_train']
        X_test = splits['X_test']
        y_train = splits['y_train']
        y_test = splits['y_test']

        categorical_information = loader.categorical_information
        assert categorical_information is not None
        categorical_feature_indices = loader.categorical_information['categorical_columns']

        params = {
            'iterations': config['iterations'],
            'learning_rate': config['learning_rate'],
            'random_strength': config['random_strength'],
            'one_hot_max_size': config['one_hot_max_size'],
            'random_seed': self.seed,
            'l2_leaf_reg': config['l2_leaf_reg'],
            'bagging_temperature': config['bagging_temperature'],
            'leaf_estimation_iterations': config['leaf_estimation_iterations'],
        }

        model = CatBoostClassifier(
            **params,
        )

        model.fit(
            X_train,
            y_train,
            cat_features=categorical_feature_indices,
            plot=False,
        )

        model.save_model(
            os.path.join(
                self.output_directory,
                'catboost_refit_model.dump',
            )
        )

        y_train_preds = model.predict(X_train)
        y_test_preds = model.predict(X_test)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if test_performance is None or test_performance is np.inf:
            test_performance = 0

        res = {
            'train_accuracy': float(train_performance),
            'test_accuracy': float(test_performance),
        }

        return res

    @staticmethod
    def get_default_configspace(
        seed: int = 11,
    ) -> cs.ConfigurationSpace:
        """Get the hyperparameter search space.

        The function provides the configuration space that is
        used to generate the algorithm specific hyperparameter
        search space.

        Parameters:
        -----------
        seed: int
            The seed used to build the configuration space.
        Returns:
        --------
        config_space: cs.ConfigurationSpace
            Configuration space for CatBoost.
        """
        config_space = cs.ConfigurationSpace(seed=seed)

        config_space.add_hyperparameter(
            cs.UniformIntegerHyperparameter(
                'iterations',
                lower=1,
                upper=1000,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'learning_rate',
                lower=1e-7,
                upper=1,
                log=True,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformIntegerHyperparameter(
                'random_strength',
                lower=1,
                upper=20,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformIntegerHyperparameter(
                'one_hot_max_size',
                lower=0,
                upper=25,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'l2_leaf_reg',
                lower=1,
                upper=10,
                log=True,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'bagging_temperature',
                lower=0,
                upper=1,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformIntegerHyperparameter(
                'leaf_estimation_iterations',
                lower=1,
                upper=10,
            )
        )

        return config_space

    @staticmethod
    def get_parameters(
            seed: int = 11,
            nr_classes: int = 2,
            task_id: int = 233088,
            output_directory: str = 'path_to_output',
    ) -> Dict[str, Union[int, str]]:
        """Get the parameters of the method.

        Get a dictionary based on the arguments given to the
        function, which will be used to as the initial configuration
        for the algorithm.

        Parameters:
        -----------
        seed: int
            The seed that will be used for the model.
        nr_classes: int
            The number of classes in the dataset, which in turn
            will be used to determine the loss.
        task_id: int
            The id of the task that will be used for the experiment.
        output_directory: str
            THe path where the output results will be stored.

        Returns:
        --------
        param: dict
            A dictionary that will be used as a configuration for the
            algorithm.
        """
        param = {
            'task_id': task_id,
            'seed': seed,
            'output_directory': output_directory,
        }

        if nr_classes != 2:
            param.update(
                {
                    'loss_function': 'MultiClass',
                    'eval_metric': metrics.Accuracy(),
                }
            )
        else:
            param.update(
                {
                    'loss_function': 'Logloss',
                    'eval_metric': metrics.BalancedAccuracy(),
                }
            )

        return param
