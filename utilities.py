from argparse import Namespace
from typing import Any, Callable, Dict, Optional, Tuple

import ConfigSpace
import pandas as pd
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

import dask.distributed

import openml
import numpy as np

from sklearn.model_selection import train_test_split

from smac.intensification.simple_intensifier import SimpleIntensifier
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
from smac.scenario.scenario import Scenario
from smac.facade.smac_ac_facade import SMAC4AC
from smac.runhistory.runhistory import RunHistory


def get_data(
    task_id: int,
    val_share: float = 0.25,
    test_size: float = 0.2,
    seed: int = 11,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Given  a task id and split size information, return
    the dataset splits based on a seed for the main algorithm
    to use.

    Args:
        task_id (int):
            The id of the task which will be used for the run.
        val_share (float):
            The validation split size from the train set.
        test_size (float):
            The test split size from the whole dataset.
        seed (int):
            The seed used for the dataset preparation.

    Returns:

        X_train, X_test, y_train, y_test, resampling_strategy_args, categorical indicator
            (tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict, np.ndarray]):

            The train examples, the test examples, the train labels, the test labels,
            the resampling strategy to be used and the categorical indicator for the features.
    """
    task = openml.tasks.get_task(task_id=task_id)
    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute,
    )

    # AutoPyTorch fails when it is given a y DataFrame with False and True values
    # and category as dtype. In its inner workings it uses sklearn which cannot
    # detect the column type.
    if isinstance(y[1], bool):
        y = y.astype('bool')

    # uncomment only for np.arrays
    """
    # patch categorical values to string
    for index_nr, categorical_feature in enumerate(categorical_indicator):
        if categorical_feature:
            X[index_nr] = X[index_nr].astype("category")
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )
    resampling_strategy_args = {
        'val_share': val_share,
    }

    """
    This was an earlier fix to the AutoPyTorch failures for imbalanced datasets. In particular
    having variables with only null values in the train set. Now this is handled inside AutoPyTorch.
    
    train_column_nan_info = X_train.isna().all()
    test_column_nan_info = X_test.isna().all()
    only_nan_columns = [label for label, value in train_column_nan_info.items() if value]
    test_nan_columns = [label for label, value in test_column_nan_info.items() if value]
    only_nan_columns.extend(test_nan_columns)
    only_nan_columns = set(only_nan_columns)
    X_train.drop(only_nan_columns, axis='columns', inplace=True)
    X_test.drop(only_nan_columns, axis='columns', inplace=True)
    """
    # TODO turn this into a dictionary

    return X_train, X_test, y_train, y_test, resampling_strategy_args, categorical_indicator


def get_smac_object(
    scenario_dict: Dict[str, Any],
    seed: int,
    ta: Callable,
    ta_kwargs: Dict[str, Any],
    n_jobs: int,
    initial_budget: int,
    max_budget: int,
    dask_client: Optional[dask.distributed.Client],
) -> SMAC4AC:
    """
    This function returns an SMAC object that is gonna be used as
    optimizer of pipelines.

    Args:
        scenario_dict (typing.Dict[str, typing.Any]): constrain on how to run
            the jobs.
        seed (int): to make the job deterministic.
        ta (typing.Callable): the function to be intensified by smac.
        ta_kwargs (typing.Dict[str, typing.Any]): Arguments to the above ta.
        n_jobs (int): Amount of cores to use for this task.
        initial_budget (int):
            The initial budget for a configuration.
        max_budget (int):
            The maximal budget for a configuration.
        dask_client (dask.distributed.Client): User provided scheduler.

    Returns:
        (SMAC4AC): sequential model algorithm configuration object
    """
    # multi-fidelity is disabled, that is why initial_budget and max_budget
    # are not used.
    rh2EPM = RunHistory2EPM4LogCost

    return SMAC4AC(
        scenario=Scenario(scenario_dict),
        rng=seed,
        runhistory2epm=rh2EPM,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        initial_configurations=None,
        run_id=seed,
        intensifier=SimpleIntensifier,
        dask_client=dask_client,
        n_jobs=n_jobs,
    )


def get_updates_for_regularization_cocktails(
    categorical_indicator: np.ndarray,
    args: Namespace,
) -> Tuple[Dict, HyperparameterSearchSpaceUpdates, Dict]:
    """
    These updates replicate the regularization cocktail paper search space.

    Args:
        categorical_indicator (np.ndarray)
            An array that indicates whether a feature is categorical or not.
        args (Namespace):
            The different updates for the setup of the run, mostly updates
            for the different regularization ingredients.

    Returns:
    ________
        pipeline_update, search_space_updates, include_updates (Tuple[dict, HyperparameterSearchSpaceUpdates, dict]):
            The pipeline updates like number of epochs, budget, seed etc.
            The search space updates like setting different hps to different values or ranges.
            Lastly include updates, which can be used to include different features.
    """
    augmentation_names_to_trainers = {
        'mixup': 'MixUpTrainer',
        'cutout': 'RowCutOutTrainer',
        'cutmix': 'RowCutMixTrainer',
        'standard': 'StandardTrainer',
        'adversarial': 'AdversarialTrainer',
    }

    include_updates = dict()
    include_updates['network_init'] = ['NoInit']

    has_cat_features = any(categorical_indicator)
    has_numerical_features = not all(categorical_indicator)
    search_space_updates = HyperparameterSearchSpaceUpdates()

    # architecture head
    search_space_updates.append(
        node_name='network_head',
        hyperparameter='__choice__',
        value_range=['no_head'],
        default_value='no_head',
    )
    search_space_updates.append(
        node_name='network_head',
        hyperparameter='no_head:activation',
        value_range=['relu'],
        default_value='relu',
    )

    # backbone architecture
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='__choice__',
        value_range=['ShapedResNetBackbone'],
        default_value='ShapedResNetBackbone',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:resnet_shape',
        value_range=['brick'],
        default_value='brick',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:num_groups',
        value_range=[2],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:blocks_per_group',
        value_range=[2],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:output_dim',
        value_range=[512],
        default_value=512,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:max_units',
        value_range=[512],
        default_value=512,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:activation',
        value_range=['relu'],
        default_value='relu',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:shake_shake_update_func',
        value_range=['even-even'],
        default_value='even-even',
    )

    # training updates
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='__choice__',
        value_range=['CosineAnnealingWarmRestarts'],
        default_value='CosineAnnealingWarmRestarts',
    )
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='CosineAnnealingWarmRestarts:n_restarts',
        value_range=[3],
        default_value=3,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='__choice__',
        value_range=['AdamWOptimizer'],
        default_value='AdamWOptimizer',
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:lr',
        value_range=[1e-3],
        default_value=1e-3,
    )
    search_space_updates.append(
        node_name='data_loader',
        hyperparameter='batch_size',
        value_range=[128],
        default_value=128,
    )

    # preprocessing
    search_space_updates.append(
        node_name='feature_preprocessor',
        hyperparameter='__choice__',
        value_range=['NoFeaturePreprocessor'],
        default_value='NoFeaturePreprocessor',
    )

    if has_numerical_features:
        search_space_updates.append(
            node_name='imputer',
            hyperparameter='numerical_strategy',
            value_range=['median'],
            default_value='median',
        )
        search_space_updates.append(
            node_name='scaler',
            hyperparameter='__choice__',
            value_range=['StandardScaler'],
            default_value='StandardScaler',
        )

    if has_cat_features:
        search_space_updates.append(
            node_name='encoder',
            hyperparameter='__choice__',
            value_range=['OneHotEncoder'],
            default_value='OneHotEncoder',
        )
        include_updates['network_embedding'] = ['LearnedEntityEmbedding']

    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:beta1',
        value_range=[0.9],
        default_value=0.9,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:beta2',
        value_range=[0.999],
        default_value=0.999,
    )

    # if the cash formulation of the cocktail is not activated,
    # otherwise the methods activation will be chosen by the SMBO optimizer.
    if not args.cash_cocktail:
        # regularization ingredients updates
        search_space_updates.append(
            node_name='network_backbone',
            hyperparameter='ShapedResNetBackbone:use_dropout',
            value_range=args.use_dropout,
            default_value=args.use_dropout[0],
        )
        search_space_updates.append(
            node_name='network_backbone',
            hyperparameter='ShapedResNetBackbone:use_batch_norm',
            value_range=args.use_batch_normalization,
            default_value=args.use_batch_normalization[0],
        )
        search_space_updates.append(
            node_name='network_backbone',
            hyperparameter='ShapedResNetBackbone:use_skip_connection',
            value_range=args.use_skip_connection,
            default_value=args.use_skip_connection[0],
        )

        multi_branch_choice = [args.mb_choice]

        search_space_updates.append(
            node_name='network_backbone',
            hyperparameter='ShapedResNetBackbone:multi_branch_choice',
            value_range=multi_branch_choice,
            default_value=multi_branch_choice[0],
        )

        search_space_updates.append(
            node_name='optimizer',
            hyperparameter='AdamWOptimizer:use_weight_decay',
            value_range=args.use_weight_decay,
            default_value=args.use_weight_decay[0],
        )

        trainer_choice = [augmentation_names_to_trainers[args.augmentation]]

        search_space_updates.append(
            node_name='trainer',
            hyperparameter='__choice__',
            value_range=trainer_choice,
            default_value=trainer_choice[0],
        )

        search_space_updates.append(
            node_name='trainer',
            hyperparameter=f'{trainer_choice[0]}:weighted_loss',
            value_range=[1],
            default_value=1,
        )
        search_space_updates.append(
            node_name='trainer',
            hyperparameter=f'{trainer_choice[0]}:use_lookahead_optimizer',
            value_range=args.use_lookahead,
            default_value=args.use_lookahead[0],
        )
        search_space_updates.append(
            node_name='trainer',
            hyperparameter=f'{trainer_choice[0]}:use_stochastic_weight_averaging',
            value_range=args.use_swa,
            default_value=args.use_swa[0],
        )
        search_space_updates.append(
            node_name='trainer',
            hyperparameter=f'{trainer_choice[0]}:use_snapshot_ensemble',
            value_range=args.use_se,
            default_value=args.use_se[0],
        )

    # No early stopping and train on gpu
    pipeline_update = {
        'early_stopping': -1,
        'min_epochs': args.epochs,
        'epochs': args.epochs,
        "device": 'cpu',
    }

    return pipeline_update, search_space_updates, include_updates


def get_incumbent_results(
    run_history_file: str,
    search_space: ConfigSpace.ConfigurationSpace,
) -> Tuple[ConfigSpace.Configuration, float]:
    """
    Get the incumbent configuration and performance from the previous run HPO
    search with AutoPytorch.

    Args:
        run_history_file (str):
            The path where the AutoPyTorch search data is located.
        search_space (ConfigSpace.ConfigurationSpace):
            The ConfigurationSpace that was previously used for the HPO
            search space.

    Returns:
        config, incumbent_run_value (Tuple[ConfigSpace.Configuration, float]):
            The incumbent configuration found from HPO search and the validation
            performance it achieved.

    """
    run_history = RunHistory()
    run_history.load_json(
        run_history_file,
        search_space,
    )

    run_history_data = run_history.data
    sorted_runvalue_by_cost = sorted(run_history_data.items(), key=lambda item: item[1].cost)
    incumbent_run_key, incumbent_run_value = sorted_runvalue_by_cost[0]
    config = run_history.ids_config[incumbent_run_key.config_id]
    return config, incumbent_run_value
