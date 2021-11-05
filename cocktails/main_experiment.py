import argparse
import json
import os
import pickle
import random
import time
import warnings

# this corresponds to the number of threads
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes, NoResamplingStrategyTypes
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch import metrics

import numpy as np

from utilities import \
    get_data, \
    get_incumbent_results, \
    get_smac_object, \
    get_updates_for_regularization_cocktails


def str2bool(v):
    if isinstance(v, bool):
        return [v, ]
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return [True, ]
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return [False, ]
    elif v.lower() == 'conditional':
        return [True, False]
    else:
        raise argparse.ArgumentTypeError('No valid value given.')


parser = argparse.ArgumentParser(
    description='Run AutoPyTorch on a benchmark.',
)
# experiment setup arguments
parser.add_argument(
    '--task_id',
    type=int,
    default=233088,
)
parser.add_argument(
    '--wall_time',
    type=int,
    default=9000,
)
parser.add_argument(
    '--func_eval_time',
    type=int,
    default=1000,
)
parser.add_argument(
    '--epochs',
    type=int,
    default=105,
)
parser.add_argument(
    '--seed',
    type=int,
    default=11,
)
parser.add_argument(
    '--tmp_dir',
    type=str,
    default='./runs/autoPyTorch_cocktails',
)
parser.add_argument(
    '--output_dir',
    type=str,
    default='./runs/autoPyTorch_cocktails',
)
parser.add_argument(
    '--nr_workers',
    type=int,
    default=1,
)
parser.add_argument(
    '--nr_threads',
    type=int,
    default=1,
)
parser.add_argument(
    '--cash_cocktail',
    help='If the regularization cocktail should be used.',
    type=bool,
    default=False,
)

# regularization ingredient arguments
parser.add_argument(
    '--use_swa',
    help='If stochastic weight averaging should be used.',
    type=str2bool,
    nargs='?',
    const=[True],
    default=[False],
)
parser.add_argument(
    '--use_se',
    help='If snapshot ensembling should be used.',
    type=str2bool,
    nargs='?',
    const=[True],
    default=[False],
)
parser.add_argument(
    '--use_lookahead',
    help='If the lookahead optimizing technique should be used.',
    type=str2bool,
    nargs='?',
    const=[True],
    default=[False],
)
parser.add_argument(
    '--use_weight_decay',
    help='If weight decay regularization should be used.',
    type=str2bool,
    nargs='?',
    const=[True],
    default=[False],
)
parser.add_argument(
    '--use_batch_normalization',
    help='If batch normalization regularization should be used.',
    type=str2bool,
    nargs='?',
    const=[True],
    default=[False],
)
parser.add_argument(
    '--use_skip_connection',
    help='If skip connections should be used. '
         'Turns the network into a residual network.',
    type=str2bool,
    nargs='?',
    const=[True],
    default=[False],
)
parser.add_argument(
    '--use_dropout',
    help='If dropout regularization should be used.',
    type=str2bool,
    nargs='?',
    const=[True],
    default=[False],
)
parser.add_argument(
    '--mb_choice',
    help='Multibranch network regularization. '
         'Only active when skip_connection is active.',
    type=str,
    choices=['none', 'shake-shake', 'shake-drop'],
    default='none',
)
parser.add_argument(
    '--augmentation',
    help='If methods that augment examples should be used',
    type=str,
    choices=['mixup', 'cutout', 'cutmix', 'standard', 'adversarial'],
    default='standard',
)


args = parser.parse_args()
options = vars(args)
print(options)


hps_for_method = {
    'stochastic_weight_averaging': 0,
    'snapshot_ensembling': 0,
    'batch_normalization': 0,
    'skip_connection': 0,
    'shake_shake': 0,
    'adversarial_training': 0,
    'cutmix': 1,
    'mixup': 1,
    'weight_decay': 1,
    'shake_drop': 1,
    'lookahead': 2,
    'cutout': 2,
    'dropout': 2,
}


if __name__ == '__main__':

    # Setting up reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    number_of_configurations_limit = 0

    if args.cash_cocktail:
        # for the cocktail we use 840 configurations
        number_of_configurations_limit = 840
    else:
        method_number_of_hps = 0
        if any(args.use_swa):
            method_number_of_hps = hps_for_method['stochastic_weight_averaging']
        elif any(args.use_se):
            method_number_of_hps = hps_for_method['snapshot_ensembling']
        elif any(args.use_batch_normalization):
            method_number_of_hps = hps_for_method['batch_normalization']
        elif any(args.use_skip_connection) and args.mb_choice == 'none':
            method_number_of_hps = hps_for_method['skip_connection']
        elif any(args.use_skip_connection) and args.mb_choice == 'shake-shake':
            method_number_of_hps = hps_for_method['shake_shake']
        elif any(args.use_skip_connection) and args.mb_choice == 'shake-drop':
            method_number_of_hps = hps_for_method['shake_drop']
        elif args.augmentation == 'cutmix':
            method_number_of_hps = hps_for_method['cutmix']
        elif args.augmentation == 'mixup':
            method_number_of_hps = hps_for_method['mixup']
        elif args.augmentation == 'cutout':
            method_number_of_hps = hps_for_method['cutout']
        elif args.augmentation == 'adversarial':
            method_number_of_hps = hps_for_method['adversarial_training']
        elif any(args.use_dropout):
            method_number_of_hps = hps_for_method['dropout']
        elif any(args.use_weight_decay):
            method_number_of_hps = hps_for_method['weight_decay']
        elif any(args.use_lookahead):
            method_number_of_hps = hps_for_method['lookahead']

        number_of_configurations_limit = 40 * method_number_of_hps

    print(f'Number of configurations limit: {number_of_configurations_limit}')

    ############################################################################
    # Data Loading
    # ============
    start_time = time.time()

    X_train, X_test, y_train, y_test, resampling_strategy_args, categorical_indicator = get_data(
        task_id=args.task_id,
        seed=args.seed,
    )

    pipeline_update, search_space_updates, include_updates = get_updates_for_regularization_cocktails(
        categorical_indicator,
        args,
    )

    output_dir = os.path.expanduser(
        os.path.join(
            args.output_dir,
            f'{args.seed}',
            f'{args.task_id}',
            f'{args.task_id}_out',
        )
    )
    temp_dir = os.path.expanduser(
        os.path.join(
            args.tmp_dir,
            f'{args.seed}',
            f'{args.task_id}',
            f'{args.task_id}_tmp',
        )
    )

    ############################################################################
    # Build and fit a classifier
    # ==========================
    # if we use HPO, we can use multiple workers in parallel
    if number_of_configurations_limit != 0:
        nr_workers = args.nr_workers
    else:
        nr_workers = 1

    api = TabularClassificationTask(
        temporary_directory=temp_dir,
        output_directory=output_dir,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        resampling_strategy=HoldoutValTypes.stratified_holdout_validation,
        resampling_strategy_args=resampling_strategy_args,
        ensemble_size=1,
        ensemble_nbest=1,
        max_models_on_disc=10,
        include_components=include_updates,
        search_space_updates=search_space_updates,
        seed=args.seed,
        n_jobs=nr_workers,
        n_threads=args.nr_threads,
    )

    api.set_pipeline_config(**pipeline_update)
    ############################################################################
    # Search for the best hp configuration
    # ====================================
    # We search for the best hp configuration only in the case of a cocktail ingredient
    # that has hyperparameters.
    if number_of_configurations_limit != 0:
        api.search(
            X_train=X_train.copy(),
            y_train=y_train.copy(),
            X_test=X_test.copy(),
            y_test=y_test.copy(),
            optimize_metric='balanced_accuracy',
            total_walltime_limit=args.wall_time,
            memory_limit=12000,
            func_eval_time_limit_secs=args.func_eval_time,
            enable_traditional_pipeline=False,
            get_smac_object_callback=get_smac_object,
            smac_scenario_args={
                'runcount_limit': number_of_configurations_limit,
            },
        )

        # Dump the pipeline for reuse in the future
        pickle_directory = os.path.expanduser(
            os.path.join(
                args.output_dir,
                f'{args.seed}',
                f'{args.task_id}',
                'estimator.pickle',
            )
        )
        with open(pickle_directory, 'wb') as file_handle:
            pickle.dump(api, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    ############################################################################
    # Refit on the best hp configuration
    # ==================================
    input_validator = TabularInputValidator(
        is_classification=True,
    )
    input_validator.fit(
        X_train=X_train.copy(),
        y_train=y_train.copy(),
        X_test=X_test.copy(),
        y_test=y_test.copy(),
    )

    dataset = TabularDataset(
        X=X_train,
        Y=y_train,
        X_test=X_test,
        Y_test=y_test,
        seed=args.seed,
        validator=input_validator,
        resampling_strategy=NoResamplingStrategyTypes.no_resampling,
    )
    dataset.is_small_preprocess = False
    print(f"Fitting pipeline with {args.epochs} epochs")

    search_space = api.get_search_space(dataset)
    # only when we perform hpo will there be an incumbent configuration
    # otherwise take a default configuration.
    if number_of_configurations_limit != 0:
        configuration, incumbent_run_value = get_incumbent_results(
            os.path.join(
                temp_dir,
                'smac3-output',
                'run_{}'.format(args.seed),
                'runhistory.json'),
                search_space,
            )
        print(f"Incumbent configuration: {configuration}")
        print(f"Incumbent trajectory: {api.trajectory}")
    else:
        # default configuration
        configuration = search_space.get_default_configuration()
        print(f"Default configuration: {configuration}")

    fitted_pipeline, run_info, run_value, dataset = api.fit_pipeline(
        configuration=configuration,
        budget_type='epochs',
        budget=args.epochs,
        dataset=dataset,
        run_time_limit_secs=args.func_eval_time,
        eval_metric='balanced_accuracy',
        memory_limit=12000,
    )

    X_train = dataset.train_tensors[0]
    y_train = dataset.train_tensors[1]
    X_test = dataset.test_tensors[0]
    y_test = dataset.test_tensors[1]

    train_predictions = fitted_pipeline.predict(X_train)
    test_predictions = fitted_pipeline.predict(X_test)

    # Store the predictions if things go south
    with open(os.path.join(output_dir, f"predictions_{args.task_id}.pickle"), 'wb') as handle:
        pickle.dump(test_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_dir, f"truth_{args.task_id}.pickle"), 'wb') as handle:
        pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_balanced_accuracy = metrics.balanced_accuracy(y_train, train_predictions.squeeze())
    test_balanced_accuracy = metrics.balanced_accuracy(y_test, test_predictions.squeeze())
    duration = time.time() - start_time

    print(f'Final Train Balanced accuracy: {train_balanced_accuracy}')
    print(f'Final Test Balanced accuracy: {test_balanced_accuracy}')
    print(f'Time taken: {duration}')

    result_directory = os.path.expanduser(
        os.path.join(
            args.output_dir,
            f'{args.seed}',
            f'{args.task_id}',
            'final_result.json',
        )
    )
    result_dict = {
        'train balanced accuracy': train_balanced_accuracy,
        'test balanced accuracy': test_balanced_accuracy,
        'task_id': args.task_id,
        'duration': duration,
    }

    with open(result_directory, 'w') as file:
        json.dump(result_dict, file)
