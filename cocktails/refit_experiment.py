import argparse
import json
import os
import random
import time
import warnings

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.resampling_strategy import NoResamplingStrategyTypes
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch import metrics

import numpy as np

from utilities import \
    get_data, \
    get_incumbent_results, \
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
    description='Refit autoPyTorch on a benchmark.'
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
    default=1300,
)
parser.add_argument(
    '--func_eval_time',
    type=int,
    default=700,
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


if __name__ == '__main__':

    # Setting up reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

    refit_out_dir = os.path.join(output_dir, 'refit')
    refit_tmp_dir = os.path.join(temp_dir, 'refit')

    ############################################################################
    # Build and fit a classifier
    # ==========================
    api = TabularClassificationTask(
        temporary_directory=refit_tmp_dir,
        output_directory=refit_out_dir,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        resampling_strategy=NoResamplingStrategyTypes.no_resampling,
        ensemble_size=1,
        ensemble_nbest=1,
        max_models_on_disc=1,
        include_components=include_updates,
        search_space_updates=search_space_updates,
        seed=args.seed,
        n_jobs=1,
    )

    api.set_pipeline_config(**pipeline_update)
    ############################################################################
    # Refit the hp configuration
    # ==========================
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
        validator=input_validator,
        seed=args.seed,
        resampling_strategy=NoResamplingStrategyTypes.no_resampling,
    )
    dataset.is_small_preprocess = False
    print(f"Fitting pipeline with {args.epochs} epochs")

    search_space = api.get_search_space(dataset)

    # There has been an hpo search, find the best hyperparameter configuration.
    run_history_path = os.path.join(
        temp_dir,
        'smac3-output',
        'run_{}'.format(args.seed),
        'runhistory.json',
    )

    inc_config, inc_value = get_incumbent_results(run_history_path, search_space)

    print(f'The value that the incumbent had on the validation set before the refit:{inc_value}')
    print(f"Incumbent configuration: {inc_config}")

    fitted_pipeline, run_info, run_value, dataset = api.fit_pipeline(
        configuration=inc_config,
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
