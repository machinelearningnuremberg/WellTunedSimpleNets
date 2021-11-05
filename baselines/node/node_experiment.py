import argparse
import json
import os
import time
from typing import List

import numpy as np

import openml

from category_encoders import LeaveOneOutEncoder

from qhoptim.pyt import QHAdam

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score

import torch, torch.nn as nn
import torch.nn.functional as F

import lib
from lib.utils import check_numpy, process_in_chunks


def get_task_list(
    benchmark_task_file: str = 'path/to/tasks.txt',
) -> List[int]:
    """Get the task id list.
    Goes through the given file and collects all of the task
    ids.
    Parameters:
    -----------
    benchmark_task_file: str
        A string to the path of the benchmark task file. Including
        the task file name.
    Returns:
    --------
    benchmark_task_ids - list
        A list of all the task ids for the benchmark.
    """
    with open(os.path.join(benchmark_task_file), 'r') as f:
        benchmark_info_str = f.readline()
        benchmark_task_ids = [int(task_id) for task_id in benchmark_info_str.split(' ')]

    return benchmark_task_ids


def get_data(
    task_id: int,
    test_size: float = 0.2,
    validation_size: float = 0.25,
    seed: int = 11,
):
    task = openml.tasks.get_task(task_id=task_id)
    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute,
    )
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    if validation_size != 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=validation_size,
            random_state=seed,
            stratify=y_train,
        )
    else:
        X_val = None
        y_val = None

    # the code below drops columns that are
    # completely null in the train set, however, are not null in the validation
    # and test set.
    train_column_nan_info = X_train.isna().all()
    only_nan_columns = [label for label, value in train_column_nan_info.items() if value]
    only_nan_columns = set(only_nan_columns)
    X_train.drop(only_nan_columns, axis='columns', inplace=True)
    X_test.drop(only_nan_columns, axis='columns', inplace=True)

    if validation_size != 0:
        X_val.drop(only_nan_columns, axis='columns', inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    column_names = X_train.columns.to_numpy()
    categorical_column_names = [column_name for column_indicator, column_name in zip(categorical_indicator, column_names) if column_indicator]

    cat_encoder.fit(X_train[categorical_column_names], y_train)
    X_train[categorical_column_names] = cat_encoder.transform(X_train[categorical_column_names])
    if validation_size != 0:
        X_val[categorical_column_names] = cat_encoder.transform(X_val[categorical_column_names])
        X_val = X_val.values.astype('float32')

    X_test[categorical_column_names] = cat_encoder.transform(X_test[categorical_column_names])
    X_train = X_train.values.astype('float32')
    X_test = X_test.values.astype('float32')

    dataset_name = dataset.name

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'name': dataset_name,
    }


def get_node_dataset(
    task_id: int,
    test_size: float = 0.2,
    validation_size: float = 0.25,
    seed: int = 11,
    refit=False,
):
    if not refit:
        data_splits = get_data(
            task_id,
            seed=seed,
            test_size=test_size,
            validation_size=validation_size,
        )

    else:
        data_splits = get_data(
            task_id,
            seed=seed,
            test_size=test_size,
            validation_size=0,
        )

    node_dataset = lib.Dataset(
        dataset=data_splits['name'],
        random_state=seed,
        quantile_transform=True,
        quantile_noise=1e-3,
        X_train=data_splits['X_train'],
        X_valid=data_splits['X_val'],
        X_test=data_splits['X_test'],
        y_train=data_splits['y_train'],
        y_valid=data_splits['y_val'],
        y_test=data_splits['y_test'],
    )

    return node_dataset


def evaluate_balanced_classification_error(
    trainer,
    X_test,
    y_test,
    device,
    batch_size=128,
):
    X_test = torch.as_tensor(X_test, device=device)
    y_test = check_numpy(y_test)
    trainer.train(False)
    with torch.no_grad():
        logits = process_in_chunks(trainer.model, X_test, batch_size=batch_size)
        logits = check_numpy(logits)
        y_pred = np.argmax(logits, axis=1)

        error_rate = 1 - balanced_accuracy_score(y_test, y_pred)

    return error_rate


def evaluate_node(
    data,
    config,
    device,
    experiment_name,
    epochs=105,
    batch_size=128,
    refit=False,
):
    config_start_time = time.time()
    num_examples = data.X_train.shape[0]
    num_features = data.X_train.shape[1]
    num_classes = len(set(data.y_train))

    model = nn.Sequential(
        lib.DenseBlock(
            num_features,
            layer_dim=config['total_tree_count'],
            num_layers=config['num_layers'],
            tree_dim=num_classes + 1,
            flatten_output=False,
            depth=config['tree_depth'],
            choice_function=lib.entmax15,
            bin_function=lib.entmoid15,
        ),
        lib.Lambda(lambda x: x[..., :num_classes].mean(dim=-2)),
    ).to(device)

    with torch.no_grad():
        res = model(torch.as_tensor(data.X_train[:batch_size], device=device))
        # trigger data-aware init

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    trainer = lib.Trainer(
        model=model,
        loss_function=F.cross_entropy,
        experiment_name=experiment_name,
        warm_start=False,
        Optimizer=QHAdam,
        optimizer_params=dict(nus=(0.7, 1.0), betas=(0.95, 0.998)),
        verbose=True,
        n_last_checkpoints=5
    )

    loss_history, err_history = [], []
    best_val_err = 1.0
    best_step = 0

    # calculate the number of early stopping rounds to
    # be around 10 epochs. Allow incomplete batches.
    number_batches_epoch = int(np.ceil(num_examples / batch_size))
    early_stopping_rounds = 10 * number_batches_epoch
    report_frequency = number_batches_epoch
    print(early_stopping_rounds)
    # Flag if early stopping is hit or not
    early_stopping_activated = False

    for batch in lib.iterate_minibatches(
            data.X_train,
            data.y_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
    ):
        metrics = trainer.train_on_batch(
            *batch,
            device=device,
        )

        loss_history.append(metrics['loss'].item())

        # calculate the information below on every epoch
        if trainer.step % report_frequency == 0:
            train_err = evaluate_balanced_classification_error(
                trainer,
                data.X_train,
                data.y_train,
                device=device,
                batch_size=batch_size,
            )
            if not refit:
                val_err = evaluate_balanced_classification_error(
                    trainer,
                    data.X_valid,
                    data.y_valid,
                    device=device,
                    batch_size=batch_size,
                )
                err_history.append(val_err)
                print("Val Error Rate: %0.5f" % (val_err))

                if val_err < best_val_err:
                    best_val_err = val_err
                    best_step = trainer.step
                    trainer.save_checkpoint(tag='best')

            print("Loss %.5f" % (metrics['loss']))
            print("Train Error Rate: %0.5f" % (train_err))

        if not refit:
            if trainer.step > best_step + early_stopping_rounds:
                print('BREAK. There is no improvement for {} steps'.format(early_stopping_rounds))
                print("Best step: ", best_step)
                print("Best Val Error Rate: %0.5f" % (best_val_err))
                early_stopping_activated = True
                break

    config_duration = time.time() - config_start_time

    if early_stopping_activated:
        best_epoch = int(best_step / report_frequency)
    else:
        best_epoch = int(trainer.step / report_frequency)
        # save the model in the end
        trainer.save_checkpoint(tag='best')

    # we will always have a best checkpoint, be it
    # from early stopping, be it from the normal training.
    trainer.load_checkpoint(tag='best')
    train_error_rate = evaluate_balanced_classification_error(
        trainer,
        data.X_train,
        data.y_train,
        device=device,
        batch_size=batch_size,
    )
    if not refit:
        val_error_rate = evaluate_balanced_classification_error(
            trainer,
            data.X_valid,
            data.y_valid,
            device=device,
            batch_size=batch_size,
        )
    else:
        val_error_rate = None

    test_error_rate = evaluate_balanced_classification_error(
        trainer,
        data.X_test,
        data.y_test,
        device=device,
        batch_size=batch_size,
    )

    run_information = {
        'train_error': train_error_rate,
        'val_error': val_error_rate,
        'test_error': test_error_rate,
        'best_epoch': best_epoch,
        'duration': config_duration
    }

    return run_information


def predict_node(
    data,
    config,
    device,
    experiment_name,
    batch_size=128,
    refit=True,
):
    num_features = data.X_train.shape[1]
    num_classes = len(set(data.y_train))

    model = nn.Sequential(
        lib.DenseBlock(
            num_features,
            layer_dim=config['total_tree_count'],
            num_layers=config['num_layers'],
            tree_dim=num_classes + 1,
            flatten_output=False,
            depth=config['tree_depth'],
            choice_function=lib.entmax15,
            bin_function=lib.entmoid15,
        ),
        lib.Lambda(lambda x: x[..., :num_classes].mean(dim=-2)),
    ).to(device)

    with torch.no_grad():
        res = model(torch.as_tensor(data.X_train[:batch_size], device=device))
        # trigger data-aware init

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    trainer = lib.Trainer(
        model=model,
        warm_start=True,
        loss_function=F.cross_entropy,
        experiment_name=experiment_name,
        Optimizer=QHAdam,
        optimizer_params=dict(nus=(0.7, 1.0), betas=(0.95, 0.998)),
        verbose=True,
        n_last_checkpoints=5
    )
    # we will always have a best checkpoint, be it
    # from early stopping, be it from the normal training.
    trainer.load_checkpoint(tag='best')

    train_error_rate = evaluate_balanced_classification_error(
        trainer,
        data.X_train,
        data.y_train,
        device=device,
        batch_size=batch_size,
    )
    if not refit:
        val_error_rate = evaluate_balanced_classification_error(
            trainer,
            data.X_valid,
            data.y_valid,
            device=device,
            batch_size=batch_size,
        )
    else:
        val_error_rate = None

    test_error_rate = evaluate_balanced_classification_error(
        trainer,
        data.X_test,
        data.y_test,
        device=device,
        batch_size=batch_size,
    )

    run_information = {
        'train_error': train_error_rate,
        'val_error': val_error_rate,
        'test_error': test_error_rate,
    }

    return run_information

parser = argparse.ArgumentParser(
    description='Run node on a benchmark'
)
# experiment setup arguments
parser.add_argument(
    '--task_id',
    type=int,
    default=233090,
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
)
parser.add_argument(
    '--epochs',
    type=int,
    default=1,
)
parser.add_argument(
    '--test_size',
    type=float,
    default=0.2,
)
parser.add_argument(
    '--validation_size',
    type=float,
    default=0.25,
)
parser.add_argument(
    '--seed',
    type=int,
    default=11,
)
parser.add_argument(
    '--device',
    type=str,
    default="cpu",
)
parser.add_argument(
    '--output_dir',
    type=str,
    default="./node_experiments",
)

args = parser.parse_args()
options = vars(args)
print(options)


if __name__ == '__main__':

    print("Experiment Started")
    start_time = time.time()
    hpo_phase = False
    task_dir = os.path.expanduser(
        os.path.join(
            args.output_dir,
            f'{args.seed}',
            f'{args.task_id}',
        )
    )
    data = get_node_dataset(
        seed=args.seed,
        task_id=args.task_id,
        test_size=args.test_size,
        validation_size=args.validation_size,
        refit=False,
    )
    if hpo_phase:
        # Start HPO Phase
        print("HPO Phase started")

        param_grid = ParameterGrid(
            {
                'num_layers': {2, 4, 8},
                'total_tree_count': {1024, 2048},
                'tree_depth': {6, 8},
                'tree_output_dim': {2, 3}
            }
        )
        results = []
        for config_counter, params in enumerate(param_grid):
            config_dir = os.path.join(task_dir, f'{config_counter}')
            print(params)
            run_information = evaluate_node(
                batch_size=args.batch_size,
                refit=False,
                data=data,
                config=params,
                device=args.device,
                experiment_name=config_dir,
                epochs=args.epochs,
            )
            print(params)
            print(run_information)
            results.append(
                {
                    'val_error': run_information['val_error'],
                    'best_epoch': run_information['best_epoch'],
                    'config': params,
                }
            )

        incumbent = sorted(results, key=lambda result: result['val_error'])[0]
        print(f"Best results, with validation error: {incumbent['val_error']}, "
              f"configuration: {incumbent['config']}")
        best_config = incumbent['config']
        best_epoch = incumbent['best_epoch']
    else:
        best_config = {
            'num_layers': 2,
            'total_tree_count': 1024,
            'tree_depth': 6,
            'tree_output_dim': 2,
        }
        run_information = evaluate_node(
            batch_size=args.batch_size,
            refit=False,
            data=data,
            config=best_config,
            device=args.device,
            experiment_name=os.path.join(task_dir, 'run'),
            epochs=args.epochs,
        )
        best_epoch = run_information['best_epoch']

    # Start Refit Phase
    print("Refit Started")
    refit_dir = os.path.join(task_dir, 'refit')
    print(f'Best epoch found for task: {args.task_id} in refit is: {best_epoch}')
    data = get_node_dataset(
        seed=args.seed,
        task_id=args.task_id,
        test_size=args.test_size,
        validation_size=0,
        refit=True,
    )

    run_information = evaluate_node(
        batch_size=args.batch_size,
        refit=True,
        data=data,
        config=best_config,
        device=args.device,
        experiment_name=refit_dir,
        epochs=best_epoch,
    )

    duration = time.time() - start_time
    os.makedirs(task_dir, exist_ok=True)

    result_dir = os.path.join(
        task_dir,
        'results.json',
    )

    result_dict = {
        'train balanced accuracy': 1 - run_information['train_error'],
        'test balanced accuracy': 1 - run_information['test_error'],
        'task_id': args.task_id,
        'duration': duration,
    }

    with open(result_dir, 'w') as file:
        json.dump(result_dict, file)