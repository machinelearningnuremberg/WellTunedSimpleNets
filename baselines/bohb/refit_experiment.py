import argparse
import json
import logging
logging.basicConfig(level=logging.DEBUG)
import os
import random

import hpbandster.core.result as hpres
import numpy as np
import openml

from data.loader import Loader
from worker import CatBoostWorker, XGBoostWorker, TabNetWorker


parser = argparse.ArgumentParser(
    description='Baseline refit experiment.'
)
parser.add_argument(
    '--run_id',
    type=str,
    help='The run id of the optimization run.',
    default='Baseline',
)
parser.add_argument(
    '--working_directory',
    type=str,
    help='The working directory where results will be stored.',
    default='.',
)
parser.add_argument(
    '--model',
    type=str,
    help='Which model to use for the experiment.',
    default='tabnet',
)
parser.add_argument(
    '--task_id',
    type=int,
    help='Minimum budget used during the optimization.',
    default=233109,
)
parser.add_argument(
    '--seed',
    type=int,
    help='Seed used for the experiment.',
    default=11,
)
parser.add_argument(
    '--nr_threads',
    type=int,
    help='Number of threads for one worker.',
    default=2,
)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

task_id = args.task_id
task = openml.tasks.get_task(task_id, download_data=False)
nr_classes = int(openml.datasets.get_dataset(task.dataset_id, download_data=False).qualities['NumberOfClasses'])

run_directory = os.path.join(
    args.working_directory,
    f'{args.task_id}',
    f'{args.seed}',
)
os.makedirs(run_directory, exist_ok=True)

worker_choices = {
    'tabnet': TabNetWorker,
    'xgboost': XGBoostWorker,
    'catboost': CatBoostWorker,
}

model_worker = worker_choices[args.model]

if args.model == 'tabnet':
    param = model_worker.get_parameters(
        task_id=args.task_id,
        seed=args.seed,
    )
elif args.model =='xgboost':
    param = model_worker.get_parameters(
        task_id=args.task_id,
        nr_classes=nr_classes,
        seed=args.seed,
        nr_threads=args.nr_threads,
        output_directory=run_directory,
    )
else:
    param = model_worker.get_parameters(
        task_id=args.task_id,
        nr_classes=nr_classes,
        seed=args.seed,
        output_directory=run_directory,
    )

print(f'Refit experiment started with task id: {args.task_id}')

worker = model_worker(
    args.run_id,
    param=param,
    nameserver='127.0.0.1',
)

result = hpres.logged_results_to_HBS_result(run_directory)
all_runs = result.get_all_runs()
id2conf = result.get_id2config_mapping()

inc_id = result.get_incumbent_id()
inc_runs = result.get_runs_by_id(inc_id)
inc_config = id2conf[inc_id]['config']
print(f"Best Configuration So far {inc_config}")

# default values to find the config with the
# best performance, so we can pull the best
# iteration number.
val_error_min = 100
best_round = 0
if 'early_stopping_rounds' in inc_config:
    for run in inc_runs:
        print(run)
        print(run.info)
        if run.loss < val_error_min:
            val_error_min = run.loss
            if 'best_round' in run.info:
                best_round = run.info['best_round']
    # no need for the early stopping rounds anymore
    del inc_config['early_stopping_rounds']
    # train only for the best performance achieved
    # for the 'best_round' iteration
    if args.model == 'tabnet':
        inc_config['max_epochs'] = best_round
    else:
        inc_config['num_round'] = best_round

    print(f'Best round for {args.model} refit: {best_round}')

refit_result = worker.refit(inc_config)
with open(os.path.join(run_directory, 'refit_result.json'), 'w') as file:
    json.dump(refit_result, file)
