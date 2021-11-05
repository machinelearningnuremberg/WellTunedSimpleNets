import argparse
import json
import logging
logging.basicConfig(level=logging.DEBUG)
import os
import pickle
import random
import time

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers import RandomSearch as RS
import numpy as np
import openml

from worker import CatBoostWorker, XGBoostWorker, TabNetWorker


parser = argparse.ArgumentParser(
    description='Baseline experiment.'
)
parser.add_argument(
    '--run_id',
    type=str,
    help='The run id of the optimization run.',
    default='tabular_baseline',
)
parser.add_argument(
    '--working_directory',
    type=str,
    help='The working directory where results will be stored.',
    default='.',
)
parser.add_argument(
    '--nic_name',
    type=str,
    help='Which network interface to use for communication.',
    default='ib0',
)
parser.add_argument(
    '--optimizer',
    type=str,
    help='Which optimizer to use for the experiment.',
    default='bohb',
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
    help='Task id used for the experiment.',
    default=233109,
)
parser.add_argument(
    '--seed',
    type=int,
    help='Seed used for the experiment.',
    default=11,
)
parser.add_argument(
    '--max_budget',
    type=float,
    help='Maximum budget used during the optimization.',
    default=1,
)
parser.add_argument(
    '--min_budget',
    type=float,
    help='Minimum budget used during the optimization.',
    default=1,
)
parser.add_argument(
    '--n_iterations',
    type=int,
    help='Number of BOHB iterations.',
    default=10,
)
parser.add_argument(
    '--n_workers',
    type=int,
    help='Number of workers to run in parallel.',
    default=2,
)
parser.add_argument(
    '--nr_threads',
    type=int,
    help='Number of threads for one worker.',
    default=2,
)
parser.add_argument(
    '--worker',
    help='Flag to turn this into a worker process',
    action='store_true',
)

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

host = hpns.nic_name_to_host(args.nic_name)

# determine the problem type, if it is binary
# or multiclass classification
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
# build the model setting configuration
if args.model == 'tabnet':
    param = model_worker.get_parameters(
        task_id=task_id,
        seed=args.seed,
    )
elif args.model =='xgboost':
    param = model_worker.get_parameters(
        task_id=task_id,
        nr_classes=nr_classes,
        seed=args.seed,
        nr_threads=args.nr_threads,
        output_directory=run_directory,
    )
else:
    param = model_worker.get_parameters(
        task_id=task_id,
        nr_classes=nr_classes,
        seed=args.seed,
        output_directory=run_directory,
    )

if args.worker:
    # short artificial delay to make sure the nameserver is already running
    time.sleep(5)
    worker = model_worker(
        run_id=args.run_id,
        host=host,
        param=param,
    )
    while True:
        try:
            worker.load_nameserver_credentials(
                working_directory=args.working_directory,
            )
            break
        except RuntimeError:
            pass
    worker.run(background=False)
    exit(0)

print(f'Experiment started with task id: {args.task_id}')


NS = hpns.NameServer(
    run_id=args.run_id,
    host=host,
    port=0,
    working_directory=args.working_directory,
)
ns_host, ns_port = NS.start()

worker = model_worker(
    run_id=args.run_id,
    host=host,
    param=param,
    nameserver=ns_host,
    nameserver_port=ns_port
)
worker.run(background=True)
result_logger = hpres.json_result_logger(directory=run_directory, overwrite=False)

optimizer_choices = {
    'bohb': BOHB,
    'random_search': RS,
}

optimizer = optimizer_choices[args.optimizer]

# for the moment only available to XGBoost
if args.model == 'xgboost':
    config_space = model_worker.get_default_configspace(
        seed=args.seed,
        early_stopping=True,
        conditional_imputation=False,
    )
else:
    config_space = model_worker.get_default_configspace(seed=args.seed)

bohb = optimizer(
    configspace=config_space,
    run_id=args.run_id,
    host=host,
    nameserver=ns_host,
    nameserver_port=ns_port,
    min_budget=args.min_budget,
    max_budget=args.max_budget,
    result_logger=result_logger,
)

res = bohb.run(
    n_iterations=args.n_iterations,
    min_n_workers=args.n_workers
)

bohb.shutdown(shutdown_workers=True)
NS.shutdown()

with open(os.path.join(run_directory, 'results.pkl'), 'wb') as fh:
    pickle.dump(res, fh)

id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()
incumbent_runs = res.get_runs_by_id(incumbent)
best_config = id2config[incumbent]['config']

# default values to find the config with the
# best performance, so we can pull the best
# iteration number.
val_error_min = 100
best_round = 0
if 'early_stopping_rounds' in best_config:
    for run in incumbent_runs:
        print(run)
        print(run.info)
        if run.loss < val_error_min:
            val_error_min = run.loss
            if 'best_round' in run.info:
                best_round = run.info['best_round']
    # no need for the early stopping rounds anymore
    del best_config['early_stopping_rounds']
    # train only for the best performance achieved
    # for the 'best_round' iteration
    if args.model == 'tabnet':
        best_config['max_epochs'] = best_round
    else:
        best_config['num_round'] = best_round

    print(f'Best round for {args.model} refit: {best_round}')


all_runs = res.get_all_runs()
print('Best found configuration:', best_config)
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'
      % (sum([r.budget for r in all_runs])/args.max_budget))
print('Total budget corresponds to %.1f full function evaluations.'
      % (sum([r.budget for r in all_runs])/args.max_budget))
print('The run took  %.1f seconds to complete.'
      % (all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

worker = model_worker(
    args.run_id,
    param=param,
    nameserver='127.0.0.1',
)
refit_result = worker.refit(best_config)
with open(os.path.join(run_directory, 'refit_result.json'), 'w') as file:
    json.dump(refit_result, file)
