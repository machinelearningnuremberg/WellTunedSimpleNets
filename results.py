import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openml
from scipy.stats import wilcoxon, rankdata
import seaborn as sns


sns.set(
    rc={
        'figure.figsize': (11.7, 8.27),
        'font.size': 35,
        'axes.titlesize': 35,
        'axes.labelsize': 35,
        'xtick.labelsize': 35,
        'ytick.labelsize': 35,
    },
    style="white"
)


def get_task_list(
    benchmark_task_file: str = 'path/to/tasks.txt',
) -> List[int]:
    """Get the task id list.

    Goes through the given file and collects all of the task
    ids.

    Args:
        benchmark_task_file (str):
            A string to the path of the benchmark task file. Including
            the task file name.

    Returns:
        benchmark_task_ids (List[int]):
            A list of all the task ids for the benchmark.
    """
    with open(os.path.join(benchmark_task_file), 'r') as f:
        benchmark_info_str = f.readline()
        benchmark_task_ids = [int(task_id) for task_id in benchmark_info_str.split(' ')]

    return benchmark_task_ids


# TODO merge all the build_table functions
def build_table_from_autopytorch_data(
    output_dir: str,
    benchmark_task_file: str,
    seed: int = 11,
):
    """
    Stores the final performance for the autopytorch algorithm on every dataset
    to a csv file in the output_dir.

    Args:
        output_dir (str): The output directory where the results are stored.
        benchmark_task_file (str): The path where the benchmark txt file is located.
        seed (int):  The seed used for the experiment.
    """
    experiment_table = {
        'Task Id': [],
        'Test Performance': [],
    }
    benchmark_task_ids = get_task_list(benchmark_task_file)
    for task_id in benchmark_task_ids:
        task_dir = os.path.join(
            output_dir,
            f'{seed}',
            f'{task_id}',
            'final_result.json'
        )

        try:
            with open(task_dir, 'r') as fp:
                task_performance_info = json.load(fp)
                task_test_performance = task_performance_info['test balanced accuracy']
                experiment_table['Task Id'].append(task_id)
                experiment_table['Test Performance'].append(task_test_performance)
        except FileNotFoundError:
            print(f'Refit for task id:{task_id} not found')


    experiment_df = pd.DataFrame.from_dict(experiment_table, orient='columns')
    df_dir = os.path.join(
        output_dir,
        'results.csv',
    )
    experiment_df.to_csv(df_dir, index=False)


def build_table_from_node_data(
    output_dir: str,
    benchmark_task_file: str,
    seed: int = 11,
):
    """
    Stores the final performance for the node algorithm on every dataset
    to a csv file in the output_dir.

    Args:
        output_dir (str): The output directory where the results are stored.
        benchmark_task_file (str): The path where the benchmark txt file is located.
        seed (int):  The seed used for the experiment.
    """
    experiment_table = {
        'Task Id': [],
        'Test Performance': [],
    }
    benchmark_task_ids = get_task_list(benchmark_task_file)
    for task_id in benchmark_task_ids:
        task_dir = os.path.join(
            output_dir,
            f'{seed}',
            f'{task_id}',
            'results.json'
        )

        try:
            with open(task_dir, 'r') as fp:
                task_performance_info = json.load(fp)
                task_test_performance = task_performance_info['test balanced accuracy']
                experiment_table['Task Id'].append(task_id)
                experiment_table['Test Performance'].append(task_test_performance)
        except FileNotFoundError:
            print(f'Refit for task id:{task_id} not found')


    experiment_df = pd.DataFrame.from_dict(experiment_table, orient='columns')
    df_dir = os.path.join(
        output_dir,
        'results.csv',
    )
    experiment_df.to_csv(df_dir, index=False)


def build_table_from_tabnet_data(
    output_dir: str,
    benchmark_task_file: str,
    seed: int = 11,
):
    """
    Stores the final performance for the TabNet algorithm on every dataset
    to a csv file in the output_dir.

    Args:
        output_dir (str): The output directory where the results are stored.
        benchmark_task_file (str): The path where the benchmark txt file is located.
        seed (int):  The seed used for the experiment.
    """
    experiment_table = {
        'Task Id': [],
        'Test Performance': [],
    }
    benchmark_task_ids = get_task_list(benchmark_task_file)
    for task_id in benchmark_task_ids:
        task_dir = os.path.join(
            output_dir,
            f'{task_id}',
            f'{seed}',
            'refit_results.json'
        )

        try:
            with open(task_dir, 'r') as fp:
                task_performance_info = json.load(fp)
                task_test_performance = task_performance_info['test_accuracy']
                experiment_table['Task Id'].append(task_id)
                experiment_table['Test Performance'].append(task_test_performance)
        except FileNotFoundError:
            print(f'Refit for task id:{task_id} not found')
            experiment_table['Task Id'].append(task_id)
            experiment_table['Test Performance'].append(-1)


    experiment_df = pd.DataFrame.from_dict(experiment_table, orient='columns')
    df_dir = os.path.join(
        output_dir,
        'results.csv',
    )
    experiment_df.to_csv(df_dir, index=False)


def build_table_from_autogluon_data(
    output_dir: str,
    benchmark_task_file: str,
    seed: int = 11,
):
    """
    Stores the final performance for the AutoGluon algorithm on every dataset
    to a csv file in the output_dir.

    Args:
        output_dir (str): The output directory where the results are stored.
        benchmark_task_file (str): The path where the benchmark txt file is located.
        seed (int):  The seed used for the experiment.
    """
    experiment_table = {
        'Task Id': [],
        'Test Performance': [],
    }
    benchmark_task_ids = get_task_list(benchmark_task_file)
    for task_id in benchmark_task_ids:
        task_dir = os.path.join(
            output_dir,
            f'{seed}',
            f'{task_id}',
            'results.csv',
        )

        try:
            performance_df = pd.read_csv(task_dir)
            score = performance_df['score'].to_numpy()
            score = score[0]
        except FileNotFoundError:
            print(f'No results for task id:{task_id}')
            score = -1
        experiment_table['Task Id'].append(task_id)
        experiment_table['Test Performance'].append(score)

    experiment_df = pd.DataFrame.from_dict(experiment_table, orient='columns')
    df_dir = os.path.join(
        output_dir,
        'results.csv',
    )
    experiment_df.to_csv(df_dir, index=False)


def build_table_from_cocktails_data(
    output_dir: str,
    benchmark_task_file: str,
    seed: int = 11,
):
    """
    Stores the final performance for the old autopytorch algorithm on every
    dataset to a csv file in the output_dir.

    Args:
        output_dir (str): The output directory where the results are stored.
        benchmark_task_file (str): The path where the benchmark txt file is located.
        seed (int):  The seed used for the experiment.
    """
    experiment_table = {
        'Task Id': [],
        'Duration': [],
    }
    benchmark_task_ids = get_task_list(benchmark_task_file)
    for task_id in benchmark_task_ids:
        task_dir = os.path.join(
            output_dir,
            '512',
            f'{task_id}',
            'refit_run',
            f'{seed}',
            'run_results.txt',
        )
        if not os.path.exists(task_dir):
            task_dir = os.path.join(
                output_dir,
                '512',
                f'{task_id}',
                'run_results.txt',
            )

        try:
            with open(task_dir, 'r') as fp:
                task_performance_info = json.load(fp)
                task_performance = float(task_performance_info['mean_test_bal_acc'])
                experiment_table['Task Id'].append(task_id)
                experiment_table['Test Performance'].append(task_performance)
        except FileNotFoundError:
            print(f'Refit for task id:{task_id} not found')
            experiment_table['Task Id'].append(task_id)
            experiment_table['Test Performance'].append(-1)


    experiment_df = pd.DataFrame.from_dict(experiment_table, orient='columns')
    df_dir = os.path.join(
        output_dir,
        'results.csv',
    )
    experiment_df.to_csv(df_dir, index=False)


def generate_times_from_autopytorch_data(
    output_dir: str,
    benchmark_task_file: str,
    seed: int = 11,
):
    """
    Stores the duration for the autopytorch algorithm on every dataset
    to a csv file in the output_dir.

    Args:
        output_dir (str): The output directory where the results are stored.
        benchmark_task_file (str): The path where the benchmark txt file is located.
        seed (int):  The seed used for the experiment.
    """
    experiment_table = {
        'Task Id': [],
        'Duration': [],
    }
    benchmark_task_ids = get_task_list(benchmark_task_file)
    for task_id in benchmark_task_ids:
        task_dir = os.path.join(
            output_dir,
            f'{seed}',
            f'{task_id}',
            'final_result.json'
        )

        try:
            with open(task_dir, 'r') as fp:
                task_performance_info = json.load(fp)
                task_duration = float(task_performance_info['duration'])
                experiment_table['Task Id'].append(task_id)
                experiment_table['Duration'].append(task_duration)
        except FileNotFoundError:
            print(f'Refit for task id:{task_id} not found')
            experiment_table['Task Id'].append(task_id)
            experiment_table['Duration'].append(-1)


    experiment_df = pd.DataFrame.from_dict(experiment_table, orient='columns')
    df_dir = os.path.join(
        output_dir,
        'durations.csv',
    )
    experiment_df.to_csv(df_dir, index=False)


def build_all_table(
    result_dir: str,
):
    """Generates a table with all the baselines and their final performances on every
    dataset.

    Args:
        result_dir (str): The results folder where the data for every baseline is organized.

    Returns:
        output (pd.DataFrame): The DataFrame with all the baseline final results.

    Note:
        The folder structure should be result_dir/baseline_name/results.csv, where results.csv
            corresponds to a table with the performance of the baseline of every task.
    """
    method_folders = [
        'plain_network',
        'dropout',
        'selu',
        'XGBoost/ES',
        'neurips_xgboost_es',
        'neurips_xgboost_no_es',
        'catboost_v2',
        'XGBoost/No ES',
        'autosklearn',
        'tabnet/ES',
        'autogluon_only_hpo',
        'tabnet/No ES',
        'node',
        'autogluon/nn_only_4_days',
        'autogluon/full_4_days',
        'cocktail',
        'new_cocktail',
        'search_cocktail',
    ]

    pretty_names = {
        'autogluon/nn_only_4_days': ' AutoGL. + Stacking',
        'autogluon/full_4_days': 'Full AutoGL',
        'autogluon_only_hpo': 'AutoGL. + HPO',
        'cocktail': ' MLP + C ',
        'new_cocktail': 'SMAC MLP + C ',
        'search_cocktail': 'Search Smac + C',
        'plain_network': ' MLP ',
        'dropout': ' MLP + D ',
        'node': ' NODE  ',
        'tabnet/ES': ' TabN. + ES ',
        'XGBoost/ES': ' XGB. + ES ',
        'tabnet/No ES': ' TabN. ',
        'XGBoost/No ES': ' XGB. ',
        'autosklearn': ' ASK-G. ',
        'selu': 'MLP + S',
        'catboost_v2': 'CatBoost',
        'neurips_xgboost_es': 'XGB. + ES + ENC',
        'neurips_xgboost_no_es': 'XGB. + ENC',
    }

    pandas_frames = []
    drop_task_ids = False
    for method in method_folders:
        method_results = os.path.join(result_dir, method)
        method_df = pd.read_csv(os.path.join(method_results, 'results.csv'))
        method_df.columns = ['Task Id', pretty_names[method]]
        if drop_task_ids:
            method_df = method_df.drop(labels=['Task Id'], axis=1)
        else:
            drop_task_ids = True
        pandas_frames.append(method_df)
    output = pd.concat(pandas_frames, join='outer', axis=1)
    task_infos = []
    for task_id in output['Task Id']:
        task = openml.tasks.get_task(task_id, download_data=False)
        dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
        task_info = f'{dataset.qualities["NumberOfInstances"]}/{dataset.qualities["NumberOfFeatures"]}'
        task_infos.append(task_info)

    # uncomment if you want to add dataset information in the form of instances
    # and features
    # output.insert(loc=1, column='Size (Ins./Feat.)', value=task_infos)
    # output['Task Id'] = output['Task Id'].apply(lambda x: openml.datasets.get_dataset(openml.tasks.get_task(x, download_data=False).dataset_id, download_data=False).name)
    output = output.fillna(-1)

    # skipping the task information columns to format in a pretty way
    # the baseline result columns.
    method_columns = output.columns[1:]

    # only keep precision up to 3 numbers after the dot/comma
    for column in method_columns:
        output[column] = output[column].apply(lambda x: f'{x * 100:.3f}' if x != -1 else np.nan)

    return output


def compare_models(
    result_dir: str,
    baseline: str = 'autogluon/full_4_days',
    cocktails: str = 'autopytorch',
):
    """Compare a baseline with the regularization cocktail.

    The method will print the necessary information based on the results.
    The results should be stored on a certain format:
    result_dir/baseline_name/results.csv

    Args:
        result_dir (str): The directory where the results are stored.
        baseline (str): The baseline name.
        cocktails (str):  The main method name.
    """
    baseline_results = os.path.join(
        result_dir,
        baseline,
        'results.csv',
    )
    cocktail_results = os.path.join(
        result_dir,
        cocktails,
        'results.csv',
    )

    cocktails_df = pd.read_csv(cocktail_results)
    baseline_df = pd.read_csv(baseline_results)

    task_ids = list(cocktails_df['Task Id'])
    task_ids = [int(task_id) for task_id in task_ids]

    cocktail_performances = []
    baseline_performances = []
    cocktail_wins = 0
    cocktail_looses = 0
    cocktail_draws = 0

    for task_id in task_ids:
        cocktail_task_performance = cocktails_df.query(f'`Task Id`=={task_id}')['Test Performance']
        cocktail_task_performance = cocktail_task_performance.to_numpy()[0]
        baseline_task_performance = baseline_df.query(f'`Task Id`=={task_id}')['Test Performance']
        baseline_task_performance = baseline_task_performance.to_numpy()[0]

        # if a task has not finished for the baseline, do not use it
        # in the comparison against the regularization cocktail.
        if baseline_task_performance != -1.0:
            cocktail_performances.append(cocktail_task_performance)
            baseline_performances.append(baseline_task_performance)
            if cocktail_task_performance > baseline_task_performance:
                cocktail_wins += 1
            elif cocktail_task_performance == baseline_task_performance:
                cocktail_draws += 1
            else:
                cocktail_looses += 1
    _, p_value = wilcoxon(cocktail_performances, baseline_performances)

    print(f'Cocktail against {baseline}, '
          f'wins {cocktail_wins}, '
          f'looses {cocktail_looses}, '
          f'draws {cocktail_draws}')
    print(f'Wilxocon p-value {p_value}')


def build_cd_diagram(
    results_dir: str,
) -> pd.DataFrame:
    """Prepare the results for a critical difference diagram.
    This function prepares all the results into a pandas dataframe
    so that it can be used to create a critical difference diagram
    of all the methods.

    Args:
        results_dir (str): The directory where the results are stored.

    Returns:
        result_df (pd.DataFrame):
            The DataFrame that contains the final results for all the baselines
            in a format that can be used as an input for hte cd-diagram plot.

    Note:
        The folder structure should be result_dir/baseline_name/results.csv, where results.csv
            corresponds to a table with the performance of the baseline of every task.
    """
    method_folders = [
        'plain_network',
        'dropout',
        'XGBoost/ES',
        'XGBoost/No ES',
        'autosklearn',
        'autogluon_only_hpo',
        'tabnet/ES',
        'tabnet/No ES',
        'node',
        'selu',
        'autogluon/nn_only_4_days',
        'autogluon/full_4_days',
        'cocktail',
        'new_cocktail',
        'search_cocktail',
        'catboost_v2',
        'neurips_xgboost_es',
        'neurips_xgboost_no_es',
    ]

    pretty_names = {
        'autogluon/nn_only_4_days': ' AutoGL. S',
        'autogluon/full_4_days': 'Full AutoGL',
        'autogluon_only_hpo': 'AutoGL. HPO',
        'cocktail': ' MLP + C ',
        'new_cocktail': 'SMAC MLP + C ',
        'search_cocktail': 'Search Smac + C',
        'plain_network': ' MLP ',
        'dropout': ' MLP + D ',
        'node': ' NODE  ',
        'tabnet/ES': ' TabN. + ES ',
        'XGBoost/ES': ' XGB. + ES ',
        'tabnet/No ES': ' TabN. ',
        'XGBoost/No ES': ' XGB. ',
        'autosklearn': ' ASK-G. ',
        'selu': 'MLP + SELU',
        'catboost_v2': 'CatBoost',
        'neurips_xgboost_es': 'XGB. + ES + ENC',
        'neurips_xgboost_no_es': 'XGB. + ENC',
    }

    table_results = {
        'Network': [],
        'Task Id': [],
        'Balanced Accuracy': [],
    }

    search_results = os.path.join(results_dir, 'cocktail')
    search_df = pd.read_csv(os.path.join(search_results, 'results.csv'))
    task_ids = list(search_df['Task Id'])
    task_ids = [int(task_id) for task_id in task_ids]

    for method in method_folders:
        method_results = os.path.join(results_dir, method)
        method_df = pd.read_csv(os.path.join(method_results, 'results.csv'))
        method_df.columns = ['Task Id', pretty_names[method]]
        for index, row in method_df.iterrows():
            if int(row['Task Id']) in task_ids:
                table_results['Network'].append(pretty_names[method])
                table_results['Task Id'].append(row['Task Id'])
                accuracy = row[pretty_names[method]]
                table_results['Balanced Accuracy'].append(accuracy if accuracy != -1 else np.nan)

    result_df = pd.DataFrame(data=table_results)

    return result_df


def generate_ranks_data(
    all_data: pd.DataFrame,
) -> pd.DataFrame:
    """Generate the ranks of the baselines for every dataset.

    Args:
        all_data (pd.DataFrame):
            A dataframe where each row consists of tasks values
            across different models.

    Returns:
        ranks_df (pd.DataFrame):
            A dataframe of the ranks of all methods over
            the different tasks.
    """
    all_ranked_data = []
    all_data.drop(columns=['Task Id'], inplace=True)
    column_names = all_data.columns
    for row in all_data.itertuples(index=False):
        task_regularization_data = list(row)
        task_regularization_data = [float(x) for x in task_regularization_data]

        task_ranked_data = rankdata(
            task_regularization_data,
            method='average',
        )
        reversed_data = len(task_ranked_data) + 1 - task_ranked_data
        all_ranked_data.append(reversed_data)
    ranks_df = pd.DataFrame(all_ranked_data, columns=column_names)

    return ranks_df


def patch_violinplot():
    """Patch seaborn's violinplot in current axis
    to workaround matplotlib's bug ##5423."""
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    for art in ax.get_children():
        if isinstance(art, PolyCollection):
            art.set_edgecolor((0.3, 0.3, 0.3))


def generate_ranks_comparison(
    all_data: pd.DataFrame,
):
    """Generate a ranks comparison between all methods.
    Creates a violin plot that showcases the ranks that
    the different methods achieve over all the tasks/datasets
    and saves it in the current executing folder.

    Args:
        all_data (pd.DataFrame):
            A dataframe where each row consists of method
            ranks over a certain task.
    """
    all_data_ranked = generate_ranks_data(all_data)
    all_data = pd.melt(
        all_data_ranked,
        value_vars=all_data.columns,
        var_name='Method',
        value_name='Rank',
    )

    fig, _ = plt.subplots()
    sns.violinplot(x='Method', y='Rank', linewidth=3, data=all_data, cut=0, kind='violin')
    patch_violinplot()
    plt.title('Ranks of the baselines and the MLP + C')
    plt.xlabel("")
    # plt.xticks(rotation=60)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        top=False,
        bottom=True,
        # ticks along the top edge are off
    )
    fig.autofmt_xdate()
    plt.savefig(
        'violin_ranks.pdf',
        bbox_inches='tight',
        pad_inches=0.15,
        margins=0.1,
    )


def plot_models_error_rate(
    result_dir,
    baseline,
    cocktails,
):
    """Plot a comparison of the models and generate descriptive
    statistics based on the results of all the models.
    Generates plots which showcase the gain of the cocktail versus
    the baseline. (Plots the error rate of the baseline divided
    by the error rate of the cocktail.) Furthermore, it
    generates information regarding the wins, looses and draws
    of both methods, including a significance result. Saves the
    plot to the current folder.

    Args:
        baseline_dir (str):
            The directory where the results are located for the baseline
            methods.
        cocktail_dir (str):
            The directory where the results are located for the regularization
            cocktails.
    """
    pretty_names = {
        'cocktail': 'MLP + C',
        'autogluon/nn_only_4_days': 'AutoGL. S',
        'XGBoost/No ES': 'XGB.',
        'autosklearn': 'ASK-G.',
    }
    cocktail_error_rates = []
    baseline_error_rates = []

    baseline_results = os.path.join(
        result_dir,
        baseline,
        'results.csv',
    )
    cocktail_results = os.path.join(
        result_dir,
        cocktails,
        'results.csv',
    )
    cocktails_df = pd.read_csv(cocktail_results)
    baseline_df = pd.read_csv(baseline_results)

    task_ids = list(cocktails_df['Task Id'])
    for task_id in task_ids:
        cocktail_task_performance = cocktails_df.query(f'`Task Id`=={task_id}')['Test Performance']
        cocktail_task_performance = cocktail_task_performance.to_numpy()[0]
        baseline_task_performance = baseline_df.query(f'`Task Id`=={task_id}')['Test Performance']
        baseline_task_performance = baseline_task_performance.to_numpy()[0]

        cocktail_task_result_error = 1 - cocktail_task_performance
        benchmark_task_result_error = 1 - baseline_task_performance
        cocktail_error_rates.append(cocktail_task_result_error)
        baseline_error_rates.append(benchmark_task_result_error)

    fig, ax = plt.subplots()
    plt.scatter(baseline_error_rates, cocktail_error_rates, s=100, c='#273E47', label='Test Error Rate')
    lims = [
        np.min([0, 0]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].set_visible(False)
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, color='r')
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel(f"{pretty_names[baseline]} Error Rate")
    plt.ylabel(f"{pretty_names[cocktails]} Error Rate")

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        top=False,
        bottom=True,
        # ticks along the top edge are off
    )
    plt.tick_params(
        axis='y',
        which='both',
        left=True,
        right=False,
    )

    # plt.title("Comparison with XGBoost")
    plt.savefig(
        f'cocktail_vs_{pretty_names[baseline]}.pdf',
        bbox_inches='tight',
        pad_inches=0.15,
        margins=0.1,
    )


def incumbent_time_dataset(
    result_path: str,
    dataset_id: int,
    seed: int,
    max_number_configs: int = 840,
    method: str = 'cocktail',
) -> Tuple[float, int]:
    """Return the time needed to find the incumbent configuration
    given a maximal number of configurations for a certain dataset
    and a certain algorithm.

    Args:
        result_path (str): The path of the folder where the results are
            stored.
        dataset_id (int): The task id-
        seed (int): The seed used for the experiment.
        max_number_configs (int): The maximal number of configurations.
        method (str): The method name.

    Returns:
        incumbent_time, incumbent_index (Tuple[float, int]):
            A tuple with the time needed to find the incumbent configuration
            and the index of the incumbent configuration.
    """
    if method == 'cocktail':
        task_result_folder = os.path.expanduser(
            os.path.join(
                result_path,
                f'{dataset_id}',
                'hpo_run',
                f'{seed}',
            )
        )
    else:
        task_result_folder = os.path.expanduser(
            os.path.join(
                result_path,
                f'{dataset_id}',
                f'{seed}',
            )
        )

    index = 0
    incumbent_accuracy = 0
    start_time = None
    incumbent_time = None
    incumbent_index = None
    x_times = []
    y_accuracies = []

    with open(os.path.join(task_result_folder, 'results.json')) as result_file:
        for line in result_file:
            config_info = json.loads(line)
            # config_id
            _ = config_info[0]
            job_stats = config_info[2]
            started = job_stats['started']
            finished = job_stats['finished']

            if index == 0:
                start_time = started
            try:
                result_info = config_info[3]['info']
            except Exception:
                print(f'Worked Died problem')

            if method == 'cocktail':
                validation_curve = result_info[0]['val_balanced_accuracy']
                validation_accuracy = validation_curve[-1]
            else:
                validation_accuracy = result_info['val_accuracy']


            if validation_accuracy > incumbent_accuracy:
                incumbent_accuracy = validation_accuracy
                incumbent_time = finished - start_time
                incumbent_index = index

            index += 1

            estimated_time = finished - start_time
            x_times.append(estimated_time)
            y_accuracies.append(incumbent_accuracy)

            if index == max_number_configs:
                print("Max number of configs reached")
                break

    return incumbent_time, incumbent_index


def runtime_dataset(
    result_path: str,
    dataset_id: int,
    seed: int,
    max_number_configs: int = 840,
    method: str = 'cocktail',
) -> float:
    """Return the time needed to perform the HPO search
    given a maximal number of configurations for a certain
    dataset and a certain algorithm.

    Args:
        result_path (str): The path of the folder where the results are
            stored.
        dataset_id (int): The task id-
        seed (int): The seed used for the experiment.
        max_number_configs (int): The maximal number of configurations.
        method (str): The method name.

    Returns:
        estimated_time (float):
            The time elapsed for the HPO search.
    """
    if method == 'cocktail':
        task_result_folder = os.path.expanduser(
            os.path.join(
                result_path,
                f'{dataset_id}',
                'hpo_run',
                f'{seed}',
            )
        )
    else:
        task_result_folder = os.path.expanduser(
            os.path.join(
                result_path,
                f'{dataset_id}',
                f'{seed}',
            )
        )

    index = 0
    start_time = None

    with open(os.path.join(task_result_folder, 'results.json')) as result_file:
        for line in result_file:
            config_info = json.loads(line)
            job_stats = config_info[2]
            started = job_stats['started']
            finished = job_stats['finished']

            if index == 0:
                start_time = started

            estimated_time = finished - start_time
            index += 1

            if index == max_number_configs:
                print("Max number of configs reached")
                break

    return estimated_time


def generate_cocktail_vs_xgboost_incumbent_times(
    cocktail_folder: str,
    baseline_folder: str,
    baseline_name: str,
    benchmark_task_file: str,
):
    """Generate the cocktail vs XGBoost incumbent times
    information.

    Generates information regarding the cocktail vs xgboost time
    performance and saves a plot with the time distributions of what
    every method took to find the incumbent configuration.

    Args:
        cocktail_folder (str): The path where the cocktail folder is located.
        baseline_folder (str): The path where the baseline results are located.
        baseline_name (str): The baseline name.
        benchmark_task_file (str): The benchmark task file path.
    """
    task_ids = get_task_list(benchmark_task_file)
    cocktail_incumbent_task_times = []
    xgboost_incumbent_task_times = []
    info_dict = {
        'Cocktail': [],
        'XGBoost': [],
    }
    for task_id in task_ids:
        print(task_id)
        cocktail_task_time, cocktail_task_index = incumbent_time_dataset(
            cocktail_folder,
            task_id,
            11,
        )
        xgboost_task_time, xgboost_task_index = incumbent_time_dataset(
            baseline_folder,
            task_id,
            11,
            method=baseline_name,
        )
        cocktail_incumbent_task_times.append(cocktail_task_time)
        xgboost_incumbent_task_times.append(xgboost_task_time)
        info_dict['Cocktail'].append(cocktail_task_time)
        info_dict['XGBoost'].append(xgboost_task_time)

    print(f'Cocktail mean: {np.mean(cocktail_incumbent_task_times)}')
    print(f'Cocktail min: {np.min(cocktail_incumbent_task_times)}')
    print(f'XGBoost mean: {np.mean(xgboost_incumbent_task_times)}')
    print(f'Cocktail std: {np.std(cocktail_incumbent_task_times)}')
    print(f'XGBoost std: {np.std(xgboost_incumbent_task_times)}')
    info_frame = pd.DataFrame.from_dict(info_dict)

    sns.boxplot(data=info_frame)
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('comparison_incumbents_times.pdf')


def incumbent_performance_time_dataset(
    result_path: str,
    dataset_id: int,
    seed: int,
    max_number_configs: int = 840,
    method: str = 'cocktail',
    time: int = 3600,
) -> float:
    """Return the test accuracy of the incumbent configuration
    given a maximal number of configurations for a certain dataset
    and a certain algorithm for a given time marker.

    Args:
        result_path (str): The path of the folder where the results are
            stored.
        dataset_id (int): The task id-
        seed (int): The seed used for the experiment.
        max_number_configs (int): The maximal number of configurations.
        method (str): The method name.
        time (int): The time marker.

    Returns:
        incumbent_test_accuracy (float):
            The incumbent test accuracy.
    """
    if method == 'cocktail':
        task_result_folder = os.path.expanduser(
            os.path.join(
                result_path,
                f'{dataset_id}',
                'hpo_run',
                f'{seed}',
            )
        )
    else:
        task_result_folder = os.path.expanduser(
            os.path.join(
                result_path,
                f'{dataset_id}',
                f'{seed}',
            )
        )

    index = 0
    incumbent_val_accuracy = 0
    incumbent_test_accuracy = 0
    start_time = None

    with open(os.path.join(task_result_folder, 'results.json')) as result_file:
        for line in result_file:
            config_info = json.loads(line)
            job_stats = config_info[2]
            started = job_stats['started']
            finished = job_stats['finished']

            # start the time
            if index == 0:
                start_time = started

            try:
                result_info = config_info[3]['info']
            except Exception:
                pass
                # print(f'Worked Died problem')

            if method == 'cocktail':
                validation_curve = result_info[0]['val_balanced_accuracy']
                validation_accuracy = validation_curve[-1]
                test_curve = result_info[0]['test_result']
                test_accuracy = test_curve[-1]
            else:
                validation_accuracy = result_info['val_accuracy']
                test_accuracy = result_info['test_accuracy']

            estimated_time = finished - start_time
            if estimated_time >= time:
                return incumbent_test_accuracy

            if validation_accuracy > incumbent_val_accuracy:
                incumbent_val_accuracy = validation_accuracy
                incumbent_test_accuracy = test_accuracy

            index += 1

            if index == max_number_configs:
                # print("Max number of configs reached")
                break

    return incumbent_test_accuracy

def generate_performance_comparison_over_time(
    cocktail_folder: str,
    baseline_folder: str,
    baseline_name: str,
    benchmark_task_file: str,
):
    """Generate the cocktail vs XGBoost incumbent
    performance over time information.

    Generates information regarding the cocktail vs xgboost time
    performance and saves a plot with the average ranks of the
    methods over time.

    Args:
        cocktail_folder (str): The path where the cocktail folder is located.
        baseline_folder (str): The path where the baseline results are located.
        baseline_name (str): The baseline name.
        benchmark_task_file (str): The benchmark task file path.
    """
    task_ids = get_task_list(benchmark_task_file)
    times = [900, 1800, 3600, 7200, 14400, 28800, 57600, 115200, 230400, 345600]

    cocktail_ranks_over_time = []
    cocktail_stds_over_time = []
    baseline_ranks_over_time = []
    baseline_stds_over_time = []

    for time in times:
        baseline_ranks = []
        cocktail_ranks = []

        cocktail_wins = 0
        cocktail_ties = 0
        cocktail_loses = 0
        cocktail_performances = []
        baseline_performances = []

        for task_id in task_ids:

            cocktail_incumbent_performance = incumbent_performance_time_dataset(
                cocktail_folder,
                task_id,
                11,
                time=time,
            )
            baseline_incumbent_performance = incumbent_performance_time_dataset(
                baseline_folder,
                task_id,
                11,
                method=baseline_name,
                time=time,
            )
            cocktail_performances.append(cocktail_incumbent_performance)
            baseline_performances.append(baseline_incumbent_performance)

            if cocktail_incumbent_performance == 0 and baseline_incumbent_performance == 0:
                continue
            elif cocktail_incumbent_performance == 0:
                cocktail_loses += 1
                cocktail_ranks.append(2)
                baseline_ranks.append(1)
                continue
            elif baseline_incumbent_performance == 0:
                cocktail_wins += 1
                cocktail_ranks.append(1)
                baseline_ranks.append(2)
                continue

            if cocktail_incumbent_performance > baseline_incumbent_performance:
                cocktail_wins += 1
                cocktail_ranks.append(1)
                baseline_ranks.append(2)
            elif cocktail_incumbent_performance == baseline_incumbent_performance:
                cocktail_ties += 1
                cocktail_ranks.append(1.5)
                baseline_ranks.append(1.5)
            else:
                cocktail_loses += 1
                cocktail_ranks.append(2)
                baseline_ranks.append(1)

        _, p_value = wilcoxon(cocktail_performances, baseline_performances)
        cocktail_ranks_over_time.append(np.mean(cocktail_ranks))
        cocktail_stds_over_time.append(np.std(cocktail_ranks))
        baseline_ranks_over_time.append(np.mean(baseline_ranks))
        baseline_stds_over_time.append(np.std(baseline_ranks))
        print(f'For a runtime of {time / 3600} hours, The cocktails won: {cocktail_wins} times, tied: {cocktail_ties} times, lost: {cocktail_loses} times\np_value: {p_value}')

    plt.plot([time / 3600 for time in times], cocktail_ranks_over_time, label='MLP + C average rank')
    plt.plot([time / 3600 for time in times], baseline_ranks_over_time, label=f'XGBoost average rank')
    plt.legend()
    plt.xlabel('Time (Hours)')
    plt.ylabel('Average Rank')
    plt.tight_layout()
    plt.savefig('average_time_ranks.pdf')
