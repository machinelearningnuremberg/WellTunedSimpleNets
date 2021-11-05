import os

import openml
import pandas as pd


suite = openml.study.get_suite(218)
task_ids = suite.tasks

dataset_table = {
    'Task Id': [],
    'Dataset Name': [],
    'Number of examples': [],
    'Number of features': [],
    'Majority class percentage': [],
    'Minority class percentage': [],
}

for task_id in task_ids:
    task = openml.tasks.get_task(task_id, download_data=False)
    dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
    dataset_table['Task Id'].append(task_id)
    dataset_table['Dataset Name'].append(dataset.name)
    dataset_table['Number of examples'].append(dataset.qualities['NumberOfInstances'])
    dataset_table['Number of features'].append(dataset.qualities['NumberOfFeatures'])
    dataset_table['Majority class percentage'].append(f"{dataset.qualities['MajorityClassPercentage']:.3f}")
    dataset_table['Minority class percentage'].append(f"{dataset.qualities['MinorityClassPercentage']:.3f}")

output_path = os.path.expanduser(
    os.path.join(
        '~',
        'Desktop',
        'dataset_collection.csv'
    )
)

dataset_info_frame = pd.DataFrame.from_dict(dataset_table)
dataset_info_frame.to_csv(output_path, index=False)
