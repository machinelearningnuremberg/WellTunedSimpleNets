# [NeurIPS 2021] Well-tuned Simple Nets Excel on Tabular Datasets

## Introduction

This repo contains the source code accompanying the paper:

**Well-tuned Simple Nets Excel on Tabular Datasets**

Authors: Arlind Kadra, Marius Lindauer, Frank Hutter, Josif Grabocka

Tabular datasets are the last "unconquered castle" for deep learning, with traditional ML methods like Gradient-Boosted Decision Trees still performing strongly even against recent specialized neural architectures. In this paper, we hypothesize that the key to boosting the performance of neural networks lies in rethinking the joint and simultaneous application of a large set of modern regularization techniques. As a result, we propose regularizing plain Multilayer Perceptron (MLP) networks by searching for the optimal combination/cocktail of 13 regularization techniques for each dataset using a joint optimization over the decision on which regularizers to apply and their subsidiary hyperparameters.

We empirically assess the impact of these **regularization cocktails** for MLPs on a large-scale empirical study comprising 40 tabular datasets and demonstrate that: (i) well-regularized plain MLPs significantly outperform recent state-of-the-art specialized neural network architectures, and (ii) they even outperform strong traditional ML methods, such as XGBoost.


*News: Our work is accepted in the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS 2021).*


## Setting up the virtual environment

Our work is built on top of AutoPyTorch. To look at our implementation of the regularization cocktail ingredients, you can do the following:


```
git clone https://github.com/automl/Auto-PyTorch.git
cd Auto-PyTorch/
git checkout regularization_cocktails
```
To install the version of AutoPyTorch that features our work, you can use these additional commands:

```
# The following commands assume the user is in the cloned directory
conda create -n reg_cocktails python=3.8
conda activate reg_cocktails
conda install gxx_linux-64 gcc_linux-64 swig
cat requirements.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

## Running the Regularization Cocktail code

The main files to run the regularization cocktails are in the `cocktails` folder and are
`main_experiment.py` and `refit_experiment.py`. The first module can be used to start a full
HPO search, while, the other module can be used to refit on certain datasets when the time does not suffice to perform the full HPO search and to complete the refit of the incumbent hyperparameter
configuration.

The main arguments for `main_experiment.py`:

- `--task_id`: The task id in OpenML. Basically the dataset that will be used in the experiment.
- `--wall_time`: The total runtime to be used. It is the total runtime for the HPO search and also final refit.
- `--func_eval_time`: The maximal time for one function evaluation parametrized by a certain hyperparameter configuration.
- `--epochs`: The number of epochs for one hyperparameter configuration to be evaluated on.
- `--seed`: The seed to be used for the run.
- `--tmp_dir`: The temporary directory for the results to be stored in.
- `--output_dir`: The output directory for the results to be stored in.
- `--nr_workers`: The number of workers which corresponds to the number of hyperparameter configurations run in parallel.
- `--nr_threads`: The number of threads. 
- `--cash_cocktail`: An important flag that activates the regularization cocktail formulation.

**A minimal example of running the regularization cocktails**:

```
python main_experiment.py --task_id 233088 --wall_time 600 --func_eval_time 60 --epochs 10 --seed 42 --cash_cocktail True
```

The example above will run the regularization cocktails for 10 minutes, with a function evaluation limit of 50 seconds for task 233088. Every
hyperparameter configuration will be evaluated for 10 epochs, the seed 42 will be used for the experiment and data splits.

**A minimal example of running only one regularization method:**
```
python main_experiment.py --task_id 233088 --wall_time 600 --func_eval_time 60 --epochs 10 --seed 42 --use_weight_decay
```
In case you would like to investigate individual regularization methods, you can look at the different arguments
that control them in the `main_experiment.py`. Additionally, if you want to remove 
the limit on the number of hyperparameter configurations, you can remove the following lines:

```
smac_scenario_args={
    'runcount_limit': number_of_configurations_limit,
}
```
## Plots

The plots that are included in our paper were generated from the functions in the module `results.py`.
Although mentioned in most function documentations, most of the functions that plot the baseline diagrams and
plots expect a folder structure as follows:

`common_result_folder/baseline/results.csv`

There are functions inside the module itself that generate the `results.csv` files.

## Baselines

The code for running the baselines can be found in the `baselines` folder.

- TabNet, XGBoost, CatBoost can be found in the `baselines/bohb` folder.
- The other baselines like AutoGluon, auto-sklearn and Node can be found in the corresponding folders named the same.

TabNet, XGBoost, CatBoost and AutoGluon have the same two main files as our regularization cocktails, `main_experiment.py` and `refit_experiment.py`.

## Figures

![alt text](https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/figures/all_baselines_diagram.png "Critical Difference diagram of all the methods")

## Citation
```
@inproceedings{kadra2021well,
  title={Well-tuned Simple Nets Excel on Tabular Datasets},
  author={Kadra, Arlind and Lindauer, Marius and Hutter, Frank and Grabocka, Josif},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
