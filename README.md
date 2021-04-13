# Consequence-aware Sequential Counterfactuals (CSCF)
Code to reproduce our paper "Consequence-aware Sequential Counterfactual Generation"

**[Click here to access the paper.](https://arxiv.org/abs/2104.05592)**

## Running instructions
Use the provided `Makefile` to reproduce our experiments.

Use `make requirements` to install all dependencies with `pip`. We suggest creating a `virtualenv` before.
Ideally you should use the Python version `3.8.7`, which is what we used.

The original experiment files are already included in the `results` folder.
Running `make analysis` will create the plots used in the paper based on these results.

> **If you want to recreate the experiments from scratch, running `make all` will reproduce everything at once.**

If you only want to recreate the experiment files (for all methods), without installing requirements and creating plots, run `make experiments`.

For individual experiments/data use one of the following:
- `make experiment_setup` creates the initial experiment instances for the `adult` and `german` datasets
  - `make experiment_setup_adult` creates the initial experiment instances for the `adult` dataset
  - `make experiment_setup_german` creates the initial experiment instances for the `german` dataset
- `make cscf` (or `make cscf_adult`) for producing the `adult` dataset results for the `cscf` method
- `make scf` for producing the `adult` *and* `german` dataset results for the `scf` method
  - `make scf_adult` for producing the `adult` dataset results for the `scf` method
  - `make scf_german` for producing the `german` dataset results for the `scf` method
- `make competitor` for producing the `adult` and `german` dataset results for the `synth` (competitor) method
  - `make competitor_adult` for producing the `adult` dataset results for the `synth` (competitor) method
  - `make competitor_german` for producing the `german` dataset results for the `synth` (competitor) method
- `make analysis` to create the plots from the result files. (*All* result files need to be in the `results` folder; see the already existing files in there for the structure)


## Folder structure

```
├── competitor                              # Contains the original implementation code of the competitor with slight modifications        
│   └── synth-action-seq                    # https://github.com/goutham7r/synth-action-seq
├── experiment_data
│   └── test_instances                      # Contains the initial experiments instances
├── LICENSE
├── Makefile                                # Makefile for reproducing our experiments
├── plots                                   # Contains the the plots used in our paper
├── README.md
├── requirements.txt                        # pip requirements for running our code
├── results                                 # Result files for each experiment 
│   ├── competitor                          # Results of the competitor for adult and german datasets
│   └── my_method                           # Results of our method(s) (CSCF & SCF) for adult and german datasets
└── src                                     # Contains the implementation code of our method(s) (CSCF & SCF)
    ├── analysis                            # Used to create the plots
    ├── backport                            # Used for compatibility with the competitor code
    ├── competitor                          # Original competitor implementation which contains their original action-cost model and the black-box models, etc.
    ├── create_experiment_instances.py      # Script to produce the initial experiment instances for adult and german dataset
    ├── cscf                                # Implementation of the (C)SCF methods (i.e. the evolutionary algorithm and its decoder)
    ├── datasets                            # Files to handle the datasets
    ├── evaluation.py                       # Main script to run all experiments for our method(s)
    ├── feature_cost_model                  # Implementation of the consequential discount model (i.e. the relationship graph)
    ├── sequential                          # Helper classes for handling a sequence and actions
    └── util                                # Contains helper functions such as the Gower's distance
```

## Creating your own actions for other datasets
> TODO: More precise instructions may follow later

We recommend using *our* action-cost model implementation instead of the model of the competitor as it is easier to use with our method.
Example files are contained in `src/sequential/adult/` for the adult dataset (note, this was not used in the experiments of the paper, however, as we used the action-cost model of the competitor).

- For defining custom ***action*** classes, have a look at the file `src/sequential/adult/adult_actions.py` which contains custom actions defined by us for the adult dataset.
- For defining action ***cost*** functions, have a look at the file `src/sequential/adult/adult_costs.py` which contains custom action cost functions defined by us for the adult dataset.
- For defining action ***constraint*** functions, have a look at the file `src/sequential/adult/adult_constraints.py` which contains custom action constraints defined by us for the adult dataset.
- For defining a ***feature relationship graph***, have a look at the file `src/sequential/adult_dependency_graph.py` which contains the feature relationship graph for the adult dataset that was also used in the experiments of the paper.