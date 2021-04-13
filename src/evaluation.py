import copy
import json
import uuid
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np

from pymoo.optimize import minimize
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from competitor.models.loader import load_env, load_model

from backport.problem_factory import ProblemFactory
from backport.utils import RepresentationTranslator
from backport.wrapper.actions_wrapper import ActionWrapper
from backport.wrapper.datasets_wrapper import DatasetWrapper
from sequential.adult_dependency_graph import get_adult_dependency_graph
from feature_cost_model.feature_dependency_graph import FeatureDependencyGraph


from cscf.CSCF import CSCF

from scipy.special import softmax

from sacred import Experiment


experiment_name = "Comparison to competitor"
ex = Experiment(
    experiment_name,
    interactive=False,
)


def write_instances_to_file(
    instances, dataset_name, target_class, identifier, prefix=""
):
    Path(f"experiment_data/test_instances/").mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    file_path = (
        f"experiment_data/test_instances/{prefix}_{dataset_name}_n={len(instances)}.csv"
    )

    instances = np.array(instances)

    # check if already exists and output a warning if so in case the content is different than
    # what we want to write
    if Path(file_path).is_file():
        existing_instances = np.loadtxt(file_path, delimiter=",")
        assert np.array_equal(
            instances, existing_instances
        ), "Initial instances are not the same. Probably the random seed is not set or wrong"
        if not np.array_equal(instances, existing_instances):
            print(f"Existing array at {file_path} is not the same as the current one!")
            # add exp_id
            file_path = f"experiment_data/test_instances/{prefix}_UNEQUAL_{identifier}_{now.strftime('%d-%m-%Y')}_{dataset_name}_n={len(instances)}.csv"
    else:
        np.savetxt(file_path, instances, delimiter=",")
        print(f"Saved {len(instances)} to {file_path}")


#############
#  CONFIG   #
#############


@ex.config
def general_cfg():
    # General
    experiment_description = "Evaluation against competitor"
    notes = ""
    seed = 1
    # ! If the competitor model is used, we cannot use parallelization
    # ! as it seems to not work with tensorflow. I.e., only with RFC!
    parallelized = False

    # Specific
    n_experiment_instances = 100
    # This causes usage of a lot of memory
    save_optimal_population_trace = True


## EA mains
@ex.named_config
def comp_adult_problem_main_cfg():
    """
    Adult dataset experiments, for the original and main
    EA variant that I want to present
    """
    dataset_name = "adult"
    target_class = 1

    chosen_objectives = [
        "feature_tweaking_frequencies",
        "summed_costs",
        "gowers_distance",
    ]


@ex.named_config
def comp_german_problem_main_cfg():
    """
    German dataset experiments, for the original and main
    EA variant that I want to present
    """
    dataset_name = "german"
    target_class = 0

    chosen_objectives = [
        "feature_tweaking_frequencies",
        "summed_costs",
        "gowers_distance",
    ]


## EA alternatives
@ex.named_config
def comp_adult_problem_alternative_cfg():
    """
    Adult dataset experiments, for the alternative
    EA variant which is more similar to the competitor
    """
    dataset_name = "adult"
    target_class = 1

    chosen_objectives = [
        "feature_tweaking_frequencies",
        "summed_costs_discounted",
        "gowers_distance",
    ]


@ex.config
def optimization_cfg():
    # For the EA
    total_pop = 500
    # 1, because we use NDS, so it's changed to the number of ND solutions anyways
    n_elites_frac = 0.2
    # n_elites = int(n_elites_frac * total_pop)
    n_elites = 1  # doesn't make a difference since we use NDS and not crowding distance
    # offsprings_frac = 0.7
    offsprings_frac = 0.8
    n_offspring = int(offsprings_frac * total_pop)
    # mutants_frac = 0.1
    mutants_frac = 0.2
    n_mutants = int(mutants_frac * total_pop)
    bias = 0.7
    eliminate_duplicates = True

    # For the minimization
    n_generations = 150


@ex.config
def model_cfg():
    """
    Legacy now replaced
    """
    # comp-NN or rfc
    # model_type = "rfc"
    model_type = "comp-NN"

    # RFC
    n_estimators = 100


#############
#  SETUPS   #
#############


@ex.capture
def setup_experiment_instances(
    dataset, model, target_class, n_experiment_instances, seed
):

    # original represenation of the instances
    data_in_x = dataset.get_optimizer_data()
    _, labels = dataset.get_classifier_data()
    # ! Use the transformed, exact, instances to make the comparison fair
    # ! Now both methods use the exact same starting conditions
    data_in_z = dataset.encode_features(data_in_x)

    full_idx = np.arange(len(labels))

    # Create conditions for the instances we want to pick from, i.e. candidates
    is_opposite_originally = labels != target_class
    is_opposite_in_blackbox_z = (
        np.argmax(model.predict(data_in_z), axis=1) != target_class
    )

    # * Actually not needed anymore, but kept since it doesn't change anything
    is_opposite_in_blackbox_x = (
        np.argmax(model.predict(dataset.encode_features(data_in_x)), axis=1)
        != target_class
    )

    # Only use target class ones
    full_idx = full_idx[
        (is_opposite_originally)
        & (is_opposite_in_blackbox_z)
        & (is_opposite_in_blackbox_x)
    ]
    # Random order
    random_order = np.random.permutation(len(full_idx))[:n_experiment_instances]

    final_chosen_idx = full_idx[random_order]

    initial_instances_in_x = data_in_x[final_chosen_idx]
    initial_instances_in_z = data_in_z[final_chosen_idx]
    print(final_chosen_idx)
    print(
        np.argmax(
            model.predict(dataset.encode_features(initial_instances_in_x)), axis=1
        )
    )
    assert (
        np.argmax(
            model.predict(dataset.encode_features(initial_instances_in_x)), axis=1
        )
        != target_class
    ).all()
    print(np.argmax(model.predict(initial_instances_in_z), axis=1))
    assert (
        np.argmax(model.predict(initial_instances_in_z), axis=1) != target_class
    ).all()
    print(labels[final_chosen_idx])
    assert len(initial_instances_in_x) == n_experiment_instances
    assert len(initial_instances_in_z) == n_experiment_instances
    assert len(final_chosen_idx) == n_experiment_instances

    return initial_instances_in_x, initial_instances_in_z, final_chosen_idx


@ex.capture
def setup_boundaries_and_allowed_values(dataset, dataset_name):
    # if dataset_name == "adult":
    #     bounds_and_values = {
    #         # 0: np.unique(dataset.xs[:, 2]).tolist(),  # Edu unique values
    #         0: [-max(abs(dataset.xs[:, 2])), max(abs(dataset.xs[:, 2]))],  # Edu
    #         1: [-max(abs(dataset.xs[:, 9])), max(abs(dataset.xs[:, 9]))],  # Cap loss
    #         2: [-max(abs(dataset.xs[:, 10])), max(abs(dataset.xs[:, 10]))],  # Work hrs
    #         3: np.unique(dataset.xs[:, 4]).tolist(),  # Enlist
    #         4: [-max(abs(dataset.xs[:, 8])), max(abs(dataset.xs[:, 8]))],  # Cap gain
    #         5: [-max(abs(dataset.xs[:, 0])), max(abs(dataset.xs[:, 0]))],  # Years
    #     }
    # elif dataset_name == "german":
    #     bounds_and_values = {
    #         0: [-max(abs(dataset.xs[:, 12])), max(abs(dataset.xs[:, 12]))],  # years
    #         1: np.unique(dataset.xs[:, 19]).tolist(),
    #         2: [-max(abs(dataset.xs[:, 4])), max(abs(dataset.xs[:, 4]))],
    #         3: [-max(abs(dataset.xs[:, 1])), max(abs(dataset.xs[:, 1]))],
    #         4: [-max(abs(dataset.xs[:, 4])), max(abs(dataset.xs[:, 4]))],
    #         5: np.unique(dataset.xs[:, 9]).tolist(),
    #         6: np.unique(dataset.xs[:, 16]).tolist(),
    #     }

    # * As z-values
    if dataset_name == "adult":
        bounds_and_values = {
            # 0: np.unique(dataset.xs[:, 2]).tolist(),  # Edu unique values
            0: [-max(abs(dataset.zs[:, 8])), max(abs(dataset.zs[:, 8]))],  # Edu
            1: [-max(abs(dataset.zs[:, 44])), max(abs(dataset.zs[:, 44]))],  # Cap loss
            2: [-max(abs(dataset.zs[:, 45])), max(abs(dataset.zs[:, 45]))],  # Work hrs
            3: np.unique(dataset.xs[:, 4]).tolist(),  # Enlist
            4: [-max(abs(dataset.zs[:, 43])), max(abs(dataset.zs[:, 43]))],  # Cap gain
            5: [-max(abs(dataset.zs[:, 0])), max(abs(dataset.zs[:, 0]))],  # Years
        }
    elif dataset_name == "german":
        bounds_and_values = {
            0: [-max(abs(dataset.zs[:, 46])), max(abs(dataset.zs[:, 46]))],  # years
            1: np.unique(dataset.xs[:, 19]).tolist(),
            2: [-max(abs(dataset.zs[:, 21])), max(abs(dataset.zs[:, 21]))],
            3: [-max(abs(dataset.zs[:, 4])), max(abs(dataset.zs[:, 4]))],
            4: [-max(abs(dataset.zs[:, 21])), max(abs(dataset.zs[:, 21]))],
            5: np.unique(dataset.xs[:, 9]).tolist(),
            6: np.unique(dataset.xs[:, 16]).tolist(),
        }
    return bounds_and_values


@ex.capture
def setup_dependency_graph(features, dataset_name):
    if dataset_name == "adult":
        dependency_graph = get_adult_dependency_graph(features)
    elif dataset_name == "german":
        dependency_graph = FeatureDependencyGraph(None)
    return dependency_graph


@ex.capture
def setup_actions(x0, legacy_actions, legacy_features, features, dataset_name):
    translator = RepresentationTranslator(legacy_features)

    if dataset_name == "german":
        (
            waitYears,
            naturalize,
            chCreditAm,
            chLoanPeriod,
            adjLoanPeriod,
            getGuarantor,
            getUnskilledJob,
        ) = legacy_actions

        actions = {
            0: {
                "action_id": 0,
                "legacy_action": waitYears,
                "translator": translator,
                "feature_idx": features["age_in_years"],
                "feature_idx_in_z": 46,
                "initial_instance": x0,
                "is_categorical": False,
                "feat_name": "age_in_years",
                "key": 0,
            },
            1: {
                "action_id": 1,
                "legacy_action": naturalize,
                "translator": translator,
                "feature_idx": features["foreign_worker"],
                "feature_idx_in_z": 19,
                "initial_instance": x0,
                "is_categorical": True,
                "feat_name": "foreign_worker",
                "key": 1,
            },
            2: {
                "action_id": 2,
                "legacy_action": chCreditAm,
                "translator": translator,
                "feature_idx": features["credit_amount"],
                "feature_idx_in_z": 21,
                "initial_instance": x0,
                "is_categorical": False,
                "feat_name": "credit_amount",
                "key": 2,
            },
            3: {
                "action_id": 3,
                "legacy_action": chLoanPeriod,
                "translator": translator,
                "feature_idx": features["loan_duration"],
                "feature_idx_in_z": 4,
                "initial_instance": x0,
                "is_categorical": False,
                "feat_name": "loan_duration",
                "key": 3,
            },
            4: {
                "action_id": 4,
                "legacy_action": adjLoanPeriod,
                "translator": translator,
                "feature_idx": features["credit_amount"],
                "feature_idx_in_z": 21,
                "initial_instance": x0,
                "is_categorical": False,
                "feat_name": "credit_amount",
                "key": 4,
            },
            5: {
                "action_id": 5,
                "legacy_action": getGuarantor,
                "translator": translator,
                "feature_idx": features["other_debtors_guarantors"],
                "feature_idx_in_z": 9,
                "initial_instance": x0,
                "is_categorical": True,
                "feat_name": "other_debtors_guarantors",
                "key": 5,
            },
            6: {
                "action_id": 6,
                "legacy_action": getUnskilledJob,
                "translator": translator,
                "feature_idx": features["job"],
                "feature_idx_in_z": 16,
                "initial_instance": x0,
                "is_categorical": True,
                "feat_name": "job",
                "key": 6,
            },
        }
    elif dataset_name == "adult":
        addEdu, chWorkHrs, chCapLoss, enlist, incCapGain, waitYears = legacy_actions

        actions = {
            0: {
                "action_id": 0,
                "legacy_action": addEdu,
                "translator": translator,
                "feature_idx": features["Education num"],
                "feature_idx_in_z": 8,
                "initial_instance": x0,
                "is_categorical": False,
                "feat_name": "Education num",
                "key": 0,
            },
            1: {
                "action_id": 1,
                "legacy_action": chCapLoss,
                "translator": translator,
                "feature_idx": features["Capital Loss"],
                "feature_idx_in_z": 44,
                "initial_instance": x0,
                "is_categorical": False,
                "feat_name": "Capital Loss",
                "key": 1,
            },
            2: {
                "action_id": 2,
                "legacy_action": chWorkHrs,
                "translator": translator,
                "feature_idx": features["Hours/Week"],
                "feature_idx_in_z": 45,
                "initial_instance": x0,
                "is_categorical": False,
                "feat_name": "Hours/Week",
                "key": 2,
            },
            3: {
                "action_id": 3,
                "legacy_action": enlist,
                "translator": translator,
                "feature_idx": features["Occupation"],
                "feature_idx_in_z": 4,
                "initial_instance": x0,
                "is_categorical": True,
                "feat_name": "Occupation",
                "key": 3,
            },
            4: {
                "action_id": 4,
                "legacy_action": incCapGain,
                "translator": translator,
                "feature_idx": features["Capital Gain"],
                "feature_idx_in_z": 43,
                "initial_instance": x0,
                "is_categorical": False,
                "feat_name": "Capital Gain",
                "key": 4,
            },
            5: {
                "action_id": 5,
                "legacy_action": waitYears,
                "translator": translator,
                "feature_idx": features["Age"],
                "feature_idx_in_z": 0,
                "initial_instance": x0,
                "is_categorical": False,
                "feat_name": "Age",
                "key": 5,
            },
        }
    return actions


@ex.capture
def setup_problem(
    dataset,
    actions,
    x0,
    blackbox,
    blackbox_prob,
    bounds_and_values,
    features,
    G,
    target_class,
    chosen_objectives,
    seed,
):

    problem = ProblemFactory(
        copy.copy(dataset),
        copy.copy(actions),
        copy.copy(x0),
        target_class=target_class,
        blackbox_classifier=blackbox,
        blackbox_probabilitiy=blackbox_prob,
        bounds_and_values=copy.copy(bounds_and_values),
        feature=features,
        categorical_features=dataset.cat_features,
        dependency_graph=G,
        chosen_objectives=chosen_objectives,
        legacy_mode=True,  # important!
    )
    return problem


@ex.capture
def setup_dataset(dataset_name, seed):
    if dataset_name == "german":
        legacy_dataset, legacy_actions, legacy_features, _ = load_env(
            dataset_name, "data.npy"
        )
    elif dataset_name == "adult":
        legacy_dataset, legacy_actions, legacy_features, _ = load_env(
            dataset_name, "data.npy"
        )

    dataset = DatasetWrapper(dataset_name, legacy_dataset, legacy_features)

    return dataset, legacy_actions, legacy_features


@ex.capture
def setup_model(dataset, dataset_name, model_type, n_estimators, seed):
    X_train, y_train = dataset.get_classifier_data()
    X_test, y_test = dataset.get_test_data()

    if model_type == "rfc":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        # Train it
        model.fit(X_train, y_train)
    elif model_type == "comp-NN":
        model = load_model(dataset_name, "model.h5")
    return model


@ex.capture
def setup_optimizer(n_elites, n_offspring, n_mutants, bias, eliminate_duplicates, seed):
    algorithm = CSCF(
        n_elites=n_elites,
        n_offsprings=n_offspring,
        n_mutants=n_mutants,
        bias=bias,
        eliminate_duplicates=eliminate_duplicates,
    )
    return algorithm


@ex.capture
def get_blackboxes(model, dataset, model_type):
    if model_type == "rfc":

        def blackbox(X):
            _X = X.copy()
            if _X.ndim == 1:
                _X = _X.reshape(1, -1)
            return model.predict(dataset.encode_features(_X))

        def blackbox_prob(X):
            _X = X.copy()
            if _X.ndim == 1:
                _X = _X.reshape(1, -1)
            return model.predict_proba(dataset.encode_features(_X))

    elif model_type == "comp-NN":

        def blackbox(X):
            _X = X.copy()
            if _X.ndim == 1:
                _X = _X.reshape(1, -1)
            return np.argmax(model.predict(dataset.encode_features(_X)), axis=1)

        def blackbox_prob(X):
            _X = X.copy()
            if _X.ndim == 1:
                _X = _X.reshape(1, -1)
            # * Use softmax to get the actual probabilities
            return softmax(model.predict(dataset.encode_features(_X)), axis=1)

    return blackbox, blackbox_prob


def run_optimization_problem(
    i,
    x0,
    legacy_actions,
    legacy_features,
    dataset,
    bounds_and_values,
    target_class,
    model_type,
    n_generations,
    seed,
    experiment_id,
    save_optimal_population_trace,
    features,
    model=None,
):
    if model is None:
        model = setup_model(dataset)
    blackbox, blackbox_prob = get_blackboxes(model, dataset)
    actions = setup_actions(x0, legacy_actions, legacy_features, features)
    dependency_graph = setup_dependency_graph(features)
    problem = setup_problem(
        dataset,
        actions,
        x0,
        blackbox,
        blackbox_prob,
        bounds_and_values,
        features,
        dependency_graph,
    )
    algorithm = setup_optimizer()
    # termination = MultiObjectiveDefaultTermination(
    #     x_tol=1e-8,
    #     cv_tol=1e-6,
    #     f_tol=0.0025,
    #     nth_gen=5,
    #     n_last=30,
    #     n_max_gen=n_generations,
    #     n_max_evals=100000,
    # )
    termination = ("n_gen", n_generations)

    print(f"Running experiment {i}")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        display=MultiObjectiveDisplay(),
        save_history=save_optimal_population_trace,
        verbose=False,
    )

    now = datetime.now()
    # Create dir if not exists
    folder_name = (
        f"{now.strftime('%d-%m-%Y')} {experiment_id} {dataset.name}/Final Results"
    )
    Path(f"results/{folder_name}/").mkdir(parents=True, exist_ok=True)
    try:
        file_path = f"results/{folder_name}/{experiment_id}_{now.strftime('%d-%m-%Y')}_{dataset.name}_res_opt_{i}.json"
        info = {
            "experiment": experiment_name,
            "experiment_id": experiment_id,
            "date": now.strftime("%d/%m/%Y at %H:%M:%S"),
            "target_class": target_class,
            "model_type": model_type,
            "x0_id": i,
            "x0": x0.tolist(),
            "iterations": n_generations,
            "seed": seed,
            "dataset": dataset.name,
            "success": True,
            "runtime": res.exec_time,
            "n_solutions": len(res.X),
            "X": res.X.tolist(),
            "F": res.F.tolist(),
            "G": res.G.tolist(),
            "pheno": res.opt.get("pheno").tolist(),
            "tweaked_res": res.opt.get("tweaked_instances").tolist(),
            "summed_costs": res.opt.get("summed_costs").tolist(),
            "summed_costs_discounted": res.opt.get("summed_costs_discounted").tolist(),
        }
    except Exception as e:
        file_path = f"results/{folder_name}/{experiment_id}_{now.strftime('%d-%m-%Y')}_{dataset.name}_res_opt_{i}_fail.json"
        info = {
            "experiment": experiment_name,
            "experiment_id": experiment_id,
            "date": now.strftime("%d/%m/%Y at %H:%M:%S"),
            "target_class": target_class,
            "model_type": model_type,
            "x0_id": i,
            "x0": x0.tolist(),
            "iterations": n_generations,
            "seed": seed,
            "dataset": dataset.name,
            "success": False,
            "exception": str(e),
            "runtime": res.exec_time,
            "n_solutions": 0,
            "X": [],
            "F": [],
            "G": [],
            "pheno": [],
            "tweaked_res": [],
            "summed_costs": [],
            "summed_costs_discounted": [],
        }
        print(f"Run {i}:", e)

    with open(file_path, "w+") as f:
        json.dump(info, f, indent=4)

    # Also save optimal population info per generation
    if save_optimal_population_trace:
        # Create dir if not exists
        folder_name = (
            f"{now.strftime('%d-%m-%Y')} {experiment_id} {dataset.name}/History Results"
        )
        Path(f"results/{folder_name}/").mkdir(parents=True, exist_ok=True)
        try:
            # i denotes the generation here
            all_X = {i: h.opt.get("X").tolist() for i, h in enumerate(res.history)}
            all_F = {i: h.opt.get("F").tolist() for i, h in enumerate(res.history)}
            all_G = {i: h.opt.get("G").tolist() for i, h in enumerate(res.history)}
            all_pheno = {
                i: h.opt.get("pheno").tolist() for i, h in enumerate(res.history)
            }
            all_tweaked = {
                i: h.opt.get("tweaked_instances").tolist()
                for i, h in enumerate(res.history)
            }
            all_summed_costs = {
                i: h.opt.get("summed_costs").tolist() for i, h in enumerate(res.history)
            }
            all_summed_costs_discounted = {
                i: h.opt.get("summed_costs_discounted").tolist()
                for i, h in enumerate(res.history)
            }

            file_path = f"results/{folder_name}/{experiment_id}_{now.strftime('%d-%m-%Y')}_{dataset.name}_history_opt_{i}.json"
            info = {
                "experiment": experiment_name,
                "experiment_id": experiment_id,
                "date": now.strftime("%d/%m/%Y at %H:%M:%S"),
                "target_class": target_class,
                "model_type": model_type,
                "x0_id": i,
                "x0": x0.tolist(),
                "iterations": n_generations,
                "seed": seed,
                "dataset": dataset.name,
                "success": True,
                "runtime": res.exec_time,
                "n_solutions": len(res.X),
                "X": all_X,
                "F": all_F,
                "G": all_G,
                "pheno": all_pheno,
                "tweaked_res": all_tweaked,
                "summed_costs": all_summed_costs,
                "summed_costs_discounted": all_summed_costs_discounted,
            }
        except Exception as e:
            file_path = f"results/{folder_name}/{experiment_id}_{now.strftime('%d-%m-%Y')}_{dataset.name}_history_opt_{i}_fail.json"
            info = {
                "experiment": experiment_name,
                "experiment_id": experiment_id,
                "date": now.strftime("%d/%m/%Y at %H:%M:%S"),
                "target_class": target_class,
                "model_type": model_type,
                "x0_id": i,
                "x0": x0.tolist(),
                "iterations": n_generations,
                "seed": seed,
                "dataset": dataset.name,
                "success": False,
                "exception": str(e),
                "runtime": res.exec_time,
                "n_solutions": 0,
                "X": [],
                "F": [],
                "G": [],
                "pheno": [],
                "tweaked_res": [],
                "summed_costs": [],
                "summed_costs_discounted": [],
            }
            print(f"Run {i}:", e)

        with open(file_path, "w+") as f:
            json.dump(info, f, indent=4)

    return res.exec_time


@ex.automain
def run(_run):
    np.random.seed(_run.config["seed"])

    # Dump config
    identifier = str(uuid.uuid1())[:7]
    ex.add_config({"experiment_id": identifier})
    now = datetime.now()

    dataset, legacy_actions, legacy_features = setup_dataset()
    features = {b: int(a) for a, b in enumerate(dataset.columns)}
    assert np.all([x in dataset.columns for x in features.keys()])
    model = setup_model(dataset)
    blackbox, blackbox_prob = get_blackboxes(model, dataset)
    bounds_and_values = setup_boundaries_and_allowed_values(dataset)

    (
        initial_instances_orig,
        initial_instances_proc,
        instances_random_idx,
    ) = setup_experiment_instances(dataset, model)
    # write the initial instances to a file for later usage
    write_instances_to_file(
        initial_instances_orig,
        _run.config["dataset_name"],
        _run.config["target_class"],
        identifier,
        prefix="original",
    )
    write_instances_to_file(
        initial_instances_proc,
        _run.config["dataset_name"],
        _run.config["target_class"],
        identifier,
        prefix="processed",
    )

    # Write config
    Path(f"results/").mkdir(parents=True, exist_ok=True)
    d_name = _run.config["dataset_name"]
    file_path = f"results/{identifier}_config_{now.strftime('%d-%m-%Y at %H-%M')}_{d_name}_{experiment_name}.json"
    additional_info = {
        "experiment_id": identifier,
        "initial_random_idx": instances_random_idx.tolist(),
    }
    config_dict = {**additional_info, **_run.config.copy()}
    with open(file_path, "w+") as f:
        json.dump(config_dict, f, indent=4)

    if _run.config["parallelized"]:
        # Parellelize
        nprocs = mp.cpu_count()
        print(f"Number of CPU cores: {nprocs}")
        with mp.Pool(processes=nprocs) as pool:
            experiments = [
                (
                    i,
                    x0,
                    legacy_actions,
                    legacy_features,
                    dataset,
                    bounds_and_values,
                    _run.config["target_class"],
                    _run.config["model_type"],
                    _run.config["n_generations"],
                    _run.config["seed"],
                    identifier,
                    _run.config["save_optimal_population_trace"],
                    features,
                    model,
                )
                for i, x0 in enumerate(initial_instances_orig)
            ]
            runtimes = pool.starmap(run_optimization_problem, experiments)
    else:
        # Sequentially
        runtimes = []
        for i, x0 in enumerate(initial_instances_orig):
            print(f"Running experiment for instance {i}")
            time = run_optimization_problem(
                i,
                x0,
                legacy_actions,
                legacy_features,
                dataset,
                bounds_and_values,
                _run.config["target_class"],
                _run.config["model_type"],
                _run.config["n_generations"],
                _run.config["seed"],
                identifier,
                _run.config["save_optimal_population_trace"],
                features,
                model,
            )
            runtimes.append(time)
            print(f"Experiment finished after {time}")
        print(runtimes)
        print(np.mean(runtimes))
        print("Finished")
