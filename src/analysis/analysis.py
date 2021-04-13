import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import copy

import seaborn as sns
from scipy.stats import wilcoxon

import sys, os

sys.path.append("..")

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from competitor.models.loader import load_env, load_model
from competitor.recourse.utils import relu_cost_fn
from backport.wrapper.datasets_wrapper import DatasetWrapper

from competitor.actions.feature import CategoricFeature, Feature

from sequential.sequence import Sequence
from scipy.special import softmax
from competitor.models.loader import load_env, load_model

from pysankey2 import Sankey, setColorConf

import evaluation


np.random.seed(1)

plt.style.use(["science"])

BASE_PLOTS_FOLDER = "plots"

MY_METHOD_RESULTS_PATH = f"results/my_method"
COMPETITOR_RESULTS_PATH = f"results/competitor"

MAIN_NAME = r"$\textsc{scf}$"
ALT_NAME = r"$\textsc{cscf}$"
COMP_NAME = r"$\textsc{synth}$"


def save_fig(title, extension="pdf", dpi=1200, tight=True, *args, **kwargs):
    Path(BASE_PLOTS_FOLDER).mkdir(parents=True, exist_ok=True)
    if tight:
        plt.savefig(
            f"{BASE_PLOTS_FOLDER}/{title}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
            *args,
            **kwargs,
        )
    else:
        plt.savefig(
            f"{BASE_PLOTS_FOLDER}/{title}.{extension}", dpi=dpi, *args, **kwargs
        )


def read_data(results_path):
    data = []
    for file_path in Path(results_path).glob("**/*"):
        if file_path.is_file():
            with open(file_path, "r") as f:
                data.append(json.load(f))
    return data


def setup_dataset(dataset_name):
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


def violates_constraints(x0, instance_in_x, dataset_name):
    x = np.around(instance_in_x, 0)  # avoid precision problems
    violates = False
    if dataset_name == "german":
        violates = (
            (x[12] < x0[12])
            or (x[12] > 120)
            or (x[4] < 0)
            or (x[4] > 100000)
            or (x[1] < 0)
            or (x[1] > 120)
        )
    elif dataset_name == "adult":
        violates = (
            (x[0] < x0[0])
            or (x[0] > 120)
            or (x[10] < 0)
            or (x[10] > 90)
            or (int(x[8]) < 0)  # precsion problems otherwise
            or (x[8] > 100000)
            or (int(x[9]) < 0)  # precsion problems otherwise
            or (x[9] > 5000)
            or (x[2] < x0[2])
            or (x[2] > 16.5)
        )
    return violates


def get_sequence_length(sequence):
    sequence = np.array(sequence)
    # action part
    sequence = sequence[: len(sequence) // 2]
    length = len(sequence[sequence != -1])
    return length


def get_data(dataset_name, max_length=2):
    DATASET = dataset_name
    my_data_path = f"{MY_METHOD_RESULTS_PATH}/{DATASET}_scf/"
    competitor_data_path = f"{COMPETITOR_RESULTS_PATH}/{DATASET}/"

    my_other_data_path = f"{MY_METHOD_RESULTS_PATH}/{DATASET}_cscf/"

    my_data = read_data(my_data_path)
    if dataset_name == "adult":
        my_other_data = read_data(my_other_data_path)
    # Since we don't have CSCF for german, just use the other data as a copy.
    # It will not be used anyway
    else:
        my_other_data = copy.deepcopy(my_data)
    competitor_data = read_data(competitor_data_path)

    md, cd, mod, index = get_polished_data(my_data, competitor_data, my_other_data)
    return md, cd, mod


def get_polished_data(_my_data, _competitor_data, _my_other_data):
    # sort all records according to the original idx so they match
    my_data_idx = [record["x0_id"] for record in _my_data]
    my_data = np.array(_my_data)[np.argsort(my_data_idx)]  # .tolist()

    my_other_data_idx = [record["x0_id"] for record in _my_other_data]
    my_other_data = np.array(_my_other_data)[np.argsort(my_other_data_idx)]  # .tolist()

    competitor_idx = [record["idx"] for record in _competitor_data]
    competitor_data = np.array(_competitor_data)[
        np.argsort(competitor_idx)
    ]  # .tolist()

    assert (np.sort(competitor_idx) == np.sort(my_data_idx)).all() and (
        np.sort(my_other_data_idx) == np.sort(my_data_idx)
    ).all()

    # for later reference
    index = np.arange(len(my_data))

    # discard unsuccesful runs
    my_successful = np.array([record["success"] for record in my_data], dtype=bool)
    my_other_successful = np.array(
        [record["success"] for record in my_other_data], dtype=bool
    )
    comp_successful = np.array(
        [record["success"] for record in competitor_data], dtype=bool
    )
    successful = my_successful & comp_successful & my_other_successful
    my_data = my_data[successful]
    my_other_data = my_other_data[successful]
    competitor_data = competitor_data[successful]
    index = index[successful]

    # violating instances of competitor
    # ! only checking competitor for violations since the EA
    # ! always produces feasible solutions
    x0s = [record["x0"] for record in my_data]
    dataset, legacy_actions, legacy_features = setup_dataset(
        competitor_data[0]["model"]
    )
    columns = dataset.columns.tolist()
    ## compeetitor
    _comp_sols = [
        record["output"]["best_result"]["final_instance_info"]
        for record in competitor_data
    ]
    comp_sols = np.full((len(_comp_sols), len(columns)), np.nan)
    # create x-space instance
    for i, sol in enumerate(_comp_sols):
        for feature_name, value in sol.items():
            actual_idx = columns.index(feature_name)
            comp_sols[i, actual_idx] = float(value)
    comp_violating = []
    for xx0, sol in zip(x0s, comp_sols):
        if sol is not np.nan:
            v = violates_constraints(xx0, sol, competitor_data[0]["model"])
        else:
            print("was nan, but shouldnt be")
            v = True
        comp_violating.append(v)
    comp_violating = np.array(comp_violating, dtype=bool)

    my_data = my_data[~comp_violating]
    my_other_data = my_other_data[~comp_violating]
    competitor_data = competitor_data[~comp_violating]
    index = index[~comp_violating]

    return my_data.tolist(), competitor_data.tolist(), my_other_data.tolist(), index


def get_df_melted(values, label, group=None):
    _df = pd.DataFrame({"values": values})
    _df["label"] = label
    if group is not None:
        _df["group"] = group
    return _df


def get_optimal_solution_costs(
    data_record,
    relevant_objectives=None,
    aggregation="sum",
    max_length=None,
    which="summed_costs",
):
    if not data_record["success"]:
        print("shouldnt here")
        return np.nan, np.nan

    # no solution was found
    if len(data_record) == 0:
        print("shouldnt here")
        return np.nan, np.nan

    # solutions is too long
    valid = [True] * len(data_record[which])
    if max_length is not None:
        sequences = np.array(data_record["pheno"])
        for i, sequence in enumerate(sequences):
            length = get_sequence_length(sequence)
            if length > max_length:
                valid[i] = False
    # no solution with required length
    if not any(valid):
        return np.nan, np.nan

    # fitness_values = np.array(data_record["F"])[valid, :]
    fitness_values = np.array([x for x in data_record[which]]).flatten()[valid]
    full_idx = np.arange(len(data_record[which]))[valid]

    aggregated = fitness_values
    assert not np.nan in aggregated
    return np.min(aggregated), full_idx[np.argmin(aggregated)]


def get_summed_costs(md, cd, mod, max_length=2, which="summed_costs"):
    # My costs
    fitness_values = [np.array(record["F"]) for record in md]
    index_of_costs = np.arange(fitness_values[0].shape[1] - 3)
    costs_and_idx = [
        get_optimal_solution_costs(
            record,
            relevant_objectives=index_of_costs,
            aggregation="sum",
            max_length=max_length,
            which=which,
        )
        for record in md
    ]
    my_least_cost_sols = [c[0] for c in costs_and_idx]
    my_least_cost_idx = [c[1] for c in costs_and_idx]

    # my other costs
    ot_fitness_values = [np.array(record["F"]) for record in mod]
    ot_index_of_costs = [0]
    ot_costs_and_idx = [
        get_optimal_solution_costs(
            record,
            relevant_objectives=ot_index_of_costs,
            aggregation="sum",
            max_length=max_length,
            which=which,
        )
        for record in mod
    ]
    ot_my_least_cost_sols = [c[0] for c in ot_costs_and_idx]
    ot_my_least_cost_idx = [c[1] for c in ot_costs_and_idx]

    # Competitor costs
    competitor_least_costs = [record["output"]["best_result"]["cost"] for record in cd]

    is_not_nan = (
        np.array([x is not np.nan for x in my_least_cost_sols], dtype=bool)
        & np.array([x is not np.nan for x in ot_my_least_cost_sols], dtype=bool)
        & np.array([x is not np.nan for x in competitor_least_costs], dtype=bool)
    )

    return (
        np.array(my_least_cost_sols)[is_not_nan],
        np.array(my_least_cost_idx)[is_not_nan],
        np.array(competitor_least_costs)[is_not_nan],
        np.array(ot_my_least_cost_sols)[is_not_nan],
        np.array(ot_my_least_cost_idx)[is_not_nan],
        is_not_nan,
    )


def compute_sequential_tweak_probabilities(
    sequences,
    tweaking_values,
    original_instance,
    likelihood_gain_oracle,
    target_class,
    valid,
):
    all_tweaked_instances = []
    seq_lengths = []
    for seq, tweak, is_valid in zip(sequences, tweaking_values, valid):
        if is_valid:
            tweaked_instances = seq.get_tweaked_instance_after_each_action(
                original_instance.copy(), tweak
            ).reshape(-1, len(original_instance))
            # add original instance
            tweaked_instances = np.row_stack([[original_instance], tweaked_instances])
            all_tweaked_instances.append(tweaked_instances)
            seq_lengths.append(seq.length + 1)
        else:
            all_tweaked_instances.append(original_instance)  # dummy
            seq_lengths.append(1)
    all_tweaked_instances = np.row_stack(
        all_tweaked_instances
    )  # .reshape(-1, len(self.x0))
    # predictions
    inv_target_class_probs = likelihood_gain_oracle(all_tweaked_instances)[
        :, target_class
    ].flatten()

    per_seq_probs = np.split(inv_target_class_probs, np.cumsum(seq_lengths))[:-1]

    return per_seq_probs[0]


def plot_costs_comparison():
    (
        my_c_german,
        my_idx_german,
        comp_c_german,
        my_ot_c_german,
        my_ot_idx_german,
        german_is_not_nan,
    ) = get_summed_costs(
        my_german, comp_german, my_other_german, max_length=2, which="summed_costs"
    )
    (
        my_c_adult,
        my_idx_adult,
        comp_c_adult,
        my_ot_c_adult,
        my_ot_idx_adult,
        adult_is_not_nan,
    ) = get_summed_costs(
        my_adult, comp_adult, my_other_adult, max_length=2, which="summed_costs"
    )

    my_df_ger = get_df_melted(my_c_german, MAIN_NAME, "german")
    # my_ot_df_ger = get_df_melted(my_ot_c_german, "Mine B", "german")
    comp_df_ger = get_df_melted(comp_c_german, COMP_NAME, "german")

    my_df_ad = get_df_melted(my_c_adult, MAIN_NAME, "adult")
    my_ot_df_ad = get_df_melted(my_ot_c_adult, ALT_NAME, "adult")
    comp_df_ad = get_df_melted(comp_c_adult, COMP_NAME, "adult")

    df = pd.concat(
        [my_df_ger, comp_df_ger, my_df_ad, my_ot_df_ad, comp_df_ad]
    ).reset_index(drop="index")
    df.columns = ["Costs", "Method", "Dataset"]

    pairs = [
        (MAIN_NAME, COMP_NAME, "adult"),
        (ALT_NAME, COMP_NAME, "adult"),
        (MAIN_NAME, ALT_NAME, "adult"),
        (MAIN_NAME, COMP_NAME, "german"),
    ]

    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)

    with plt.style.context(["science"]):
        i = 0
        for ax, pair in zip(axes.flatten(), pairs):
            A, B, dat = pair
            MINE = df[(df["Method"] == A) & (df["Dataset"] == dat)]["Costs"]
            COMP = df[(df["Method"] == B) & (df["Dataset"] == dat)]["Costs"]
            diff = (COMP.values - MINE.values) / np.maximum(MINE.values, COMP.values)
            # diff = -1*(MINE.values - COMP.values)  / COMP.values
            ddd = pd.DataFrame(diff)
            ddd.columns = ["dif"]
            ddd["colors"] = ["red" if x < 0 else "green" for x in ddd["dif"]]
            ddd.sort_values("dif", inplace=True)
            ddd.reset_index(inplace=True)

            # Draw plot
            # fig = plt.figure(figsize=(4,3))#, dpi= 80)
            fig.set_size_inches(10, 2)
            higher = ddd.dif >= 0
            ax.hlines(
                y=ddd.index[higher],
                xmin=0,
                xmax=ddd.dif[higher],
                color=ddd.colors[higher],
                alpha=0.95,
                linewidth=1.1,
                label=f"{A} ($A$)",
            )
            lower = ddd.dif < 0
            ax.hlines(
                y=ddd.index[lower],
                xmin=0,
                xmax=ddd.dif[lower],
                color=ddd.colors[lower],
                alpha=0.95,
                linewidth=1.1,
                label=f"{B} ($B$)",
            )

            minimal_pos_element = ddd.dif.tolist().index(ddd.dif[ddd.dif >= 0].min())
            ax.axhline(minimal_pos_element, linewidth=2)

            # Decorations
            ax.grid(linestyle="--", alpha=0.5)
            leg = ax.legend(
                loc="lower right", handlelength=0.7, borderpad=0.1, prop={"size": 8}
            )
            for legobj in leg.legendHandles:
                legobj.set_linewidth(3.0)

            ax.set_title(f"{A} vs. {B} ({dat.title()})")
            li = max(np.abs(ddd["dif"]))
            ax.set_xlim(-1, 1)

            si = 14

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(si)

            ax.set_xlabel("Relative Difference", fontsize=si)
            if i == 0:
                ax.set_ylabel(f"Experiment", fontsize=si)
            i += 1

    save_fig(f"cost_diff", tight=True)


def plot_positional_action_probabilities():
    # german
    german_mapping = {
        0.0: "waitYears",
        1.0: "naturalize",
        2.0: "chCreditAm",
        3.0: "chLoanPeriod",
        4.0: "adjLoanPeriod",
        5.0: "guarantor",
        6.0: "unskilledJob",
        -1.0: "unused",
    }
    # adult
    adult_mapping = {
        0.0: "addEdu",
        1.0: "chCapLoss",
        2.0: "chWorkHrs",
        3.0: "enlist",
        4.0: "chCapGain",
        5.0: "waitYears",
        -1.0: "unused",
    }

    DATASET_NAME = "adult"
    target_class = 1

    which_data = my_other_adult

    x0s = np.array([np.array(record["x0"]) for record in which_data])

    assignments1 = pd.DataFrame(
        columns=[
            "Probability of $f$",
            "Position $t$ in $\mathcal{S}$",
            "Action",
            "Method",
        ]
    )

    for i, x0 in enumerate(x0s):
        seed = 1
        dataset, legacy_actions, legacy_features = evaluation.setup_dataset(
            DATASET_NAME, seed
        )
        features = {b: int(a) for a, b in enumerate(dataset.columns)}
        model = evaluation.setup_model(
            dataset=dataset,
            dataset_name=DATASET_NAME,
            model_type="comp-NN",
            n_estimators=1,
            seed=seed,
        )
        blackbox, blackbox_prob = evaluation.get_blackboxes(
            model=model, dataset=dataset, model_type="comp-NN"
        )
        actions = evaluation.setup_actions(
            x0, legacy_actions, legacy_features, features, DATASET_NAME
        )
        bounds_and_values = evaluation.setup_boundaries_and_allowed_values(
            dataset, DATASET_NAME
        )
        problem = evaluation.setup_problem(
            dataset,
            actions,
            x0,
            blackbox,
            blackbox_prob,
            bounds_and_values,
            features,
            G=None,
            target_class=target_class,
            chosen_objectives=["summed_costs"],
            seed=seed,
        )

        final_sols = np.row_stack(which_data[i]["pheno"])
        assert final_sols.ndim == 2
        for sol in final_sols:
            if problem.seq_length(sol) <= 100:
                seqs = problem.create_sequence(sol)
                tweaking_values = problem.get_tweaking_values(sol)
                probs = compute_sequential_tweak_probabilities(
                    [seqs], [tweaking_values], x0, blackbox_prob, target_class, [True]
                )[1:]
                j = 1
                for acs, p in zip(seqs.sequence, probs):
                    assignments1 = assignments1.append(
                        {
                            "Probability of $f$": p,
                            "Position $t$ in $\mathcal{S}$": j,
                            "Action": acs.key,
                            "Method": f"{ALT_NAME}",
                        },
                        ignore_index=True,
                    )
                    j += 1

    assignments1["Action"] = assignments1["Action"].replace(adult_mapping)

    # next

    which_data = my_adult

    x0s = np.array([np.array(record["x0"]) for record in which_data])

    assignments2 = pd.DataFrame(
        columns=[
            "Probability of $f$",
            "Position $t$ in $\mathcal{S}$",
            "Action",
            "Method",
        ]
    )

    for i, x0 in enumerate(x0s):
        seed = 1
        dataset, legacy_actions, legacy_features = evaluation.setup_dataset(
            DATASET_NAME, seed
        )
        features = {b: int(a) for a, b in enumerate(dataset.columns)}
        model = evaluation.setup_model(
            dataset=dataset,
            dataset_name=DATASET_NAME,
            model_type="comp-NN",
            n_estimators=1,
            seed=seed,
        )
        blackbox, blackbox_prob = evaluation.get_blackboxes(
            model=model, dataset=dataset, model_type="comp-NN"
        )
        actions = evaluation.setup_actions(
            x0, legacy_actions, legacy_features, features, DATASET_NAME
        )
        bounds_and_values = evaluation.setup_boundaries_and_allowed_values(
            dataset, DATASET_NAME
        )
        problem = evaluation.setup_problem(
            dataset,
            actions,
            x0,
            blackbox,
            blackbox_prob,
            bounds_and_values,
            features,
            G=None,
            target_class=target_class,
            chosen_objectives=["summed_costs"],
            seed=seed,
        )

        final_sols = np.row_stack(which_data[i]["pheno"])
        assert final_sols.ndim == 2
        for sol in final_sols:
            if problem.seq_length(sol) <= 100:
                seqs = problem.create_sequence(sol)
                tweaking_values = problem.get_tweaking_values(sol)
                probs = compute_sequential_tweak_probabilities(
                    [seqs], [tweaking_values], x0, blackbox_prob, target_class, [True]
                )[1:]
                j = 1
                for acs, p in zip(seqs.sequence, probs):
                    assignments2 = assignments2.append(
                        {
                            "Probability of $f$": p,
                            "Position $t$ in $\mathcal{S}$": j,
                            "Action": acs.key,
                            "Method": f"{MAIN_NAME}",
                        },
                        ignore_index=True,
                    )
                    j += 1

    assignments2["Action"] = assignments2["Action"].replace(adult_mapping)

    _assignments = pd.concat([assignments2, assignments1], ignore_index=True)

    _assignments.columns = [
        "$f(\mathbf{x}_t)$",
        "Position $t$ in $\mathcal{S}$",
        "Action",
        "Method",
    ]
    _assignments["Position $t$ in $\mathcal{S}$"] = _assignments[
        "Position $t$ in $\mathcal{S}$"
    ].astype(int)

    aa = _assignments[
        (_assignments["Method"] == MAIN_NAME) | (_assignments["Method"] == ALT_NAME)
    ]
    # aa = _assignments
    g = sns.FacetGrid(
        aa, col="Action", col_wrap=3, height=1.2, aspect=1.5, legend_out=False
    )  # ylim=(0, 10))
    g.map(
        sns.pointplot,
        "Position $t$ in $\mathcal{S}$",
        "$f(\mathbf{x}_t)$",
        "Method",
        capsize=0.1,
        dodge=True,
        palette="tab10",
        estimator=np.median,
        ci=95,
    )
    g.add_legend()
    save_fig("adult_action_position_probabilities")


def plot_full_sankeys():
    ## my method

    # german
    german_mapping = {
        0.0: "waitYears",
        1.0: "naturalize",
        2.0: "chCreditAm",
        3.0: "chLoanPeriod",
        4.0: "adjLoanPeriod",
        5.0: "guarantor",
        6.0: "unskilledJob",
        -1.0: "unused",
    }
    # adult
    adult_mapping = {
        0.0: "addEdu",
        1.0: "chCapLoss",
        2.0: "chWorkHrs",
        3.0: "enlist",
        4.0: "chCapGain",
        5.0: "waitYears",
        -1.0: "unused",
    }

    combinations = [
        (german_mapping, my_german, "german_EA_A"),
        # (german_mapping, my_other_german, "german_EA_B"),
        (adult_mapping, my_adult, "adult_EA_A"),
        (adult_mapping, my_other_adult, "adult_EA_B"),
    ]

    for mapping, _data, title in combinations:
        # labs = list(set(df[0]).union(df[1]).union(df[2]).union(df[3]).union(df[4]))
        all_labs = list(mapping.values()) + [None]
        all_labs = list(sorted(all_labs, key=lambda x: (x is None, x)))

        all_actions = np.row_stack([np.row_stack(record["pheno"]) for record in _data])
        all_actions = all_actions[:, : all_actions.shape[1] // 2]
        # my_ger_all_actions[my_ger_all_actions==-1] = np.nan

        df = pd.DataFrame(all_actions)

        df = df.replace(mapping)
        df[df == "unused"] = None
        df = df.dropna(axis="columns", how="all")

        # Specified the colors.
        # Here, we use 'Pastel1' colormaps(a shy but fresh palette:)).
        # See matplotlib cmap for more colormaps:
        # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        colors = setColorConf(len(all_labs), colors="tab20")
        hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]  # [::-1]
        hatches = [x * 2 for x in hatches]

        # colors = setColorConf(len(all_labs),colors='Accent')

        labs = set()
        for col in df.columns:
            labs = labs.union(df[col])
        labs = list(labs)
        # labs = list(set(df[0]).union(df[1]).union(df[2]).union(df[3]).union(df[4]))
        # cls_map = dict(zip(labs,colors))
        cls_map = {label: colors[all_labs.index(label)] for label in labs}
        hatch_map = {label: hatches[all_labs.index(label)] for label in labs}

        # set stripColor="left"
        # ! set iloc to max number of columns where it starts to be only nan after
        _sankey = Sankey(
            df.iloc[:, :],
            colorMode="global",
            stripColor="left",
            colorDict=cls_map,
            hatches=hatch_map,
        )

        si = 22
        fig, ax = _sankey.plot(
            figSize=(6, 3),  ## set the figure size
            fontSize=si + 4 if title == "german_EA_A" else si,  ## font size
            # fontPos=(0.05,0.5), ## font position relative to the box ,0.05: 5% to the right of each box,
            ## 0.5: 50% to the bottom of each box
            boxInterv=0,  # 0.01,    ## set zero gap between boxes
            kernelSize=25,  ## kernelSize determines the smoothness of the strip( default=25)
            bot_dist=30 if title.startswith("german") else 60,
            # stripShrink=0.15,  ## setting he curve shrink slightly
            # stripShrink=1.5,
            boxWidth=5 if title == "adult_EA_A" else 4,
            # boxWidth=10 if title == "adult_EA_A" else 4,
            # stripLen=100 if title == "adult_EA_A" else 10,
            # strip_kws={"alpha": 1.0}
        )  # text_kws={'size':20})
        # fig.text(0.14, 0.5, 'Frequencies of $a_i$ at position $t$ in $\mathcal{S}$', ha='center', va='center', rotation='vertical')
        # if title == "german_EA_A":
        #    ax.plot(np.zeros(1), np.zeros([1,3]), color='w', alpha=0, label=' ')
        if title == "german_EA_A":
            fig.text(
                0.13,
                0.5,
                "Frequencies",
                ha="center",
                va="center",
                rotation="vertical",
                size=si + 4,
            )
            ax.legend(ncol=1, prop={"size": 15.5}, labelspacing=0.05)
            # handles,labels = ax.get_legend_handles_labels()
            # ax.legend(ncol=1,prop={'size': 12})
        elif title == "adult_EA_B":
            fig.text(
                0.13,
                0.5,
                "Frequencies",
                ha="center",
                va="center",
                rotation="vertical",
                size=si + 2,
            )
            ax.legend(ncol=1, prop={"size": 13}, labelspacing=0.05)
        else:
            fig.text(
                0.13,
                0.5,
                "Frequencies",
                ha="center",
                va="center",
                rotation="vertical",
                size=si + 3,
            )
            ax.legend(prop={"size": 13}, labelspacing=0.05)
        save_fig(f"{title}", extension="pdf")


def plot_competitor_sankeys():
    ## competitor

    # german
    german_mapping = {
        0.0: "waitYears",
        1.0: "naturalize",
        2.0: "chCreditAm",
        3.0: "chLoanPeriod",
        4.0: "adjLoanPeriod",
        5.0: "guarantor",
        6.0: "unskilledJob",
        -1.0: "unused",
    }
    # adult
    adult_mapping = {
        0.0: "addEdu",
        1.0: "chCapLoss",
        2.0: "chWorkHrs",
        3.0: "enlist",
        4.0: "chCapGain",
        5.0: "waitYears",
        -1.0: "unused",
    }

    comp_mapping_ad = {
        "AddEducation": "addEdu",
        "IncreaseCapitalGain": "chCapGain",
        "ChangeWorkingHours": "chWorkHrs",
        "ChangeCapitalLoss": "chCapLoss",
        "Enlist": "enlist",
        "WaitYears": "waitYears",
    }
    comp_mapping_ger = {
        "AdjustLoanPeriod": "adjLoanPeriod",
        "ChangeCreditAmount": "chCreditAm",
        "ChangeLoanPeriod": "chLoanPeriod",
        "GetGuarantor": "guarantor",
        "Naturalize": "naturalize",
    }

    combinations = [
        (comp_mapping_ger, german_mapping, comp_german, "german_competitor"),
        (comp_mapping_ad, adult_mapping, comp_adult, "adult_competitor"),
    ]

    for mapping, main_mapping, _data, title in combinations:
        all_labs = list(main_mapping.values()) + [None]
        all_labs = list(sorted(all_labs, key=lambda x: (x is None, x)))

        all_atc = [record["output"]["best_result"]["sequence"] for record in _data]
        all_actions = np.empty((len(all_atc), 2), dtype="object")
        for i, row in enumerate(all_atc):
            for j, r in enumerate(row):
                all_actions[i, j] = r

        df = pd.DataFrame(all_actions)
        df = df.replace(mapping)
        # Specified the colors.
        # Here, we use 'Pastel1' colormaps(a shy but fresh palette:)).
        # See matplotlib cmap for more colormaps:
        # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        labs = list(set(df[0]).union(df[1]))  # .union(df[3]).union(df[4]))
        colors = setColorConf(len(all_labs), colors="tab20")
        hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]  # [::-1]
        hatches = [x * 1 for x in hatches]
        # cls_map = dict(zip(labs,colors))
        # cls_map = {l:c for l,c in zip(labs, colors)}
        cls_map = {label: colors[all_labs.index(label)] for label in labs}
        hatch_map = {label: hatches[all_labs.index(label)] for label in labs}

        # set stripColor="left"
        # ! set iloc to max number of columns where it starts to be only nan after
        _sankey = Sankey(
            df.iloc[:, :],
            colorMode="global",
            stripColor="left",
            colorDict=cls_map,
            hatches=hatch_map,
        )

        # set a bigger font size
        si = 26
        fig, ax = _sankey.plot(
            figSize=(6, 3),  ## set the figure size
            fontSize=si,  ## font size
            # fontPos=(0.05,0.5), ## font position relative to the box ,0.05: 5% to the right of each box,
            ## 0.5: 50% to the bottom of each box
            boxInterv=0,  # 0.01,    ## set zero gap between boxes
            kernelSize=25,  ## kernelSize determines the smoothness of the strip( default=25)
            bot_dist=10,
            # stripShrink=0.15,  ## setting he curve shrink slightly
            # stripShrink=1.5,
            # boxWidth=1.5,
            # stripLen=10,
            # strip_kws={"alpha": 1.0}
        )  # text_kws={'size':20})
        fig.text(
            0.13,
            0.5,
            "Frequencies",
            ha="center",
            va="center",
            rotation="vertical",
            size=si,
        )

        # plt.xlabel('Generation', fontsize=si)
        # plt.ylabel('IGD (Median + 25-75\%)', fontsize=si)
        if title == "adult_competitor":
            plt.legend(ncol=1, prop={"size": 11}, labelspacing=0.05)
        else:
            plt.legend(prop={"size": 12}, labelspacing=0.05)
        save_fig(f"{title}")


def plot_shorter_than_two_sankeys():
    ## my method

    (
        my_c_german,
        my_idx_german,
        comp_c_german,
        my_ot_c_german,
        my_ot_idx_german,
        german_is_not_nan,
    ) = get_summed_costs(
        my_german, comp_german, my_other_german, max_length=2, which="summed_costs"
    )
    (
        my_c_adult,
        my_idx_adult,
        comp_c_adult,
        my_ot_c_adult,
        my_ot_idx_adult,
        adult_is_not_nan,
    ) = get_summed_costs(
        my_adult, comp_adult, my_other_adult, max_length=2, which="summed_costs"
    )

    # german
    german_mapping = {
        0.0: "waitYears",
        1.0: "naturalize",
        2.0: "chCreditAm",
        3.0: "chLoanPeriod",
        4.0: "adjLoanPeriod",
        5.0: "guarantor",
        6.0: "unskilledJob",
        -1.0: "unused",
    }
    # adult
    adult_mapping = {
        0.0: "addEdu",
        1.0: "chCapLoss",
        2.0: "chWorkHrs",
        3.0: "enlist",
        4.0: "chCapGain",
        5.0: "waitYears",
        -1.0: "unused",
    }

    combinations = [
        (german_mapping, my_german, "german_EA_A"),
        # (german_mapping, my_other_german, "german_EA_B"),
        (adult_mapping, my_adult, "adult_EA_A"),
        (adult_mapping, my_other_adult, "adult_EA_B"),
    ]

    for mapping, _data, title in combinations:
        # labs = list(set(df[0]).union(df[1]).union(df[2]).union(df[3]).union(df[4]))
        all_labs = list(mapping.values()) + [None]
        all_labs = list(sorted(all_labs, key=lambda x: (x is None, x)))

        all_actions = np.row_stack([np.row_stack(record["pheno"]) for record in _data])
        all_actions = all_actions[:, : all_actions.shape[1] // 2]
        ac_len = np.array([sum(xx != -1) for xx in all_actions])
        all_actions = all_actions[ac_len <= 2, :]

        # my_ger_all_actions[my_ger_all_actions==-1] = np.nan

        df = pd.DataFrame(all_actions)

        df = df.replace(mapping)
        df[df == "unused"] = None
        df = df.dropna(axis="columns", how="all")

        # Specified the colors.
        # Here, we use 'Pastel1' colormaps(a shy but fresh palette:)).
        # See matplotlib cmap for more colormaps:
        # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        colors = setColorConf(len(all_labs), colors="tab20")
        labs = set()
        hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]  # [::-1]
        hatches = [x * 1 for x in hatches]
        for col in df.columns:
            labs = labs.union(df[col])
        labs = list(labs)
        # labs = list(set(df[0]).union(df[1]).union(df[2]).union(df[3]).union(df[4]))
        # cls_map = dict(zip(labs,colors))
        cls_map = {label: colors[all_labs.index(label)] for label in labs}
        hatch_map = {label: hatches[all_labs.index(label)] for label in labs}

        # set stripColor="left"
        # ! set iloc to max number of columns where it starts to be only nan after
        _sankey = Sankey(
            df.iloc[:, :],
            colorMode="global",
            stripColor="left",
            colorDict=cls_map,
            hatches=hatch_map,
        )

        # set a bigger font size
        si = 26
        fig, ax = _sankey.plot(
            figSize=(6, 3),  ## set the figure size
            fontSize=si,  ## font size
            # fontPos=(0.05,0.5), ## font position relative to the box ,0.05: 5% to the right of each box,
            ## 0.5: 50% to the bottom of each box
            boxInterv=0,  # 0.01,    ## set zero gap between boxes
            kernelSize=25,  ## kernelSize determines the smoothness of the strip( default=25)
            bot_dist=45,
            # stripShrink=0.15,  ## setting he curve shrink slightly
            # stripShrink=1.5,
            # boxWidth=1.5,
            # stripLen=10,
            # strip_kws={"alpha": 1.0}
        )  # text_kws={'size':20})
        fig.text(
            0.13,
            0.5,
            "Frequencies",
            ha="center",
            va="center",
            rotation="vertical",
            size=si,
        )
        # fig.text(0.14, 0.5, 'Frequencies of $a_i$ at position $t$ in $\mathcal{S}$', ha='center', va='center', rotation='vertical')
        if title == "german_EA_A":
            plt.legend(ncol=1, prop={"size": 11.0}, labelspacing=0.05)
        else:
            plt.legend(ncol=2, prop={"size": 9.5}, labelspacing=0.05)
        save_fig(f"{title}_len=2_all", extension="pdf")


my_german, comp_german, my_other_german = get_data("german")
my_adult, comp_adult, my_other_adult = get_data("adult")

print("Plotting cost comparison")
plot_costs_comparison()
print("Plotting action position probabilities")
plot_positional_action_probabilities()
print("Plot full sankeys")
plot_full_sankeys()
print("Plot competitor sankeys")
plot_competitor_sankeys()
print("Plot t<=2 sankeys")
plot_shorter_than_two_sankeys()