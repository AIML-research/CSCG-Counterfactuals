import argparse
import numpy as np
import evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")

    args = parser.parse_args()
    dataset_name = args.dataset
    assert (
        dataset_name == "adult" or dataset_name == "german"
    ), f"{dataset_name} is not supported"

    seed = 1
    np.random.seed(seed)

    if dataset_name == "adult":
        target_class = 1
    elif dataset_name == "german":
        target_class = 0
    else:
        raise Exception(f"{dataset_name} not supported")

    dataset, legacy_actions, legacy_features = evaluation.setup_dataset(
        dataset_name, seed
    )
    model = evaluation.setup_model(dataset, dataset_name, "comp-NN", 100, seed)

    (
        initial_instances_orig,
        initial_instances_proc,
        instances_random_idx,
    ) = evaluation.setup_experiment_instances(
        dataset, model, target_class, n_experiment_instances=100, seed=seed
    )
    # write the initial instances to a file for later usage
    evaluation.write_instances_to_file(
        initial_instances_orig,
        dataset_name,
        target_class,
        "",
        prefix="original",
    )
    evaluation.write_instances_to_file(
        initial_instances_proc,
        dataset_name,
        target_class,
        "",
        prefix="processed",
    )