#!/usr/bin/env python3
# pyre-strict
import argparse
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

# Add parent directory to sys.path so opensfm can be imported
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from opensfm import context, dataset, log, matching, pairs_selection

logger: logging.Logger = logging.getLogger("train_dif_projection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline training of DIF binary projection."
    )
    parser.add_argument(
        "json_file",
        help="Path to JSON file listing dataset folders and their associated K.",
    )
    parser.add_argument(
        "--default-k",
        type=int,
        default=5,
        help="Default K (number of pairs) if not specified in JSON.",
    )
    return parser.parse_args()


def load_datasets_from_json(json_file: str, default_k: int) -> List[Tuple[str, int]]:
    with open(json_file, "r") as f:
        config_data = json.load(f)

    datasets_to_process = []
    if isinstance(config_data, list):
        for entry in config_data:
            if isinstance(entry, dict) and "path" in entry:
                datasets_to_process.append(
                    (entry["path"], entry.get("k", default_k)))
            elif isinstance(entry, str):
                datasets_to_process.append((entry, default_k))
    elif isinstance(config_data, dict):
        k_val = config_data.get("k", default_k)
        datasets = config_data.get("datasets", [])
        if isinstance(datasets, list):
            for d in datasets:
                if isinstance(d, dict) and "path" in d:
                    datasets_to_process.append((d["path"], d.get("k", k_val)))
                elif isinstance(d, str):
                    datasets_to_process.append((d, k_val))
    return datasets_to_process


def main() -> None:
    args = parse_args()
    log.setup()

    datasets_to_process = load_datasets_from_json(
        args.json_file, args.default_k)
    logger.info(f"Loaded {len(datasets_to_process)} datasets to process.")

    processed_datasets = []
    all_pos_d1 = []
    all_pos_d2 = []
    all_neg_d1 = []
    all_neg_d2 = []
    feature_types = []

    for path, k in datasets_to_process:
        logger.info(f"Processing dataset: {path} with K={k}")
        if not os.path.exists(path):
            logger.error(f"Dataset path does not exist: {path}")
            continue

        try:
            data = dataset.DataSet(path)
            images = data.images()
            if not images:
                logger.warning(f"No images found in dataset: {path}")
                continue

            exifs = {im: data.load_exif(im) for im in images}
            cameras = data.load_camera_models()

            pairs, preport = pairs_selection.match_candidates_from_metadata(
                images,
                images,
                exifs,
                data,
                data.config,
            )
            if not pairs:
                logger.warning(f"No candidate pairs found in dataset: {path}")
                continue

            n_sample = min(k, len(pairs))
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(len(pairs), n_sample, replace=False)
            sample_pairs = [pairs[i] for i in sample_idx]

            # Force FLANN matching on CPU for training pairs
            training_config = dict(data.config)
            training_config["use_opencl_matching"] = False
            training_config["matcher_type"] = "FLANN"
            training_config["use_robust_matching"] = True

            training_args = list(
                matching.match_arguments(
                    sample_pairs, data, training_config, cameras, exifs, None, None
                )
            )
            processes = data.config.get("processes", 1)
            processes = context.processes_that_fit_in_memory(processes, 512)

            logger.info(
                f"Matching {len(sample_pairs)} training pairs with FLANN ({processes} processes)"
            )
            training_matches = context.parallel_map(
                matching.match_unwrap_args, training_args, processes, 2
            )

            pos_d1, pos_d2, neg_d1, neg_d2 = matching.collect_training_pairs(
                data,
                training_matches,
                training_config,
            )

            if len(pos_d1) == 0:
                logger.warning(
                    f"No positive pairs collected from dataset {path}")
                continue

            all_pos_d1.append(pos_d1)
            all_pos_d2.append(pos_d2)
            all_neg_d1.append(neg_d1)
            all_neg_d2.append(neg_d2)
            feature_types.append(data.config.get("feature_type", "HAHOG"))
            processed_datasets.append(path)
            logger.info(f"Successfully collected training pairs for {path}")

        except Exception as e:
            logger.exception(
                f"Exception raised while processing dataset {path}: {e}")

    if not all_pos_d1:
        logger.error(
            "No training data was collected from any of the datasets.")
        sys.exit(1)

    logger.info("Concatenating descriptors from all datasets...")
    pos_d1_total = np.concatenate(all_pos_d1, axis=0)
    pos_d2_total = np.concatenate(all_pos_d2, axis=0)
    neg_d1_total = np.concatenate(all_neg_d1, axis=0)
    neg_d2_total = np.concatenate(all_neg_d2, axis=0)

    # Check for feature type consistency
    if len(set(feature_types)) > 1:
        logger.warning(
            f"Multiple feature types detected: {list(set(feature_types))}. Using {feature_types[0]}."
        )
    descriptor_used = feature_types[0] if feature_types else "HAHOG"

    # Train DIF projection
    logger.info("Training joint DIF binary projection...")
    P, t = matching.train_dif_projection(
        pos_d1_total, pos_d2_total, neg_d1_total, neg_d2_total, clear_features_cache=True
    )

    # Save P and t
    dif_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "opensfm", "data", "dif")
    )
    os.makedirs(dif_dir, exist_ok=True)

    dataset_count = len(processed_datasets)
    k_values = [entry[1]
                for entry in datasets_to_process if entry[0] in processed_datasets]
    if len(set(k_values)) == 1:
        K_filename = k_values[0]
    else:
        K_filename = k_values[0] if k_values else args.default_k

    filename = f"{dataset_count}_{K_filename}_{descriptor_used}.npz"
    filepath = os.path.join(dif_dir, filename)

    logger.info(f"Saving P and t under {filepath}")
    np.savez(filepath, P=P, t=t)
    logger.info("Offline training completed successfully.")


if __name__ == "__main__":
    main()
