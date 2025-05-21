from __future__ import print_function
import re
import os
import argparse
import random
import torch
import json
import numpy as np

from dassl.utils import set_random_seed, collect_env_info
from dassl.engine import build_trainer

import trainers.coop
import trainers.cocoop

from eval_utils import print_args, setup_cfg, get_parsed_args


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def update_and_save_evaluation(
    model_name,
    dataset_name,
    accuracy,
    f1_score,
    average_precision,
    output_path,
    model_evaluations,
):
    # Check if the model key exists in the dictionary
    if model_name not in model_evaluations:
        model_evaluations[model_name] = {}
    # Check if the dataset key exists for the given model
    if dataset_name not in model_evaluations[model_name]:
        model_evaluations[model_name][dataset_name] = {}
    # Update evaluation results
    model_evaluations[model_name][dataset_name]["accuracy"] = accuracy
    model_evaluations[model_name][dataset_name]["f1_score"] = f1_score
    model_evaluations[model_name][dataset_name]["average_precision"] = average_precision

    output_path = output_path.replace("\\", "/")
    model_name = model_name.replace("\\", "/")

    if "context" in model_name:
        save_file_name = output_path + model_name.split("/")[1] + ".json"
    elif "finetuned" in model_name:
        save_file_name = output_path + model_name.split("/")[1] + ".json"
    elif "clipadapter" in model_name:
        save_file_name = output_path + model_name.split("/")[1] + ".json"
    else:
        save_file_name = (
            output_path + "/" + model_name.split("/")[-1].split(".")[0] + ".json"
        )

    # Save the updated dictionary to a JSON file
    with open(save_file_name, "w") as json_file:
        json.dump(model_evaluations, json_file, indent=2)


def dummy_parse_args():
    return


def eval_prompt_tuning(args, dataset_path, dataset_names, image_extensions, device):
    print("*************")
    print("Evaluating Prompt Tuning Method!")
    weights_dir = "/home/john/Dev/university/weights/selected_coop_models"

    if "100k_16" in args.model:
        model_names = [f"{weights_dir}/100000_16context_best_until_now/"]
    elif "100k_8" in args.model:
        model_names = [f"{weights_dir}/100000_8context/"]
    elif "100k_4" in args.model:
        model_names = [f"{weights_dir}/100000_4context/"]

    model_evaluations = {}
    splitted_string = model_names[0].split("/")[-2].split("_")[1]
    num_ctx_tokens = int(re.split("(\d+)", splitted_string)[1])
    print("Num. Context Tokens: ", num_ctx_tokens)
    args.parser = dummy_parse_args()
    for dataset in dataset_names:
        coop_args = get_parsed_args(
            model_names[0], dataset, num_ctx_tokens, dataset_path
        )
        cfg = setup_cfg(coop_args)
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
        if torch.cuda.is_available() and cfg.USE_CUDA:
            print("Using CUDA!!!")
            torch.backends.cudnn.benchmark = True

        print_args(coop_args, cfg)
        print("Collecting env info ...")
        print("** System info **\n{}\n".format(collect_env_info()))

        trainer = build_trainer(cfg)
        trainer.load_model(coop_args.model_dir, epoch=coop_args.load_epoch)

        results, results_dict = trainer.test()
        update_and_save_evaluation(
            model_names[0],
            dataset,
            results_dict["accuracy"],
            results_dict["macro_f1"],
            results_dict["average_precision"],
            args.output,
            model_evaluations,
        )


def main(args):
    print("Starting Evaluation!")

    seed = 17
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.tif"]
    dataset_path = args.dataset.replace("\\", "/")
    print("Dataset path: " + dataset_path)
    print("Output path: " + args.output)

    dataset_names = [
        "collab_diff",
        "midjourney",
        "stargan",
        "starganv2",
        "styleclip",
        "whichfaceisreal",
    ]

    if args.variant == "promptTuning":
        eval_prompt_tuning(args, dataset_path, dataset_names, image_extensions, device)
    else:
        print("Unrecognized method!!!")

    print("Evaluation completed!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        default="linearProbing",
        choices=["linearProbing", "promptTuning", "fineTuning", "adapterNetwork"],
        help="name of the adaptation method",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["100k", "100k_16", "100k_8", "100k_4"],
        default="100k",
        help="name of linear probing model to evaluate",
    )
    parser.add_argument("--dataset", type=str, default="", help="path to dataset")
    parser.add_argument(
        "--output", type=str, default="", help="output directory to write results"
    )

    args = parser.parse_args()
    main(args)
