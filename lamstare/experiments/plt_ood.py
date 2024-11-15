from functools import lru_cache
import logging
from typing import Union

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import pandas
import yaml
from lamstare.experiments.plt_test import fetch_dptest_res
from lamstare.infra import Record
from lamstare.infra.ood_database import OODRecord
from lamstare.utils.plot import sendimg
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

COLOR = [
    "red",
    "yellowgreen",
    "dodgerblue",
    "mediumpurple",
    "orange",
    "lightsalmon",
    "hotpink",
    "tan",
    "cyan",
    "navy",
    "orchid",
]

with open(
    "/mnt/workspace/public/multitask/LAMstare/lamstare/release/new_OOD_DATASET.yml", "r"
) as f:
    OOD_DATASET = yaml.load(f, Loader=yaml.FullLoader)
OOD_DATASET = (
    DataFrame(OOD_DATASET["OOD_TO_HEAD_MAP"])
    .T.reset_index(names="Dataset")
    .set_index("Dataset")
    .infer_objects()
)
print(OOD_DATASET)

OOD_DATASET_STD = pandas.read_csv(
    "/mnt/workspace/cc/LAMstare_new/lamstare/release/ood_data_std.csv"
)
# Dataset, energy_std, force_std, virial_std
OOD_DATASET_STD.set_index("Dataset", inplace=True)
OOD_DATASET_STD = OOD_DATASET_STD.infer_objects()


def parse_record_dict_to_df(data: dict) -> DataFrame:
    # Convert dictionary to a list of tuples
    data_tuples = [
        (dataset, training_steps, metrics)
        for dataset, training_steps_dict in data.items()
        for training_steps, metrics in training_steps_dict.items()
    ]

    # Extract the first metrics dictionary to get the keys
    columns = ["Dataset", "Training Steps"] + list(data_tuples[0][2].keys())

    # Create DataFrame
    df = DataFrame(
        [
            (dataset, training_steps, *metrics.values())
            for dataset, training_steps, metrics in data_tuples
        ],
        columns=columns,
    )

    # Set multi-index
    df.set_index(["Dataset", "Training Steps"], inplace=True)
    # sort by Dataset
    df.sort_index(level=0, inplace=True)
    return df

@lru_cache
def get_weighted_result(exp_path: str) -> DataFrame:
    run_id = exp_path.split("/")[-1]  # Get basename as id
    all_records = fetch_dptest_res(run_id, OODRecord)
    all_records_df = parse_record_dict_to_df(all_records)
    all_records_df_raw = all_records_df.copy()
    # print(all_records_df)

    # Remove records with zero weights
    all_records_df.mask(all_records_df.isna(), inplace=True) # FIXME: need teests

    weighted_avg = all_records_df.groupby("Training Steps").mean() # provide a baseline with same shape
    # mask.inplace and update() won't work; need to assign to a new variable
    for efv in ["energy", "force", "virial"]:
        data = all_records_df.loc[:, [key for key in all_records_df.keys() if efv in key]]
        weights=OOD_DATASET[efv+"_weight"]
        data.mask(weights==0, inplace=True)
        weighted_avg_efv = (
            data.apply(np.log)
            .mul(weights, axis="index")
            .groupby("Training Steps")
            .mean()
            .apply(np.exp)
        )
        weighted_avg.update(weighted_avg_efv)

    weighted_avg["Dataset"] = "Weighted"
    weighted_avg.reset_index(inplace=True)
    weighted_avg.set_index(["Dataset", "Training Steps"], inplace=True)
    all_records_df = pandas.concat(
        [all_records_df_raw, weighted_avg]
    )  # Preserve masked values
    print(all_records_df)
    return all_records_df


def plotting(
    dataset_to_subplot: dict[str, list[Axes]],
    all_records_df: DataFrame,
    color: str,
    legend_handles: list[Line2D],
):
    for dataset, records in all_records_df.groupby("Dataset"):
        assert dataset in dataset_to_subplot.keys(), f"Dataset {dataset} not presented"
        subplot=dataset_to_subplot[dataset] # type: ignore
        # print(dataset)
        records = records.droplevel("Dataset")
        # print(records)
        subplot[0].set_ylabel(f"{dataset}")
        for efv, suffix, subsubplot in zip(
            ["energy", "force", "virial"], ["_natoms", "", "_natoms"], subplot
        ):
            metric_name = efv + "_rmse" + suffix
            line = subsubplot.loglog(
                records.index,  # step
                records[metric_name],
                "o-",
                label=dataset,
                color=color,
                alpha=0.8,
            )
            if dataset in OOD_DATASET_STD.index:
                subsubplot.axhline(OOD_DATASET_STD.loc[dataset, f"{efv}_std"], color="purple", linestyle="-.")  # type: ignore
            # FIXME: this will draw duplicated lines
    legend_handles.extend(line)  # type: ignore


def main(exps: list[str]):
    # Get all datasets
    datasets: list[str] = sorted(
        set(
            dataset
            for exp in exps
            for dataset in get_weighted_result(exp)
            .index.get_level_values("Dataset")
            .unique()
            .tolist()
        )
    )
    datasets.remove("Weighted") # Assuming it exists
    datasets.append("Weighted") # Move to the end
    print(datasets)

    fig, ax = plt.subplots(
        len(datasets), 3, figsize=(12, 3 * len(datasets)), sharex=True
    )
    ax: list[list[Axes]]
    legend_handles: list[Line2D] = []
    # get axis by dataset name to prevent plotting on wrong axis
    dataset_to_subplot = dict(zip(datasets, ax))
    # add energy/force/virial to the beginning of plots
    for axis, efv in zip(ax[0], ["energy", "force", "virial"]):
        axis.set_title(efv)

    for exp_path, color in zip(exps, COLOR):
        all_records_df = get_weighted_result(exp_path)
        plotting(dataset_to_subplot, all_records_df, color, legend_handles)

    fig.tight_layout()
    fig.subplots_adjust(top=0.975)
    title = "Compare OOD"
    # fig.suptitle(title) # Poor placement
    fig.legend(
        handles=legend_handles,
        labels=[exp_path.split("/")[-1] for exp_path in exps],
        loc="upper left",
        frameon=False,
    )
    filename = title + ".jpg"
    fig.savefig(filename, dpi=300)
    print(f"Saved to {filename}")
    sendimg([filename], title)


if __name__ == "__main__":
    exps = [
        # "/mnt/data_nas/public/multitask/training_exps/1107_shareft_pref0021_1000100_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1110_newdata_shareft_240by6_medium_l6_atton_37head_tanh_40GPU",
        # "/mnt/data_nas/public/multitask/training_exps/1110_newdata_shareft_pref0021_1000100_medium_l6_atton_37head_tanh_40GPU",
        "/mnt/data_nas/public/multitask/training_exps/1110_newdata_sharft_lr1e-3_1e-5_medium_l6_atton_37head_tanh_40GPU",
        "/mnt/data_nas/public/multitask/training_exps/1113_shareft_960by3_lr1e-3_1e-5_medium_l6_atton_37head_tanh_40GPU",
        "/mnt/data_nas/public/multitask/training_exps/1113_shareft_lr1e-3_1e-5_pref0220_10020_medium_l6_atton_37head_tanh_40GPU"
    ]
    main(exps)
