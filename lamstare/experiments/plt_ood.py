from functools import lru_cache
import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas
import yaml
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from pandas import DataFrame

from lamstare.experiments.plt_test import COLOR, fetch_dptest_res, parse_record_dict_to_df
from lamstare.infra.ood_database import OODRecord
from lamstare.utils.plot import sendimg


with open(os.path.dirname(__file__) + "/../release/ood_test/OOD_DATASET_v2.yml", "r") as f:
    OOD_DATASET = yaml.load(f, Loader=yaml.FullLoader)
OOD_DATASET = (
    DataFrame(OOD_DATASET["OOD_TO_HEAD_MAP"]).T.rename_axis("Dataset").infer_objects()
)
# normalize weights
for index in OOD_DATASET.keys():
    if "weight" in index:
        efv_weight = OOD_DATASET[index]
        efv_weight /= efv_weight.mean()
print(OOD_DATASET)

OOD_DATASET_STD = pandas.read_csv(
    "/mnt/data_nas/public/multitask/LAMstare/lamstare/release/ood_test/ood_data_std.csv"
).infer_objects()
OOD_DATASET_STD.set_index("Dataset", inplace=True)
print(OOD_DATASET_STD)

@lru_cache
def get_weighted_result(exp_path: str) -> DataFrame:
    run_id = exp_path.split("/")[-1]  # Get basename as id
    all_records = fetch_dptest_res(run_id, OODRecord)
    all_records_df = parse_record_dict_to_df(all_records)
    all_records_df_raw = all_records_df.copy()
    # print(all_records_df)

    # Remove records with zero weights
    all_records_df.mask(all_records_df.isna(), inplace=True)

    weighted_avg = all_records_df.groupby(
        "Training Steps"
    ).mean().map(lambda x: np.nan)  # provide a df with same shape

    # mask.inplace and update() won't work; need to assign to a new variable
    for efv in ["energy", "force", "virial"]:
        data = all_records_df.loc[
            :, [key for key in all_records_df.keys() if efv in key]
        ]
        weights = OOD_DATASET[efv + "_weight"]
        weighted_avg_efv = (
            data.apply(np.log)
            .mul(weights, axis="index")
            .groupby("Training Steps")
            .mean()
            .apply(np.exp)
        )
        # mask out the results where NAN exists in the original data
        weighted_avg_efv.mask(all_records_df["energy_mae"].isna().groupby("Training Steps").any(), inplace=True)
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
    metric_key: str="rmse"
):
    for dataset, records in all_records_df.groupby("Dataset"):
        # removing two ood test sets
        if dataset in ["Sub_Alex_val", "raw_torsionnet500", "HEA25", "HEMC_HEMB", "WBM", "ANI", "MD22"]:
            continue
        assert dataset in dataset_to_subplot.keys(), f"Dataset {dataset} not presented"
        subplot = dataset_to_subplot[dataset]  # type: ignore
        # print(dataset)
        records = records.droplevel("Dataset")
        # print(records)
        subplot[0].set_ylabel(f"{dataset}")
        for efv, suffix, subsubplot in zip(
            ["energy", "force", "virial"], ["_natoms", "", "_natoms"], subplot
        ):
            if dataset in OOD_DATASET_STD.index:
                std: numpy.float = OOD_DATASET_STD.loc[dataset, f"{efv}_std"]  # type: ignore
                if efv == "virial" and dataset != "Weighted" and np.isnan(std):
                    break # Careful! virial should be the last element in for loop
                subsubplot.axhline(std, color="purple", linestyle="-.")
                # note: this will draw duplicated lines

            metric_name = efv + f"_{metric_key}" + suffix
            line = subsubplot.loglog(
                records.index,  # step
                records[metric_name],
                "o-",
                label=dataset,
                color=color,
                alpha=0.8,
            )
    legend_handles.extend(line)  # type: ignore


def main(exps: list[str], metric_key: str="rmse"):
    # Get dataset list from yaml file to preserve the order
    datasets: list[str] = OOD_DATASET.index.tolist()
    datasets.append("Weighted")
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
        plotting(dataset_to_subplot, all_records_df, color, legend_handles, metric_key)

    ## to set finer tick
    from matplotlib.ticker import FixedLocator
    ax[-1][0].yaxis.set_major_locator(FixedLocator(np.arange(0.02, 0.04, 0.002)))
    ax[-1][1].yaxis.set_major_locator(FixedLocator(np.arange(0.2, 0.5, 0.04)))
    ## to handle hpt explosion
    # for ax in dataset_to_subplot["HPt_NC_2022"]:
    #     ax.set_ylim(0.05,0.2)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.975)
    title = f"Compare OOD-{metric_key}"
    # fig.suptitle(title) # Poor placement
    fig.legend(
        handles=legend_handles,
        labels=[exp_path.split("/")[-1] for exp_path in exps],
        loc="upper left",
        frameon=False,
    )
    filename = title + ".jpg"
    fig.savefig(filename)
    print(f"Saved to {filename}")
    sendimg([filename], title)


if __name__ == "__main__":
    exps = [
        # "/aisi/public/multitask/training_exps/250701_dpa3_openlam_v2_old_weight_8M_l16",
        # "/aisi/public/multitask/training_exps/250701_dpa3_openlam_v2_new_weight_8M_l16",
        "/aisi/public/multitask/training_exps/dpa3.1-3m",
        # "/aisi/public/multitask/training_exps/250714_dpa3_openlam_v2_old_weight_8M_L16_only_change_omol",
        # "/aisi/public/multitask/training_exps/250714_dpa3_openlam_v2_old_weight_22task",
        # "/aisi/public/multitask/training_exps/250630_dpa3_openlam_v1_old_weight_8M_compare0415",
        # "/aisi/public/multitask/training_exps/250703_dpa3_openlam_v2_old_weight_8M_L16_only_change_omol",
        "/aisi/public/multitask/training_exps/250722_dpa3_openlam_v2_old_weight_22task_deepcsp_mpgen",
        "/aisi-nas/public/training_experiments/250728_dpa3_l24_lr1e-3_1e-6_pref0.2_20_100_60_openlam_v2_GPU64_H20_filter128"
        # "/aisi/public/multitask/training_exps/250707_dpa3_openlam_v2_8M_L16_omol_2nd_fitting",
        # "/aisi/public/multitask/training_exps/250708_dpa3_openlam_v2_8M_L16_omol_2fting_with_default_fparam",
        # "/aisi/public/multitask/training_exps/250710_dpa3_openlam_v2_old_weight_8M_L16_remove_mptrj",
    ]
    main(exps)
    # main(exps, "mae")
