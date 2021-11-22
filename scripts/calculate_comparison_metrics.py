import hydra
from omegaconf import DictConfig
import h5py
import os
from tqdm import tqdm
import torch
from tsgrasp.utils.metric_utils.metrics import framewise_sc_curve, precision_recall_curve

import pandas as pd

@hydra.main(config_path="../conf", config_name="scripts/calculate_comparison_metrics")
def main(cfg : DictConfig):
    with h5py.File(cfg.tsgrasp_output_path, 'r') as ts_ds:
        with h5py.File(cfg.ctn_output_path, 'r') as ctn_ds:
            s_c_curves = make_s_c_curves(ts_ds, ctn_ds, cfg)

def make_s_c_curves(ts_ds, ctn_ds, cfg):

    ts_curves = []
    ctn_curves = []
    ts_pr_curves = []
    ctn_pr_curves = []
    i = 0
    for example_num in tqdm(ts_ds['outputs']):
        i+=1
        if i > 10:
            break

        ts_results = ts_ds['outputs'][example_num]
        ctn_results = ctn_ds['outputs'][example_num]

        # Load relevant confidence and position information from each model
        ts_confs = torch.sigmoid(torch.Tensor(ts_results['class_logits']))
        ts_pos = torch.Tensor(ts_results['positions'])
        ts_gt_contact_pts = torch.Tensor(ts_results['gt_contact_pts'])
        ts_labels = torch.Tensor(ts_results['pt_labels'])

        ctn_confs = torch.sigmoid(torch.Tensor(ctn_results['class_logits']))
        ctn_pos = torch.Tensor(ctn_results['positions'])
        ctn_gt_contact_pts = torch.Tensor(ctn_results['gt_contact_pts'])
        ctn_labels = torch.Tensor(ctn_results['pt_labels'])

        ## Make success-coverage curves
        ts_curves.append(
            framewise_sc_curve(ts_confs, ts_pos, ts_labels, ts_gt_contact_pts)
        )
        ctn_curves.append(
            framewise_sc_curve(ctn_confs, ctn_pos, ctn_labels, ctn_gt_contact_pts)
        )

        ## Make precision-recall curves
        ts_pr_curves.append(
            precision_recall_curve(ts_confs, ts_labels)
        )
        ctn_pr_curves.append(
            precision_recall_curve(ts_confs, ts_labels)
        )

    ## Concatenate the outputs from every example
    # Success-coverage curves
    ts_curves = pd.concat(ts_curves, keys=range(len(ts_curves)))
    ts_curve = ts_curves.groupby('confidence').mean()

    ctn_curves = pd.concat(ctn_curves, keys=range(len(ctn_curves)))
    ctn_curve = ctn_curves.groupby('confidence').mean()

    # Precision-recall curves
    ts_pr_curves = pd.concat(ts_pr_curves, keys=range(len(ts_pr_curves)))
    ts_pr_curve = ts_pr_curves.groupby('confidence').mean()

    ctn_pr_curves = pd.concat(ctn_pr_curves, keys=range(len(ctn_pr_curves)))
    ctn_pr_curve = ctn_pr_curves.groupby('confidence').mean()

    ## Write dataframes to disk
    # Success-coverage curves
    ts_curves.to_csv(cfg.tsgrasp_sc_csv_path)
    ctn_curves.to_csv(cfg.ctn_sc_csv_path)

    # Precision-recall curves
    ts_pr_curves.to_csv(cfg.tsgrasp_pr_csv_path)
    ctn_curves.to_csv(cfg.ctn_pr_csv_path)

    ## Plot curves
    from tsgrasp.utils.metric_utils.metrics import plot_sc_curve, plot_pr_curve
    import matplotlib.pyplot as plt

    # Success-coverage curves
    fig, ax = plt.subplots()
    plot_sc_curve(ts_curve, ax=ax, label='tsgraspnet')
    plot_sc_curve(ctn_curve, ax=ax, label='CTN')
    fig.legend()
    plt.savefig(cfg.sc_png_path)

    # Precision-recall curves
    fig, ax = plt.subplots()
    plot_pr_curve(ts_pr_curve, ax=ax, label='tsgraspnet')
    plot_pr_curve(ctn_pr_curve, ax=ax, label='CTN')
    fig.legend()

    plt.savefig(cfg.pr_png_path)


if __name__ == "__main__":
    main()