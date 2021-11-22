import hydra
from omegaconf import DictConfig
import h5py
import os
from tqdm import tqdm
import torch
from tsgrasp.utils.metric_utils.metrics import success_coverage_curve
import pandas as pd

@hydra.main(config_path="../conf", config_name="scripts/calculate_comparison_metrics")
def main(cfg : DictConfig):
    with h5py.File(cfg.tsgrasp_output_path, 'r') as ts_ds:
        with h5py.File(cfg.ctn_output_path, 'r') as ctn_ds:
            s_c_curves = make_s_c_curves(ts_ds, ctn_ds, cfg)

def make_s_c_curves(ts_ds, ctn_ds, cfg):

    ts_curves = []
    ctn_curves = []
    sc_ctn_curves = []
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

        ts_curves.append(
            framewise_sc_curve(ts_confs, ts_pos, ts_labels, ts_gt_contact_pts)
        )
        ctn_curves.append(
            framewise_sc_curve(ctn_confs, ctn_pos, ctn_labels, ctn_gt_contact_pts)
        )

    ts_curves = pd.concat(ts_curves, keys=range(len(ts_curves)))
    ts_curve = ts_curves.groupby('confidence').mean()

    ctn_curves = pd.concat(ctn_curves, keys=range(len(ctn_curves)))
    ctn_curve = ctn_curves.groupby('confidence').mean()

    ts_curves.to_csv(cfg.tsgrasp_csv_path)
    ctn_curves.to_csv(cfg.ctn_csv_path)

    from tsgrasp.utils.metric_utils.metrics import plot_s_c_curve
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plot_s_c_curve(ts_curve, ax=ax, label='tsgraspnet')
    plot_s_c_curve(ctn_curve, label='CTN')

    plt.savefig(cfg.png_path)

def framewise_sc_curve(confs, pred_grasp_locs, labels, gt_contact_pts):
    """Make a success-coverage curve from a sequence of time series data."""
    curves = []
    for t in range(len(confs)):
        curves.append(success_coverage_curve(
            confs[t], pred_grasp_locs[t], labels[t], gt_contact_pts[t]
        ))
    return pd.concat(curves).groupby('confidence').mean()
    

    

if __name__ == "__main__":
    main()