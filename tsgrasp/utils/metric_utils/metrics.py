import torch
import pandas as pd
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

class SCCurve(Callback):
    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        ## Run forward inference on this batch
        positions = batch['positions']
        contact_pts = batch['pos_contact_pts_cam']

        B, T, N_PTS, D = positions.shape

        pts = positions.to(pl_module.device)
        (class_logits, baseline_dir, approach_dir, grasp_offset, pts 
        ) = pl_module.forward(pts)

        grasp_widths = [
            torch.linalg.norm(
                cp[...,0, :] - cp[...,1, :],
                dim=-1
            ).unsqueeze(-1)
            for cp in contact_pts
        ]

        fig, ax = plt.subplots()

        for b in range(B):
            ## Compute labels
            pt_labels_b, width_labels_b = pl_module.model.class_width_labels(
                contact_pts[b], pts[b], grasp_widths[b], 
                pl_module.model.pt_radius
            )
            for t in range(T):
                cp = torch.cat([contact_pts[b][t][...,0,:], contact_pts[b][t][...,1,:]], dim=-2)
                curve = success_coverage_curve(
                    confs=torch.sigmoid(class_logits[b][t]).squeeze(),
                    pred_grasp_locs=pts[b][t],
                    gt_labels= pt_labels_b[t],
                    pos_gt_grasp_locs=cp
                )
                name = "tsgraspnet"
                curve.to_csv(f"compare/{name}_{b}_{t}.csv")

                plot = plot_s_c_curve(curve, ax=ax)
                plt.pause(0.001)


def success_from_labels(pred_classes, actual_classes):
    """Return success, as proportion, from ground-truth point labels.
    
    (correct positive predictions) / (total positive predictions)"""
    return torch.mean(actual_classes[pred_classes==1].float())

def coverage(pos_pred_grasp_locs, pos_gt_grasp_locs, radius=0.005):
    """Get the proportion of ground truth grasps within epsilon of predicted.
    
    (gt close to predicted) / (total gt)
    """
    if pos_pred_grasp_locs.shape[0] == 0: return torch.Tensor([0.0])

    # for each grasp coordinate in gt_grasps, find the distance to every grasp grasp_coordinate in pred_grasps
    dists = torch.cdist(pos_gt_grasp_locs, pos_pred_grasp_locs) # (n_gt, n_pred)

    closest_dists, idxs = dists.min(axis=1)
    return (closest_dists < radius).float().sum() / len(pos_gt_grasp_locs)

def success_coverage_curve(confs, pred_grasp_locs, gt_labels, pos_gt_grasp_locs):
    """Determine the success and coverage at various threshold confidence values."""

    res = []
    thresholds = torch.linspace(0, 1, 1000)

    # quantiles = torch.linspace(0, 1, 1000)
    # thresholds = [torch.quantile(confs, q) for q in quantiles]

    for t in thresholds:
        pred_classes = confs > t
        pos_pred_grasp_locs = pred_grasp_locs[pred_classes == 1, :]
        res.append({
            "confidence": t,
            # "success": true_positive(pred_classes, gt_labels),  # precision
            # "coverage": recall(pred_classes, gt_labels.ravel()) # recall
            "success": success_from_labels(pred_classes, gt_labels),
            "coverage": coverage(pos_pred_grasp_locs, pos_gt_grasp_locs)
        })
    
    return pd.DataFrame(res).astype(float)


def true_positive(pred, des):
    return float(torch.mean((des.bool()[pred.bool()].float())))

def recall(pred, label):
    return float(torch.mean(pred.bool()[label.bool()].float()))

def plot_s_c_curve(df, ax=None, title="Coverage vs. Success", **plot_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    
    plot = ax.plot(df['coverage'], df['success'], **plot_kwargs)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Success")
    ax.set_title(title)

    return plot