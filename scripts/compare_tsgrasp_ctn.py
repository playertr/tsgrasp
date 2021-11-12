# compare_ctn_tsgrasp.py
# Tim Player, 6 November 2021, playert@oregonstate.edu
# Script for comparing TSGrasp to Contact-Torchnet, the PyTorch port of 
# Contact-Graspnet.
import hydra
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
import torch
from tqdm import tqdm
from tsgrasp.net.lit_minkowski_graspnet import LitMinkowskiGraspNet
from tsgrasp.net.lit_temporal_contact_torchnet import LitTemporalContactTorchNet

from tsgrasp.utils.metric_utils.metrics import success_coverage_curve

def compare(tsgrasp: LitMinkowskiGraspNet, ctn: LitTemporalContactTorchNet, test_dataloader):

    sc_curves = []
    for batch_idx, batch in enumerate(tqdm(test_dataloader)):

        
        ## Get the results from tsgrasp on this batch
        tsgrasp_results = tsgrasp.test_step(batch, batch_idx)

        ## Prepare a 20,000-point sampled batch for CTN
        def subsample_idxs(N, num_tot):
            """Indices to randomly draw N samples without replacement. """
            return torch.randperm(num_tot, dtype=torch.int32, device='cpu')[:N].sort()[0].long()
        from copy import deepcopy
        ctn_batch = deepcopy(batch)
        idxs = subsample_idxs(20_000, batch['all_pos'].shape[2])
        ctn_batch['all_pos'] = batch['all_pos'][:, :, idxs, :]

        ## Get the results from CTN on the sampled batch
        ctn_results = ctn.test_step(ctn_batch, batch_idx)

        ctn_confs = [torch.sigmoid(r[0]) for r in ctn_results['outputs']]
        ctn_pred_pts = [r[4] for r in ctn_results['outputs']]
        
        tsgrasp_pred_pts = batch['positions'].squeeze()
        tsgrasp_confs = torch.sigmoid(tsgrasp_results['outputs'][0].reshape(*tsgrasp_pred_pts.shape[:2]))
        tsgrasp_labels = tsgrasp_results['pt_labels'].reshape(*tsgrasp_pred_pts.shape[:2])

        sc_ctns = []
        sc_tsgrasps = []
        sc_tsgrasp_ctns = []
        ntimes = len(ctn_confs)
        for t in tqdm(range(ntimes)):
            ctn_pred_grasp_locs = ctn_pred_pts[t].squeeze()
            ctn_gt_labels = ctn_results['pt_labels'][t]

            # S-C curve of Contact-Torchnet
            sc_ctn = success_coverage_curve(
                confs=ctn_confs[t],
                pred_grasp_locs=ctn_pred_grasp_locs,
                gt_labels=ctn_gt_labels,
                pos_gt_grasp_locs=ctn_pred_grasp_locs[ctn_gt_labels]
            )
            sc_ctns.append(sc_ctn)

            # S-C curve of TSGrasp
            sc_tsgrasp = success_coverage_curve(
                confs = tsgrasp_confs[t],
                pred_grasp_locs=tsgrasp_pred_pts[t],
                gt_labels=tsgrasp_labels[t],
                pos_gt_grasp_locs=tsgrasp_pred_pts[t][tsgrasp_labels[t].bool()]
            )
            sc_tsgrasps.append(sc_tsgrasp)

            idcs = deindex(tsgrasp_pred_pts[t], ctn_pred_grasp_locs)
            

            pred_grasp_locs = tsgrasp_pred_pts[t][idcs]
            gt_labels = tsgrasp_labels[t][idcs]
            # S-C curve of TSGrasp, using Contact-Torchnet's points
            sc_tsgrasp_ctn = success_coverage_curve(
                confs = tsgrasp_confs[t][idcs],
                pred_grasp_locs=pred_grasp_locs,
                gt_labels=gt_labels,
                pos_gt_grasp_locs=pred_grasp_locs[gt_labels.bool()]
            )
            sc_tsgrasp_ctns.append(sc_tsgrasp_ctn)

            # True positive, false positive, true negative, false negative at 0.5
            ctn_preds = ctn_confs[t] > 0.5
            tp_ctn = torch.mean(ctn_gt_labels[ctn_preds].float())
            fp_ctn = torch.mean((~ctn_gt_labels[ctn_preds]).float())
            tn_ctn = torch.mean((~ctn_gt_labels[~ctn_preds]).float())
            fn_ctn = torch.mean(ctn_gt_labels[~ctn_preds].float())
            print("CTN")
            print(f"TP: {tp_ctn:.2f} \tFP: {fp_ctn:.2f} \tTN: {tn_ctn:.2f} \FN: {fn_ctn:.2f}")

            tsgrasp_preds = tsgrasp_confs[t] > 0.5
            ts_labels = tsgrasp_labels[t].bool()
            tp_tsgrasp = torch.mean(ts_labels[tsgrasp_preds].float())
            fp_tsgrasp = torch.mean((~ts_labels[tsgrasp_preds]).float())
            tn_tsgrasp = torch.mean((~ts_labels[~tsgrasp_preds]).float())
            fn_tsgrasp = torch.mean(ts_labels[~tsgrasp_preds].float())
            print("TSGrasp")
            print(f"TP: {tp_tsgrasp:.2f} \tFP: {fp_tsgrasp:.2f} \tTN: {tn_tsgrasp:.2f} \FN: {fn_tsgrasp:.2f}")


            import pickle
            with open(f"/home/tim/Research/tsgrasp/compare/out{batch_idx}_{t}.pickle", 'wb') as f:
                pickle.dump({
                    "ctn": dict(
                        confs=ctn_confs[t].detach().cpu(),
                        pred_grasp_locs=ctn_pred_grasp_locs.detach().cpu(),
                        gt_labels=ctn_gt_labels.detach().cpu(),
                        pos_gt_grasp_locs=ctn_pred_grasp_locs[ctn_gt_labels].detach().cpu()
                    ),
                    "tsgrasp": dict(
                        confs = tsgrasp_confs[t].detach().cpu(),
                        pred_grasp_locs=tsgrasp_pred_pts[t].detach().cpu(),
                        gt_labels=tsgrasp_labels[t].detach().cpu(),
                        pos_gt_grasp_locs=tsgrasp_pred_pts[t][tsgrasp_labels[t].bool()].detach().cpu()
                    ),
                    "tsgrasp_ctn_pts": dict(
                        confs = tsgrasp_confs[t][idcs].detach().cpu(),
                        pred_grasp_locs=pred_grasp_locs.detach().cpu(),
                        gt_labels=gt_labels.detach().cpu(),
                        pos_gt_grasp_locs=pred_grasp_locs[gt_labels.bool()]
                    ),
                }, f)

        import pandas as pd
        super_sc_ctn = pd.concat(sc_ctns).groupby('confidence').mean()
        super_sc_tsgrasp = pd.concat(sc_tsgrasps).groupby('confidence').mean()
        super_sc_tsgrasp_ctn = pd.concat(sc_tsgrasp_ctns).groupby('confidence').mean()

        super_sc_ctn.to_csv(f"/home/tim/Research/tsgrasp/compare/super_sc_ctn_{batch_idx}.csv")
        super_sc_tsgrasp.to_csv(f"/home/tim/Research/tsgrasp/compare/super_sc_tsgrasp_{batch_idx}.csv")
        super_sc_tsgrasp_ctn.to_csv(f"/home/tim/Research/tsgrasp/compare/super_sc_tsgrasp_ctn_{batch_idx}.csv")

        sc_curves.append((super_sc_ctn, super_sc_tsgrasp, super_sc_tsgrasp_ctn))

        from tsgrasp.utils.metric_utils.metrics import plot_s_c_curve
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        plot_s_c_curve(super_sc_ctn, ax=ax, label='CTN')
        plot_s_c_curve(super_sc_tsgrasp, ax=ax, label='TSGrasp')
        plot_s_c_curve(super_sc_tsgrasp_ctn, ax=ax, label='TSGrasp-CTNpts')
        fig.legend()

        plt.savefig(f"/home/tim/Research/tsgrasp/compare/sc_{batch_idx}.png")

        
# torch.mean(
#     ((tsgrasp_confs[0] > 0.5) == tsgrasp_labels[0]).float()
# )

def deindex(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Find the row indices idcs such that A[idcs] = B. 
    
    A and B are 2D matrices."""
    idcs = []
    for row in B:
        idx, = torch.where((A == row).all(dim=1))
        idcs.append(idx)
    return torch.cat(idcs)


def plot_point_cloud(xyz):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    o3d.visualization.draw_geometries([pcd])


@hydra.main(config_path="../conf", config_name="compare")
def main(cfg: DictConfig):

    ## Instantiate the test dataset from the config
    pl_datamodule = instantiate(cfg.data, batch_size=cfg.training.batch_size)
    pl_datamodule.setup()
    test_dataloader = pl_datamodule.test_dataloader()

    ## Instantiate both Lightning modules from the config and prepare for inference
    tsgrasp = instantiate(cfg.tsgrasp_model, training_cfg=cfg.training)
    ctn = instantiate(cfg.ctn_model, training_cfg=cfg.training)

    tsgrasp.load_state_dict(torch.load(cfg.tsgrasp_ckpt)['state_dict'])
    ctn.load_state_dict(torch.load(cfg.ctn_ckpt)['state_dict'])
    
    tsgrasp.eval()
    ctn.eval()

    ## Compare the outputs of each network on the datamodule's examples
    compare(tsgrasp, ctn, test_dataloader)


if __name__ == '__main__':
    main()