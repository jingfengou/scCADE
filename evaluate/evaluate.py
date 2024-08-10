import numpy as np

from sklearn.metrics import r2_score

import torch
import scanpy as sc
import pandas as pd
from utils.general_utils import unique_ind

def prepare_and_plot_umap(yp_collected, yt_collected, pert_types_collected, save_path='umap_plot.png'):
    # 合并所有数据
    yp_combined = torch.cat(yp_collected, dim=0)
    yt_combined = torch.cat(yt_collected, dim=0)
    data_combined = torch.cat([yp_combined, yt_combined], dim=0)

    # 准备UMAP绘图
    adata = sc.AnnData(data_combined.numpy())
    adata.obs['labels'] = ['Prediction'] * len(yp_combined) + ['True Data'] * len(yt_combined)
    adata.obs['pert_type'] = pert_types_collected

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['labels', 'pert_type'], save=save_path)

def evaluate(model, datasets, epoch, batch_size=None):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distribution (ood) splits.
    """

    model.eval()
    with torch.no_grad():
        evaluation_stats = {
            "training": evaluate_r2(
                model,
                datasets["training"].subset_condition(control=False),
                datasets["training"].subset_condition(control=None),
                epoch,
                batch_size=batch_size
            ),
            "test": evaluate_r2(
                model,
                datasets["test"].subset_condition(control=False),
                datasets["training"].subset_condition(control=None),
                epoch,
                batch_size=batch_size
            ),
            "ood": evaluate_r2(
                model,
                datasets["ood"],
                datasets["test"].subset_condition(control=None),
                epoch,
                batch_size=batch_size,
                ood=True
            ),
        }
    model.train()
    return evaluation_stats

def evaluate_r2(model, dataset, dataset_control, epoch,
                batch_size=None, min_samples=30, ood=False):
    """
    Measures different quality metrics about an `model`, when
    tasked to translate some `genes_control` into each of the perturbation/covariates
    combinations described in `dataset`.

    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """
    # yp_collected = []
    # yt_collected = []
    # pert_types_collected = []
    mean_score_mean, mean_score_robust = [], []
    mean_score_de_mean, mean_score_de_robust = [], []

    cov_cats = unique_ind(dataset.cov_names)
    cov_cats_control = unique_ind(dataset_control.cov_names)
    pert_cats = unique_ind(dataset.pert_names)
    for cov_category in cov_cats.keys():
        idx_control = cov_cats_control[cov_category]
        genes_control = dataset_control.genes[idx_control]
        perts_control = dataset_control.perturbations[idx_control]
        covars_control = [covar[idx_control] for covar in dataset_control.covariates]

        pert_names_control = dataset_control.pert_names[idx_control]
        pert_names_control_cats = unique_ind(pert_names_control)

        num = genes_control.size(0)
        if batch_size is None:
            batch_size = num
        for pert_category in pert_cats.keys():
            idx = np.intersect1d(cov_cats[cov_category], pert_cats[pert_category])
            # estimate metrics only for reasonably-sized perturbation/cell-type combos
            if len(idx) > min_samples:
                cov_pert = dataset.cov_pert[idx[0]]
                # 使用 get() 获取 genes 列表，如果 cov_pert 键不存在，则返回空数组
                de_genes = np.array(dataset.de_genes.get(cov_pert, []))

                # 检查是否有基因数据
                de_idx = np.where(dataset.var_names.isin(de_genes))[0] if de_genes.size > 0 else np.array([])
                # de_idx = np.where(
                #     dataset.var_names.isin(np.array(dataset.de_genes[cov_pert]))
                # )[0]

                perts = dataset.perturbations[idx[0]].view(1, -1).repeat(num, 1).clone()


                yp = []
                for num_eval in range(0, num, batch_size):
                    end = min(num_eval + batch_size, num)
                    out = model.predict(
                        genes_control[num_eval:end],
                        perts_control[num_eval:end],
                        perts[num_eval:end],
                        [covar[num_eval:end] for covar in covars_control]
                    )
                    yp.append(out.detach().cpu())
                yp = torch.cat(yp, 0)

                # true means
                yt = dataset.genes[idx, :]
                yt_m = yt.mean(0)

                yp_m = yp.mean(0)
                # yp_collected.append(yp)
                # yt_collected.append(yt)
                # pert_types_collected.extend([pert_category]*(len(yp)+len(yt)))

                mean_score_mean.append(r2_score(yt_m, yp_m))

                if de_idx.size > 0:
                    mean_score_de_mean.append(r2_score(yt_m[de_idx], yp_m[de_idx]))
                # if ood:
                #     prepare_and_plot_umap(yp_collected, yt_collected, pert_types_collected, save_path= 'epoch' +str(epoch) + 'umap_plot.png')

                if pert_category in pert_names_control_cats:
                    pert_idx = pert_names_control_cats[pert_category]

                    yp_r = yp_m + (genes_control[pert_idx] - yp[pert_idx]).mean(0)
                    mean_score_robust.append(r2_score(yt_m, yp_r))
                    if de_idx.size > 0:
                        mean_score_de_robust.append(r2_score(yt_m[de_idx], yp_r[de_idx]))

    return [
        np.mean(s) if s else -1
        for s in [
            mean_score_mean, mean_score_robust,
            mean_score_de_mean, mean_score_de_robust
        ]
    ]

# def evaluate_r2(model, dataset, dataset_control, epoch,
#                 batch_size=None, min_samples=30, ood=False):
#     yp_collected = []
#     yt_collected = []
#     pert_types_collected = []
#     mean_score_mean, mean_score_robust = [], []
#     mean_score_de_mean, mean_score_de_robust = [], []
#
#     # 预计算索引映射
#     cov_cats = unique_ind(dataset.cov_names)
#     cov_cats_control = unique_ind(dataset_control.cov_names)
#     pert_cats = unique_ind(dataset.pert_names)
#
#     # 预先提取并缓存相关数据
#     genes_control = dataset_control.genes
#     perts_control = dataset_control.perturbations
#     covars_control = dataset_control.covariates
#     pert_names_control = dataset_control.pert_names
#
#     for cov_category in cov_cats:
#         idx_control = cov_cats_control[cov_category]
#
#         num = len(idx_control)
#         if batch_size is None:
#             batch_size = num
#
#         for pert_category in pert_cats:
#             idx = np.intersect1d(cov_cats[cov_category], pert_cats[pert_category])
#             if len(idx) > min_samples:
#                 cov_pert = dataset.cov_pert[idx[0]]
#                 de_genes = np.array(dataset.de_genes.get(cov_pert, []))
#
#                 if de_genes.size > 0:
#                     de_idx = np.where(dataset.var_names.isin(de_genes))[0]
#                 else:
#                     de_idx = np.array([])
#
#                 perts_batch = dataset.perturbations[idx[0]].unsqueeze(0).repeat(num, 1)
#
#                 num_eval = 0
#                 yp = []
#                 while num_eval < num:
#                     end = min(num_eval + batch_size, num)
#                     out = model.predict(
#                         genes_control[idx_control][num_eval:end],
#                         perts_control[idx_control][num_eval:end],
#                         perts_batch[num_eval:end],
#                         [cov[num_eval:end] for cov in covars_control]
#                     )
#                     yp.append(out.detach().cpu())
#                     num_eval += batch_size
#
#                 yp = torch.cat(yp, 0)
#                 yt = dataset.genes[idx, :]
#                 yt_m = yt.mean(0)
#                 yp_m = yp.mean(0)
#
#                 mean_score_mean.append(r2_score(yt_m, yp_m))
#                 if de_idx.size > 0 and yt_m[de_idx].size > 0 and yp_m[de_idx].size > 0:
#                     mean_score_de_mean.append(r2_score(yt_m[de_idx], yp_m[de_idx]))
#                 else:
#                     mean_score_de_mean.append(None)  # 或其他合适的默认值
#
#     return [
#         np.mean(s) if s else -1 for s in [
#             mean_score_mean, mean_score_robust, mean_score_de_mean, mean_score_de_robust
#         ]
#     ]
#####################################################
#                 CLASSIC EVALUATION                #
#####################################################
def evaluate_classic(model, datasets, batch_size=None):
    """
    `evaluate` used in CPA
    https://github.com/facebookresearch/CPA
    """

    model.eval()
    with torch.no_grad():
        evaluation_stats = {
            "training": evaluate_r2_classic(
                model,
                datasets["training"].subset_condition(control=False),
                datasets["training"].subset_condition(control=True),
                batch_size=batch_size
            ),
            "test": evaluate_r2_classic(
                model,
                datasets["test"].subset_condition(control=False),
                datasets["test"].subset_condition(control=True),
                batch_size=batch_size
            ),
            "ood": evaluate_r2_classic(
                model,
                datasets["ood"],
                datasets["test"].subset_condition(control=True),
                batch_size=batch_size
            ),
        }
    model.train()
    return evaluation_stats

def evaluate_r2_classic(model, dataset, dataset_control, batch_size=None, min_samples=30):
    """
    `evaluate_r2` used in CPA
    https://github.com/facebookresearch/CPA
    """

    mean_score, mean_score_de = [], []
    #var_score, var_score_de = [], []
    genes_control = dataset_control.genes
    perts_control = dataset_control.perturbations
    num = genes_control.size(0)
    if batch_size is None:
        batch_size = num

    for pert_category in np.unique(dataset.cov_pert):
        # pert_category category contains: 'cov_pert' info
        de_idx = np.where(
            dataset.var_names.isin(np.array(dataset.de_genes[pert_category]))
        )[0]

        idx = np.where(dataset.cov_pert == pert_category)[0]

        # estimate metrics only for reasonably-sized perturbation/cell-type combos
        if len(idx) > min_samples:
            perts = dataset.perturbations[idx][0].view(1, -1).repeat(num, 1).clone()
            covars = [
                covar[idx][0].view(1, -1).repeat(num, 1).clone()
                for covar in dataset.covariates
            ]

            num_eval = 0
            yp = []
            while num_eval < num:
                end = min(num_eval+batch_size, num)
                out = model.predict(
                    genes_control[num_eval:end],
                    perts_control[num_eval:end],
                    perts[num_eval:end],
                    [covar[num_eval:end] for covar in covars]
                )
                yp.append(out.detach().cpu())

                num_eval += batch_size
            yp = torch.cat(yp, 0)
            yp_m = yp.mean(0)

            # true means
            yt = dataset.genes[idx, :].numpy()
            yt_m = yt.mean(axis=0)

            mean_score.append(r2_score(yt_m, yp_m))
            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))

    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de]
    ]
