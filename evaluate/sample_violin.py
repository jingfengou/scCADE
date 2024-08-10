import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from utils.general_utils import unique_ind


def sample_violin(model1, model2, datasets, batch_size=None):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distribution (ood) splits.
    """
    model1.eval()
    model2.eval()
    with torch.no_grad():
        evaluation_stats = {
            "ood": sample_violin_plot(
                model1,
                model2,
                datasets["ood"],
                datasets["test"].subset_condition(control=None),
                batch_size=batch_size
            ),
        }

    return evaluation_stats


def sample_violin_plot(model1, model2, dataset, dataset_control,
                         batch_size=None, min_samples=30):
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
                de_genes = dataset.de_genes[cov_pert] # Select the top 10 DE genes
                # de_genes = [pert_category] + ["IFNG", "CD2", "SELL", "CD55"]
                de_idx = np.where(dataset.var_names.isin(np.array(de_genes)))[0]

                perts = dataset.perturbations[idx[0]].view(1, -1).repeat(num, 1).clone()

                num_eval = 0
                yp1 = []
                yp2 = []
                while num_eval < num:
                    end = min(num_eval + batch_size, num)
                    out1 = model1.predict(
                        genes_control[num_eval:end],
                        perts_control[num_eval:end],
                        perts[num_eval:end],
                        [covar[num_eval:end] for covar in covars_control]
                    )
                    out2 = model2.predict(
                        genes_control[num_eval:end],
                        perts_control[num_eval:end],
                        perts[num_eval:end],
                        [covar[num_eval:end] for covar in covars_control]
                    )
                    yp1.append(out1.detach().cpu())
                    yp2.append(out2.detach().cpu())
                    num_eval += batch_size
                yp1 = torch.cat(yp1, 0)
                yp2 = torch.cat(yp2, 0)
                # true values for DE genes
                yt = dataset.genes[idx, :][:, de_idx]
                # predicted values for DE genes
                yp1 = yp1[:, de_idx]
                yp2 = yp2[:, de_idx]
                # control values for DE genes
                yc = genes_control[:, de_idx]

                # Create violin plots for DE genes
                plt.figure(figsize=(15, 12))  # Increase figure size
                data = []
                labels = []
                conditions = ['Control', 'scCADE', 'GraphVCI', 'True']  # Switch 'True' and 'Pred'
                condition_labels = []
                for i, gene in enumerate(de_genes):
                    data.append(yc[:, i].numpy())
                    labels.extend([gene] * yc[:, i].shape[0])
                    condition_labels.extend([conditions[0]] * yc[:, i].shape[0])
                    data.append(yp1[:, i].numpy())
                    labels.extend([gene] * yp1[:, i].shape[0])
                    condition_labels.extend([conditions[1]] * yp1[:, i].shape[0])
                    data.append(yp2[:, i].numpy())
                    labels.extend([gene] * yp2[:, i].shape[0])
                    condition_labels.extend([conditions[2]] * yp2[:, i].shape[0])
                    data.append(yt[:, i].numpy())
                    labels.extend([gene] * yt[:, i].shape[0])
                    condition_labels.extend([conditions[3]] * yt[:, i].shape[0])

                # Combine data and labels
                data = np.concatenate(data)

                # Create a DataFrame for seaborn
                df = pd.DataFrame({
                    'Expression': data,
                    'Gene': labels,
                    'Condition': condition_labels
                })

                sns.violinplot(x='Gene', y='Expression', hue='Condition', data=df, inner='point', scale='width', bw=0.25,
                               width=0.9)  # Adjust scale, bw, and width
                plt.title(f"Violin Plot for DE Genes {cov_category} - {pert_category}")
                plt.xlabel("Genes")
                plt.ylabel("Gene Expression Levels")
                plt.xticks(rotation=45)
                plt.legend(title='Condition')
                plt.show()
    return None  # No need to return R2 scores anymore