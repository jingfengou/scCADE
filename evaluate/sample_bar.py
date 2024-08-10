import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from utils.general_utils import unique_ind


def sample_bar(model1, model2, datasets, batch_size=None):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distribution (ood) splits.
    """
    model1.eval()
    model2.eval()
    with torch.no_grad():
        evaluation_stats = {
            "ood": sample_bar_plot(
                model1,
                model2,
                datasets["ood"],
                datasets["test"].subset_condition(control=True),
                batch_size=batch_size
            ),
        }

    return evaluation_stats


def sample_bar_plot(model1, model2, dataset, dataset_control,
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
        for pert_category in ["SLA2", "CD247", "AKAP12"]:
            idx = np.intersect1d(cov_cats[cov_category], pert_cats[pert_category])
            # estimate metrics only for reasonably-sized perturbation/cell-type combos
            if len(idx) > min_samples:
                cov_pert = dataset.cov_pert[idx[0]]
                de_genes = dataset.de_genes[cov_pert] # Select the top 10 DE genes
                # de_genes = [pert_category] + ["IFNG", "CD2", "SELL", "CD55"]
                de_idx = np.where(dataset.var_names.isin(np.array(de_genes)))[0]
                # Select a single perturbation sample
                sample_idx = idx[0]

                yp1 = model1.predict(
                    dataset.genes[sample_idx].unsqueeze(0),
                    dataset.perturbations[sample_idx].unsqueeze(0),
                    dataset.perturbations[sample_idx].unsqueeze(0),
                    [covar[sample_idx].unsqueeze(0) for covar in dataset.covariates]
                ).detach().cpu()

                yp2 = model2.predict(
                    dataset.genes[sample_idx].unsqueeze(0),
                    dataset.perturbations[sample_idx].unsqueeze(0),
                    dataset.perturbations[sample_idx].unsqueeze(0),
                    [covar[sample_idx].unsqueeze(0) for covar in dataset.covariates]
                ).detach().cpu()

                # true values for DE genes
                yt = dataset.genes[sample_idx, de_idx].unsqueeze(0)
                # predicted values for DE genes
                yp1 = yp1[:, de_idx]
                yp2 = yp2[:, de_idx]
                # control values for DE genes
                yc = genes_control[:, de_idx]

                # Prepare data for bar plots
                data_control = yc.mean(dim=0).numpy()
                data_GraphVCI = yp2.mean(dim=0).numpy()  # Switched position
                data_scCADE = yp1.mean(dim=0).numpy()  # Switched position
                data_true = yt.mean(dim=0).numpy()
                genes = [gene for gene in de_genes]

                # Determine local min and max values for y axis
                local_max = max(data_control.max(), data_scCADE.max(), data_GraphVCI.max(), data_true.max())
                local_min = min(data_control.min(), data_scCADE.min(), data_GraphVCI.min(), data_true.min())

                # Create a new figure and subplots for each set of data
                fig, axes = plt.subplots(4, 1, figsize=(12.5, 12.5))  # Create a figure with 4 subplots (4 rows, 1 column)

                # Create bar plots for each condition
                axes[0].bar(genes, data_control, color='blue')
                axes[0].set_title("Control", fontsize=16)
                axes[0].set_ylabel("Expression Levels", fontsize=14)
                axes[0].set_ylim(local_min, local_max)

                axes[1].bar(genes, data_GraphVCI, color='green')  # Switched position
                axes[1].set_title("GraphVCI", fontsize=16)  # Switched position
                axes[1].set_ylabel("Expression Levels", fontsize=14)
                axes[1].set_ylim(local_min, local_max)

                axes[2].bar(genes, data_scCADE, color='orange')  # Switched position
                axes[2].set_title("scCADE", fontsize=16)  # Switched position
                axes[2].set_ylabel("Expression Levels", fontsize=14)
                axes[2].set_ylim(local_min, local_max)

                axes[3].bar(genes, data_true, color='red')
                axes[3].set_title("True", fontsize=16)
                axes[3].set_ylabel("Expression Levels", fontsize=14)
                axes[3].set_xlabel("Genes", fontsize=16)
                axes[3].set_ylim(local_min, local_max)

                for ax in axes:
                    ax.tick_params(axis='x', rotation=45)

                plt.tight_layout()
                plt.show()

                # Break after plotting the first set

    #             perts = dataset.perturbations[idx[0]].view(1, -1).repeat(num, 1).clone()
    #
    #             num_eval = 0
    #             yp1 = []
    #             yp2 = []
    #             while num_eval < num:
    #                 end = min(num_eval + batch_size, num)
    #                 out1 = model1.predict(
    #                     genes_control[num_eval:end],
    #                     perts_control[num_eval:end],
    #                     perts[num_eval:end],
    #                     [covar[num_eval:end] for covar in covars_control]
    #                 )
    #                 out2 = model2.predict(
    #                     genes_control[num_eval:end],
    #                     perts_control[num_eval:end],
    #                     perts[num_eval:end],
    #                     [covar[num_eval:end] for covar in covars_control]
    #                 )
    #                 yp1.append(out1.detach().cpu())
    #                 yp2.append(out2.detach().cpu())
    #                 num_eval += batch_size
    #             yp1 = torch.cat(yp1, 0)
    #             yp2 = torch.cat(yp2, 0)
    #             # true values for DE genes
    #             yt = dataset.genes[idx, :][:, de_idx]
    #             # predicted values for DE genes
    #             yp1 = yp1[:, de_idx]
    #             yp2 = yp2[:, de_idx]
    #             # control values for DE genes
    #             yc = genes_control[:, de_idx]
    #
    #             # Create violin plots for DE genes
    #             plt.figure(figsize=(15, 12))  # Increase figure size
    #             data = []
    #             labels = []
    #             conditions = ['Control', 'scCADE', 'GraphVCI', 'True']  # Switch 'True' and 'Pred'
    #             condition_labels = []
    #             for i, gene in enumerate(de_genes):
    #                 data.append(yc[:, i].numpy())
    #                 labels.extend([gene] * yc[:, i].shape[0])
    #                 condition_labels.extend([conditions[0]] * yc[:, i].shape[0])
    #                 data.append(yp1[:, i].numpy())
    #                 labels.extend([gene] * yp1[:, i].shape[0])
    #                 condition_labels.extend([conditions[1]] * yp1[:, i].shape[0])
    #                 data.append(yp2[:, i].numpy())
    #                 labels.extend([gene] * yp2[:, i].shape[0])
    #                 condition_labels.extend([conditions[2]] * yp2[:, i].shape[0])
    #                 data.append(yt[:, i].numpy())
    #                 labels.extend([gene] * yt[:, i].shape[0])
    #                 condition_labels.extend([conditions[3]] * yt[:, i].shape[0])
    #
    #             # Combine data and labels
    #             data = np.concatenate(data)
    #
    #             # Create a DataFrame for seaborn
    #             df = pd.DataFrame({
    #                 'Expression': data,
    #                 'Gene': labels,
    #                 'Condition': condition_labels
    #             })
    #
    #             sns.violinplot(x='Gene', y='Expression', hue='Condition', data=df, inner='point', scale='width', bw=0.25,
    #                            width=0.9)  # Adjust scale, bw, and width
    #             plt.title(f"Violin Plot for DE Genes {cov_category} - {pert_category}")
    #             plt.xlabel("Genes")
    #             plt.ylabel("Gene Expression Levels")
    #             plt.xticks(rotation=45)
    #             plt.legend(title='Condition')
    #             plt.show()
    return None  # No need to return R2 scores anymore