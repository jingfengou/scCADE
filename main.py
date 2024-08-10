import torch
from torch.utils.data import DataLoader
from models.model import DSE
from train.train import train
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from models.dataset import load_dataset_splits
from utils.data_utils import data_collate
import numpy as np


import argparse
# check cuda
import re
import subprocess
def select_least_used_gpu():
    if torch.cuda.is_available():
        smi_output = subprocess.check_output('nvidia-smi', encoding='utf-8')
        # 正则表达式来找到显存使用信息
        gpu_memory = [int(x) for x in re.findall(r'(\d+)MiB / \d+MiB', smi_output)]
        gpu_index = gpu_memory.index(min(gpu_memory))
        return torch.device(f"cuda:{gpu_index}")
    else:
        return torch.device("cpu")
def prepare(args, state_dict=None):
    """
    Instantiates model and dataset to run an experiment.
    """

    datasets = load_dataset_splits(
        args["data_path"],
        sample_cf=(True if args["dist_mode"] == "match" else False),
    )

    args["num_outcomes"] = datasets["training"].num_genes
    args["num_treatments"] = datasets["training"].num_perturbations
    args["num_covariates"] = datasets["training"].num_covariates


    return datasets
def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser()

    # setting arguments
    parser.add_argument("--name", default="default_run")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="scCADE")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--contrast_loss", type=bool, default=True)
    parser.add_argument("--dataset", type=str, required=True)
    # model arguments
    parser.add_argument("--perturbation_embed_dim", type=int, default=128,  help="dim for perturbation embed")
    parser.add_argument("--hidden_dim", type=int, default=256, help="dim for hidden variate")
    parser.add_argument("--latent_dim", type=int, default=64, help="dim for latent layer")
    parser.add_argument("--x_embed_dim", type=int, default=512, help="dim for gene embed")
    parser.add_argument("--gene_embed_dim", type=int, default=128, help="dim for gene feature")
    parser.add_argument("--attention_heads", type=int, default="4")
    parser.add_argument("--kl_weight", type=float, default="0.1")
    parser.add_argument("--contrastive_weight", type=float, default="1.0")
    parser.add_argument("--margin", type=float, default="1.0")
    parser.add_argument("--rec_weight", type=float, default="1.0")
    parser.add_argument("--new_weight", type=float, default="1.0")
    parser.add_argument("--dist_mode", type=str, default="match", help="discriminate;fit;match")

    # training arguments
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    parser.add_argument("--eval_mode", type=str, default="native", help="classic;native")

    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(filename='training.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    args = parse_arguments()



    device = select_least_used_gpu()

    if args["seed"] is not None:
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])
        torch.cuda.manual_seed_all(args["seed"])

    if args["dataset"] == "sciplex":
        input_dim = 2000
        covariate_embedding_sizes = [(3, 64), (2, 64)]
        perturbation_dim = 189
    elif args["dataset"] == "marson":
        input_dim = 2013
        covariate_embedding_sizes = [(2, 64), (2, 64), (2, 64)]
        perturbation_dim = 70
    elif args["dataset"] == "L008":
        input_dim = 2048
        covariate_embedding_sizes = [(3, 64)]
        perturbation_dim = 80
    datasets = prepare(args)
    datasets.update(
        {
            "loader_tr": torch.utils.data.DataLoader(
                datasets["training"],
                batch_size=args["batch_size"],
                shuffle=True,
                collate_fn=(lambda batch: data_collate(batch, nb_dims=1))
            )
        }
    )


    model = DSE(input_dim=input_dim, covariate_embedding_sizes=covariate_embedding_sizes, perturbation_dim=perturbation_dim,
                perturbation_embed_dim=args["perturbation_embed_dim"],
                hidden_dim=args["hidden_dim"],
                latent_dim=args["latent_dim"],
                x_embed_dim=args["x_embed_dim"],
                gene_embed_dim=args["gene_embed_dim"],
                attention_heads=args["attention_heads"],
                device=device)
    model.to(device)  # 将模型移动到GPU（如果可用）
    # 使用 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 使用学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train(args, model, datasets, optimizer, device=device, print_every=100,
              eval_mode="native")

