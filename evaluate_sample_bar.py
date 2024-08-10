import torch
import importlib
from torch.utils.data import DataLoader
import os
from models.model import DSE
from evaluate.sample_bar import sample_bar
from train.train import train
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from models.dataset import load_dataset_splits
from utils.data_utils import data_collate
import argparse
import numpy as np
from graphVCI.gvci.model import load_graphVCI
# 配置日志


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
def find_latest_model_checkpoint(base_path, pattern, seed):
    matched_dirs = [d for d in os.listdir(base_path) if re.match(pattern, d)]
    if not matched_dirs:
        raise FileNotFoundError("No matching directories found.")

    latest_dir = max(matched_dirs, key=lambda x: os.path.getmtime(os.path.join(base_path, x)))
    model_files = [f for f in os.listdir(os.path.join(base_path, latest_dir)) if
                   re.match(rf'model_seed={seed}_epoch=\d+\.pt', f)]

    if not model_files:
        raise FileNotFoundError("No matching model files found.")

    latest_model_file = max(model_files, key=lambda x: int(re.search(r'epoch=(\d+)', x).group(1)))

    return os.path.join(base_path, latest_dir, latest_model_file)

def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser()
    # scCADE
    # setting arguments
    parser.add_argument("--name", default="default_run")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="scCADE")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--contrast_loss", type=bool, default=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--scCADE_model_checkpoint_path", type=str, required=True)
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
    parser.add_argument("--eval_mode", type=str, default="native", help="classic;native")

    # GraphVCI
    # setting arguments
    parser.add_argument('--name', default='train_epoch-1000')
    parser.add_argument("--artifact_path", type=str, default="graphVCI/artifact")
    parser.add_argument("--data_path", type=str, default="graphVCI/datasets/marson_prepped.h5ad")  # sciplex_prepped.h5ad, L008_prepped.h5ad
    parser.add_argument("--graph_path", type=str, default="graphVCI/graphs/marson_grn_128.pth")  # sciplex_grn_128.pth, L008_grn_128.pth
    parser.add_argument('--cpu', default=None, action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument("--GraphVCI_model_checkpoint_path", type=str, required=True)
    # model arguments
    parser.add_argument("--omega0", type=float, default=1.0, help="weight for reconstruction loss")
    parser.add_argument("--omega1", type=float, default=50.0, help="weight for distribution loss")
    parser.add_argument("--omega2", type=float, default=0.1, help="weight for KL divergence")
    parser.add_argument("--graph_mode", type=str, default="sparse", help="dense;sparse")
    parser.add_argument("--outcome_dist", type=str, default="normal", help="nb;zinb;normal")
    parser.add_argument("--dist_mode", type=str, default="match", help="classify;discriminate;fit;match")
    parser.add_argument("--encode_aggr", type=str, default="sum", help="sum;mlp")
    parser.add_argument("--decode_aggr", type=str, default="dot", help="dot;mlp;att")
    parser.add_argument("--hparams", type=str, default="hparams.json")

    return dict(vars(parser.parse_args()))

if __name__ == "__main__":
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


    scCADE_model = DSE(input_dim=input_dim, covariate_embedding_sizes=covariate_embedding_sizes, perturbation_dim=perturbation_dim,
                perturbation_embed_dim=args["perturbation_embed_dim"],
                hidden_dim=args["hidden_dim"],
                latent_dim=args["latent_dim"],
                x_embed_dim=args["x_embed_dim"],
                gene_embed_dim=args["gene_embed_dim"],
                attention_heads=args["attention_heads"],
                device=device)
    scCADE_model.to(device)  # 将模型移动到GPU（如果可用）

    # Load scCADE model parameters

    try:

        # Check the content of the checkpoint
        checkpoint = torch.load(args["scCADE_model_checkpoint_path"], map_location=device)

        # Print the type and keys of the checkpoint
        print(f"Checkpoint type: {type(checkpoint)}")

        if isinstance(checkpoint, tuple):
            state_dict, history = checkpoint
            scCADE_model.load_state_dict(state_dict)
        else:
            raise TypeError("Expected checkpoint to be a tuple, got {type(checkpoint)}.")

    except FileNotFoundError as e:
        print(str(e))
    except TypeError as e:
        print(str(e))




    # Load GraphVCI model parameters

    checkpoint = torch.load(args["GraphVCI_model_checkpoint_path"], map_location=device)

    # 提取模型状态字典
    if isinstance(checkpoint, tuple):
        state_dict = checkpoint[0]  # 假设第一个元素是 state_dict
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")
    graphVCI_model = load_graphVCI(args, state_dict)
    evaluation_stats = sample_bar(scCADE_model, graphVCI_model, datasets, args["batch_size"])