import torch
from torch.utils.data import DataLoader
from models.model import DSE, loss_function
import matplotlib.pyplot as plt
import logging
import os
from evaluate.evaluate import evaluate, evaluate_classic
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.general_utils import initialize_logger, ljson
import numpy as np
import importlib
import random
def plot_loss(epochs, train_losses, val_losses, detailed_train_losses, detailed_val_losses, save_path='loss_plot.png'):
    plt.figure(figsize=(15, 10))

    # 绘制训练和验证总损失
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses[:len(epochs)], label='Training Loss')
    # if len(val_losses) > 0:
    #     plt.plot(epochs[:len(val_losses)], val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Total Loss')

    # 绘制详细的训练和验证损失组件
    plt.subplot(2, 1, 2)
    for key, values in detailed_train_losses.items():
        plt.plot(epochs, values[:len(epochs)], label=f'Train {key}')
    # for key, values in detailed_val_losses.items():
    #     if len(values) > 0:
    #         plt.plot(epochs[:len(values)], values, label=f'Val {key}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Components')
    plt.legend()
    plt.title('Detailed Loss Components')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(args, model, datasets, optimizer, device='cpu', print_every=100,
           eval_mode="native", evaluate_every=20):
    # if args["dataset"] == "L008":
    #     evaluate_module = importlib.import_module("evaluate.evaluate_L008")
    # else:
    #     evaluate_module = importlib.import_module("evaluate.evaluate")
    # evaluate = getattr(evaluate_module, "evaluate")
    # evaluate_classic = getattr(evaluate_module, "evaluate_classic")

    best_val_loss = float('inf')
    min_delta = 0.001
    logger = logging.getLogger()
    model.train()
    model.to(device)
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

    writer = SummaryWriter(log_dir= "runs/" + args["dataset"] + "_" + args["model"] + "_CLoss" + str(args["contrast_loss"]) +"_" + str(args["seed"]) + "_" + dt)
    save_dir = "saves/" + args["dataset"] + "_" + args["model"] + "_CLoss" + str(args["contrast_loss"]) + "_" + str(args["seed"]) + "_" + dt
    os.makedirs(save_dir, exist_ok=True)

    initialize_logger(save_dir)
    # ljson({"training_args": args})
    # ljson({"model_params": model.hparams})
    logging.info("")


    start_time = time.time()
    for epoch in range(args["max_epochs"]):
        total_loss = 0
        epoch_detailed_loss = {
            "recon_loss": 0,
            "kl_loss": 0,
            "cont_loss": 0,
            "positive_loss": 0,
            "negative_loss": 0,
            "new_recon_loss": 0,
            "total_loss": 0
        }

        # Training phase
        model.train()
        batch_idx = 0

        batch_num = len(datasets["loader_tr"])


        for data in datasets["loader_tr"]:
            batch_idx+=1

            (genes, perts, cf_genes, cf_perts, covariates) = (
                data[0], data[1], data[2], data[3], data[4:])
            # print(len(covariates))
            # print(covariates[0].shape)
            # print(perts[0].shape)
            optimizer.zero_grad()

            loss, loss_components = model.update(genes, perts, cf_genes, cf_perts, covariates, args)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            for key in loss_components:
                epoch_detailed_loss[key] += loss_components[key]

            if (batch_idx + 1) % print_every == 0:
                # logger.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{batch_num}")
                for key in epoch_detailed_loss:
                    avg_detail = epoch_detailed_loss[key] / ((batch_idx + 1) * len(genes))
                    # logger.info(f"    {key}: {avg_detail:.4f}")
                    print(f"Epoch {epoch + 1}, Batch {batch_idx + 1} - {key}: {avg_detail:.4f}")

        # decay learning rate if necessary
        # also check stopping condition:
        # patience ran out OR max epochs reached

        for key in epoch_detailed_loss:
            epoch_detailed_loss[key] = epoch_detailed_loss[key] / len(datasets["loader_tr"])
            if not (key in model.history.keys()):
                model.history[key] = []
            model.history[key].append(epoch_detailed_loss[key])
        model.history["epoch"].append(epoch)

        ellapsed_minutes = (time.time() - start_time) / 60
        model.history["elapsed_time_min"] = ellapsed_minutes

        # decay learning rate if necessary
        # also check stopping condition:
        # patience ran out OR max epochs reached
        stop = (epoch == args["max_epochs"] - 1)

        if (epoch % evaluate_every) == 0 or stop:
            if eval_mode == "native":
                evaluation_stats = evaluate(model, datasets, epoch, args["batch_size"])
            elif eval_mode == "classic":
                evaluation_stats = evaluate_classic(model, datasets, args["batch_size"])
            else:
                raise ValueError("eval_mode not recognized")

            for key, val in evaluation_stats.items():
                if not (key in model.history.keys()):
                    model.history[key] = []
                model.history[key].append(val)
            model.history["stats_epoch"].append(epoch)

            ljson(
                {
                    "epoch": epoch,
                    "training_stats": epoch_detailed_loss,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                }
            )

            for key, val in epoch_detailed_loss.items():
                writer.add_scalar(key, val, epoch)

            torch.save(
                (model.state_dict(), model.history),
                os.path.join(
                    save_dir,
                    "model_seed={}_epoch={}.pt".format(args["seed"], epoch),
                ),
            )

            ljson(
                {
                    "model_saved": "model_seed={}_epoch={}.pt\n".format(
                        args["seed"], epoch
                    )
                }
            )

            # 早停逻辑
            val_value = np.mean(evaluation_stats["test"])
            if val_value < best_val_loss - min_delta:
                best_val_loss = val_value
                patience_counter = args["patience"]  # 重置耐心计数器
                # 保存模型
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'New best model saved at Epoch {epoch}')
            else:
                patience_counter -= 1
                print(f'No improvement. Patience: {patience_counter}')
                if patience_counter == 0:
                    print('Early stopping triggered')
                    ljson({"early_stop": epoch})
                    break
    writer.close()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


# def validate(model, val_loader, device):
#     logger = logging.getLogger()
#
#     model.eval()
#     total_val_loss = 0
#     epoch_detailed_loss = {
#         "recon_loss": 0,
#         "kl_loss": 0,
#         "cont_loss": 0,
#         "positive_loss": 0,
#         "negative_loss": 0,
#         "new_recon_loss": 0,
#         "total_loss": 0
#     }
#     total_samples = 0
#
#     with torch.no_grad():
#         for batch_idx, data in enumerate(val_loader):
#             x, cell_type, perturbation, dose, x_other, t_other, dose_other = [d.to(device) for d in data]
#             x_recon, mu, logvar, z = model(x, cell_type, perturbation, dose)
#             val_loss, loss_components = loss_function(x_recon, x, mu, logvar, z, cell_type, None, None)
#
#             batch_size = x.size(0)
#             total_val_loss += val_loss.item() * batch_size
#             for key in loss_components:
#                 epoch_detailed_loss[key] += loss_components[key] * batch_size
#             total_samples += batch_size
#
#     avg_val_loss = total_val_loss / total_samples
#     avg_detailed_loss = {key: epoch_detailed_loss[key] / total_samples for key in epoch_detailed_loss}
#
#     logger.info(f"Validation Loss: {avg_val_loss:.4f}")
#     print(f"Validation Loss: {avg_val_loss:.4f}")
#     for key, value in avg_detailed_loss.items():
#         logger.info(f"    {key}: {value:.4f}")
#         print(f"    {key}: {value:.4f}")
#
#     return avg_val_loss, avg_detailed_loss