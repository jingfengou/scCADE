import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

class Disentangler(nn.Module):
    def __init__(self, input_dim, covariate_embedding_sizes, perturbation_dim,
                 perturbation_embed_dim, hidden_dim, latent_dim, x_embed_dim, gene_embed_dim, attention_heads=4):
        super(Disentangler, self).__init__()

        # 1D卷积层
        self.gene_embed = nn.Linear(1, gene_embed_dim)

        self.C_embed = nn.ModuleList([
            nn.Linear(1, gene_embed_dim) for i in range(len(covariate_embedding_sizes))])
        self.p_embed = nn.Linear(1, gene_embed_dim)
        self.relu = nn.ReLU()
        self.attention = nn.MultiheadAttention(embed_dim=gene_embed_dim, num_heads=attention_heads)
        self.squeeze = nn.Linear(gene_embed_dim, 1)
        # Embeddings
        self.x_embedding = nn.Linear(input_dim, x_embed_dim)

        self.covariate_embeddings = nn.ModuleList([
            nn.Linear(size, dim) for size, dim in covariate_embedding_sizes
        ])
        self.perturbation_embedding = nn.Linear(perturbation_dim, perturbation_embed_dim)

        total_covariate_embedding_dim = sum(dim for _, dim in covariate_embedding_sizes)

        combined_input_dim = x_embed_dim + total_covariate_embedding_dim + perturbation_embed_dim

        self.fc1 = nn.Linear(combined_input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, genes, covariates, pert):
        x_embed = F.relu(self.x_embedding(genes))
        x_embed =x_embed.unsqueeze(-1)
        x_embed = self.relu(self.gene_embed(x_embed))

        c_embed = [embed(c_part) for embed, c_part in zip(self.covariate_embeddings, covariates)]

        # 假设pert是一个整数列表，每个整数是一个索引

        p_embed = self.perturbation_embedding(pert)
        p_embed = p_embed.unsqueeze(-1)
        p_embed = self.relu(self.p_embed(p_embed))
        if isinstance(c_embed, list):
            # 检查c_embed中的元素是否都是Tensor
            if all(isinstance(x, torch.Tensor) for x in c_embed):
                # 将c_embed中的所有Tensor在特定维度（例如dim=1）上拼接起来
                c_embed = [self.relu(c_embed(c.unsqueeze(-1))) for c, c_embed in zip(c_embed, self.C_embed)]

                c_embed = torch.cat(c_embed, dim=1)
            else:
                raise TypeError("All elements in c_embed must be tensors.")
        else:
            raise TypeError("c_embed is expected to be a list of tensors.")

        # 现在所有的嵌入向量都是Tensor，可以安全地拼接

        embeddings = [x_embed, c_embed, p_embed]
        embeddings = torch.cat(embeddings, dim=1)
        embeddings = embeddings.permute(1, 0, 2)
        embeddings, _ = self.attention(embeddings, embeddings, embeddings)
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = self.squeeze(embeddings)
        h = embeddings.squeeze(-1)
        h = F.relu(self.fc1(h))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)

        return mu, logvar

class Entangler(nn.Module):
    def __init__(self, latent_dim, Disentangler, perturbation_embed_dim, hidden_dim,
                 output_dim):
        super(Entangler, self).__init__()

        self.perturbation_embedding = Disentangler.perturbation_embedding

        input_features = latent_dim + perturbation_embed_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, z, perturbation):
        p_embed = self.perturbation_embedding(perturbation)
        components = [z, p_embed]


        h = torch.cat(components, dim=1)
        h = self.fc_layers(h)
        x_recon = self.fc2(h)
        return x_recon

class DSE(nn.Module):
    def __init__(self, input_dim, covariate_embedding_sizes, perturbation_dim,
                 perturbation_embed_dim, hidden_dim, latent_dim, x_embed_dim, gene_embed_dim, attention_heads, device="cpu"):
        super(DSE, self).__init__()
        self.disentangler = Disentangler(input_dim, covariate_embedding_sizes, perturbation_dim,
                 perturbation_embed_dim, hidden_dim, latent_dim, x_embed_dim, gene_embed_dim, attention_heads)
        self.entangler = Entangler(latent_dim, self.disentangler, perturbation_embed_dim, hidden_dim,
                 input_dim)
        self.to_device(device)

        self.history = {"epoch": [], "stats_epoch": []}

    def to_device(self, device):
        self.device = device
        self.to(self.device)
    def move_input(self, input):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if isinstance(input, list):
            return [i.to(self.device) if i is not None else None for i in input]
        else:
            return input.to(self.device)

    def move_inputs(self, *inputs: torch.Tensor):
        """
        Move minibatch tensors to CPU/GPU.
        """
        return [self.move_input(i) if i is not None else None for i in inputs]
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, genes, covariates, pert):
        mu, logvar = self.disentangler(genes, covariates, pert)
        z = self.reparameterize(mu, logvar)
        x_recon = self.entangler(z, pert)
        return x_recon, mu, logvar, z
    def update(self, genes, perts, cf_genes, cf_perts, covariates, args):
        """
        Update VCI's parameters given a minibatch of outcomes, treatments, and covariates.
        """
        genes, perts, cf_genes, cf_perts, covariates = self.move_inputs(
            genes, perts, cf_genes, cf_perts, covariates
        )

        x_recon, mu, logvar, z = self.forward(
            genes, covariates, perts
        )
        new_x_recon = self.entangler(z, cf_perts)
        cf_outcomes_mean = [torch.mean(tensor, dim=0) if tensor is not None else None for tensor in cf_genes]
        loss, loss_components = loss_function(x_recon, genes, mu, logvar, z, covariates, args, new_x_recon, cf_outcomes_mean)

        return loss, loss_components
    def predict(self, genes, perts, cf_perts, covariates):
        """
        Update VCI's parameters given a minibatch of outcomes, treatments, and covariates.
        """
        genes, perts, cf_perts, covariates = self.move_inputs(
            genes, perts, cf_perts, covariates
        )

        mu, logvar = self.disentangler(genes, covariates, perts)
        z = self.reparameterize(mu, logvar)

        new_x_recon = self.entangler(z, cf_perts)


        return new_x_recon


def binary_to_int_labels(binary_labels):
    """将二进制编码的标签tensor转换为整数标签。

    Args:
    binary_labels (Tensor): 一个包含二进制标签的tensor，形状为 [n_samples, n_features]

    Returns:
    Tensor: 转换后的整数标签，形状为 [n_samples, 1]
    """
    multiplier = 1
    int_labels = torch.zeros(binary_labels.size(0), dtype=torch.long, device=binary_labels.device)
    # 从最低位到最高位计算整数值
    for i in reversed(range(binary_labels.size(1))):
        int_labels += binary_labels[:, i] * multiplier
        multiplier *= 2
    return int_labels


def contrastive_loss(z, labels_list, margin=1, epsilon=1e-8):
    batch_size = z.size(0)
    distance_matrix = torch.cdist(z, z, p=2)

    # 转换标签并计算整数标签
    int_labels = [binary_to_int_labels(label.long()) for label in labels_list]

    # 初始化相似性矩阵
    similarity_matrix = torch.zeros(batch_size, batch_size, device=z.device)

    # 更新权重策略：0号标签的权重为2，其余标签的权重为1
    weights = [2.0 if i == 0 else 1.0 for i in range(len(int_labels))]

    # 计算每个标签对的相似性并加权
    for label, weight in zip(int_labels, weights):
        similarity = (label.unsqueeze(1) == label.unsqueeze(0)).float()
        similarity_matrix += weight * similarity

    # 计算对比损失
    positive_pairs = similarity_matrix
    negative_pairs = 1 - positive_pairs

    positive_loss = (positive_pairs * distance_matrix).sum() / (positive_pairs.sum() + epsilon)
    negative_distance = F.relu(margin - distance_matrix)
    negative_loss = (negative_pairs * negative_distance).sum() / (negative_pairs.sum() + epsilon)

    total_loss = positive_loss + negative_loss
    return total_loss, {'positive_loss': positive_loss.item(), 'negative_loss': negative_loss.item()}


def loss_function(x_recon, x, mu, logvar, z, labels,args, new_x_recon=None, new_x=None, beta=0.1, alph=1):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    cont_loss, cont_details = contrastive_loss(z, labels)
    if args["contrast_loss"]:
        total_loss = recon_loss + beta * kl_loss + alph * cont_loss
    else:
        total_loss = recon_loss + beta * kl_loss
    # total_loss = recon_loss
    loss_components = {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "cont_loss": cont_loss.item(),
        "positive_loss": cont_details["positive_loss"],
        "negative_loss": cont_details["negative_loss"],
        "total_loss": total_loss.item()
    }

    if new_x_recon is not None and new_x is not None:
        # Filter out None values and their corresponding elements in new_x_recon
        valid_indices = [i for i, item in enumerate(new_x) if item is not None]
        if valid_indices:
            # Select non-None items from both lists
            valid_new_x = torch.stack([new_x[i] for i in valid_indices])
            valid_new_x_recon = torch.stack([new_x_recon[i] for i in valid_indices])

            # Calculate MSE loss on non-None items
            new_recon_loss = F.mse_loss(valid_new_x_recon, valid_new_x, reduction='sum')
            total_loss += new_recon_loss
            loss_components["new_recon_loss"] = new_recon_loss.item()
            loss_components["total_loss"] += new_recon_loss.item()

    return total_loss, loss_components