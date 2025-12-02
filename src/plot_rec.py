import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from train import LSTMAutoencoder


# ---------------------------------------------------------
# Load trained AE model
# ---------------------------------------------------------
def load_model(model_path, input_dim=13, hidden_dim=128, latent_dim=32, seq_len=30, device="cuda"):
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        seq_len=seq_len
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# ---------------------------------------------------------
# Extract latent vectors for ALL windows
# ---------------------------------------------------------
def extract_latents(model, windows, device="cuda"):
    """
    windows: (N,30,47)
    returns: (N, latent_dim)
    """
    model.eval()

    latents = []
    with torch.no_grad():
        for w in windows:
            w_t = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device)
            z = model.encoder(w_t)          # (1, latent_dim)
            latents.append(z.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    return latents    # (N, latent_dim)


# ---------------------------------------------------------
# PCA Visualization
# ---------------------------------------------------------
def plot_pca(latents, video_ids):
    pca = PCA(n_components=2)
    z2d = pca.fit_transform(latents)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(z2d[:, 0], z2d[:, 1], c=video_ids, cmap="viridis", s=12)
    plt.colorbar(scatter, label="Video ID")
    plt.title("PCA of Latent Space (AE)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# t-SNE Visualization
# ---------------------------------------------------------
def plot_tsne(latents, video_ids):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    z2d = tsne.fit_transform(latents)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(z2d[:, 0], z2d[:, 1], c=video_ids, cmap="plasma", s=12)
    plt.colorbar(scatter, label="Video ID")
    plt.title("t-SNE of Latent Space (AE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# MAIN: Run Visualization
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load dataset windows
    npz_path = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX\data\train\processed\2d_data.npz"
    data = np.load(npz_path, allow_pickle=True)

    windows = data["windows"]      # (N,30,47)
    video_ids = data["video_ids"]  # (N,)
    print("Loaded windows:", windows.shape)

    # Load trained AE model
    model_path = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX\models\lstm_ae_final.pth"
    model = load_model(model_path)

    # Extract latent vectors
    latents = extract_latents(model, windows)
    print("Latents shape:", latents.shape)

    # PCA visualization
    plot_pca(latents, video_ids)

    # t-SNE visualization
    plot_tsne(latents, video_ids)
