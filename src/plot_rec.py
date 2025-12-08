import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib  # [필수 추가] 스케일러 로드용
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from train import LSTMAutoencoder


# ---------------------------------------------------------
# 1. Load trained AE model
# ---------------------------------------------------------
def load_model(model_path, input_dim=13, hidden_dim=256, latent_dim=64, seq_len=32, device="cuda"):
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        seq_len=seq_len
    ).to(device)

    map_location = device if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.eval()
    return model


# ---------------------------------------------------------
# 2. Extract latent vectors (With Scaling!)
# ---------------------------------------------------------
def extract_latents(model, windows, scaler_path, device="cuda"):
    """
    [수정됨] 스케일러를 로드하여 데이터를 변환한 뒤 Latent 추출
    """
    # 1. 스케일러 로드 및 적용
    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)

    # 3차원 -> 2차원 펼치기 -> 스케일링 -> 3차원 복구
    N, T, F = windows.shape
    windows_flat = windows.reshape(-1, F)
    windows_scaled_flat = scaler.transform(windows_flat)  # Transform!
    windows_scaled = windows_scaled_flat.reshape(N, T, F)

    print("Data scaled successfully.")

    model.eval()
    latents = []

    # 배치 단위 처리 (메모리 보호)
    batch_size = 256
    with torch.no_grad():
        for i in range(0, N, batch_size):
            # 스케일링 된 데이터를 모델에 입력
            batch = windows_scaled[i: i + batch_size]
            batch_t = torch.tensor(batch, dtype=torch.float32).to(device)

            z = model.encoder(batch_t)  # (Batch, latent_dim)
            latents.append(z.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    return latents


# ---------------------------------------------------------
# 3. Calculate Relative Progress (0.0 ~ 1.0)
# ---------------------------------------------------------
def calculate_progress(video_ids):
    progress = np.zeros_like(video_ids, dtype=float)
    unique_ids = np.unique(video_ids)

    print(f"Calculating progress for {len(unique_ids)} videos...")

    for vid in unique_ids:
        idxs = np.where(video_ids == vid)[0]
        count = len(idxs)

        if count > 1:
            prog = np.linspace(0, 1, count)
        else:
            prog = np.array([0.0])

        progress[idxs] = prog

    return progress


# ---------------------------------------------------------
# 4. Visualization Functions
# ---------------------------------------------------------
def plot_pca(latents, progress):
    pca = PCA(n_components=2)
    z2d = pca.fit_transform(latents)

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(z2d[:, 0], z2d[:, 1], c=progress, cmap="turbo", s=10, alpha=0.7)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Motion Progress (0: Start -> 1: End)")

    plt.title("PCA of Latent Space (Scaled)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_tsne(latents, progress):
    tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, n_iter=1000, init='pca')
    z2d = tsne.fit_transform(latents)

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(z2d[:, 0], z2d[:, 1], c=progress, cmap="turbo", s=10, alpha=0.7)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Motion Progress (0: Start -> 1: End)")

    plt.title("t-SNE of Latent Space (Scaled)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # 설정
    INPUT_DIM = 13
    SEQ_LEN = 32
    HIDDEN_DIM = 256
    LATENT_DIM = 64

    # 경로 확인 (train.py에서 scaler가 저장된 경로여야 함)
    DATA_PATH = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX\data\train\processed\2d_data.npz"
    MODEL_PATH = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX\models\lstm_ae_final.pth"
    SCALER_PATH = r"C:\Users\Yul\PycharmProjects\PitcherKinetiX\models\scaler.pkl"  # [필수]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 데이터 로드
    data = np.load(DATA_PATH, allow_pickle=True)
    windows = data["windows"]
    video_ids = data["video_ids"]
    print(f"Loaded windows: {windows.shape}")

    # 모델 로드
    model = load_model(MODEL_PATH,
                       input_dim=INPUT_DIM,
                       hidden_dim=HIDDEN_DIM,
                       latent_dim=LATENT_DIM,
                       seq_len=SEQ_LEN,
                       device=device)

    # [수정됨] 스케일러 경로 전달
    latents = extract_latents(model, windows, SCALER_PATH, device=device)
    print(f"Latents extracted: {latents.shape}")

    # 진행률 계산
    progress = calculate_progress(video_ids)

    # 시각화
    print("Plotting PCA...")
    plot_pca(latents, progress)

    print("Plotting t-SNE...")
    plot_tsne(latents, progress)