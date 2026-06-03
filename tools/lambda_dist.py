import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def rand_bbox(H, W, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cx = np.random.randint(H)
    cy = np.random.randint(W)
    bbx1 = int(np.clip(cx - cut_h // 2, 0, H))
    bbx2 = int(np.clip(cx + cut_h // 2, 0, H))
    bby1 = int(np.clip(cy - cut_w // 2, 0, W))
    bby2 = int(np.clip(cy + cut_w // 2, 0, W))
    return bbx1, bbx2, bby1, bby2


def simulate_lam_area(H, W, N=100_000, alpha=1.0):
    lam_sampled = np.random.beta(alpha, alpha, N)
    lam_area = np.empty(N)
    for i, lam in enumerate(tqdm(lam_sampled)):
        bbx1, bbx2, bby1, bby2 = rand_bbox(H, W, lam)
        cut_area = (bbx2 - bbx1) * (bby2 - bby1)
        lam_area[i] = 1.0 - cut_area / (H * W)
    return lam_sampled, lam_area


N = 1_000_000
configs = [
    (64, 64, "64×64 (TinyImageNet)"),
    (224, 224, "224×224 (ImageNet)"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("CutMix λ distributions: sampled vs actual area ratio", fontsize=14)

for row, (H, W, title) in enumerate(configs):
    lam_sampled, lam_area = simulate_lam_area(H, W, N)
    delta = np.abs(lam_area - lam_sampled)

    ax0, ax1, ax2 = axes[row]

    # Panel 1: lam_sampled
    ax0.hist(lam_sampled, bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
    ax0.axvline(np.mean(lam_sampled), color="k", linestyle="--", label=f"mean={np.mean(lam_sampled):.3f}")
    ax0.axvline(np.median(lam_sampled), color="orange", linestyle=":", label=f"median={np.median(lam_sampled):.3f}")
    ax0.set_title(f"{title}\nλ_sampled  (Beta(1,1))")
    ax0.set_xlabel("λ_sampled")
    ax0.set_ylabel("Count")
    ax0.legend(fontsize=8)

    # Panel 2: lam_area
    ax1.hist(lam_area, bins=40, color="seagreen", edgecolor="white", linewidth=0.3)
    ax1.axvline(np.mean(lam_area), color="k", linestyle="--", label=f"mean={np.mean(lam_area):.3f}")
    ax1.axvline(np.median(lam_area), color="orange", linestyle=":", label=f"median={np.median(lam_area):.3f}")
    ax1.axvline(0.7795, color="red", linestyle="-.", linewidth=1.2, label="theory mean=0.780")
    ax1.set_title(f"{title}\nλ_area  (actual bbox area)")
    ax1.set_xlabel("λ_area")
    ax1.legend(fontsize=8)

    # Panel 3: |lam_area - lam_sampled|
    ax2.hist(delta, bins=40, color="salmon", edgecolor="white", linewidth=0.3)
    ax2.axvline(np.mean(delta), color="k", linestyle="--", label=f"mean={np.mean(delta):.3f}")
    ax2.axvline(np.median(delta), color="orange", linestyle=":", label=f"median={np.median(delta):.3f}")
    ax2.set_title(f"{title}\n|λ_area − λ_sampled|")
    ax2.set_xlabel("|λ_area − λ_sampled|")
    ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plots/lambda_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to lambda_distribution.png")
