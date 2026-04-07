import sys, os, glob, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import ColorbarBase
import torch, torch.nn.functional as F

BASE = '/home/aparna.ayyalasomayaj/wildfire_burn_scar_mapping'
sys.path.insert(0, BASE)
from unet_training_v2 import ResUNet  # import model class for checkpoint loading

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load model ────────────────────────────────────────────────────────────────
ckpt  = f'{BASE}/outputs/sentinel2_results_v2/resunet_best_model.pth'
model = ResUNet(input_channels=7, num_classes=4, f=32, drop=0.3).to(DEVICE)
ckpt_data = torch.load(ckpt, map_location=DEVICE)
sd = ckpt_data['model'] if isinstance(ckpt_data, dict) and 'model' in ckpt_data else ckpt_data
model.load_state_dict(sd)
model.eval()

# ── Channel stats ─────────────────────────────────────────────────────────────
stats = np.load(f'{BASE}/outputs/sentinel2_results_v2/channel_stats.npz')
mean  = stats['mean']
std   = stats['std']

# ── Select test patches (one OR, one NM — best burned coverage) ───────────────
with open(f'{BASE}/data/sentinel2/splits_v2.json') as f:
    splits = json.load(f)

all_patches = glob.glob(f'{BASE}/data/sentinel2/patches_v2/**/*.npz', recursive=True)

def best_patch(fire_id):
    """Select the patch with the highest burned-pixel fraction for a given fire.
    Using the most burned patch ensures Figure 1 shows clear severity gradients
    rather than a mostly-unburned patch where predictions are trivially correct.
    Limits to first 40 candidates to keep selection fast.
    """
    cands = [p for p in all_patches if fire_id in p][:40]
    best, best_b = None, 0
    for p in cands:
        d = np.load(p, allow_pickle=True)
        b = (d['y'] > 0).mean()   # fraction of pixels with class > 0 (burned)
        if b > best_b:
            best_b, best = b, p
    return best

fp_or = best_patch('OR4372612216720220801')    # Oregon 2022
fp_nm = best_patch('NM3571810539920220406')   # New Mexico 2022

# ── Colour setup ──────────────────────────────────────────────────────────────
SEV_COLORS = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
SEV_NAMES  = ['Unburned/Low', 'Moderate', 'High', 'Very High']
cmap_sev   = mcolors.ListedColormap(SEV_COLORS)
norm_sev   = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_sev.N)
cmap_err   = mcolors.ListedColormap(['#e74c3c', '#27ae60'])

# ── Process patch ─────────────────────────────────────────────────────────────
def process(fp):
    data  = np.load(fp, allow_pickle=True)
    X_raw = data['X'].astype(np.float32)
    y     = data['y'].astype(np.int64)
    rgb   = X_raw[:3].copy()
    for c in range(3):
        rgb[c] = rgb[c] * std[c] + mean[c]
    rgb = np.stack([rgb[0], rgb[1], rgb[2]])
    pre_rgb = np.clip(
        (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8), 0, 1
    ).transpose(1, 2, 0)
    dnbr = X_raw[6]
    Xt   = torch.from_numpy(X_raw).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(Xt), dim=1).squeeze(0).cpu().numpy()
    pred  = probs.argmax(0)
    conf  = probs.max(0)
    error = (pred == y).astype(np.uint8)
    return pre_rgb, dnbr, y, pred, error, conf, error.mean() * 100

fires = [
    ('Fire A — Oregon (OR4372, Aug 2022)',        *process(fp_or)),
    ('Fire B — New Mexico (NM3571, Apr 2022)',    *process(fp_nm)),
]

# ── Figure layout ─────────────────────────────────────────────────────────────
# 3 rows × 6 cols: [label | f1_rgb | f1_dnbr | f2_rgb | f2_dnbr | cbar]
fig = plt.figure(figsize=(10.0, 8.5), facecolor='white', dpi=200)
gs  = GridSpec(3, 6, figure=fig,
               left=0.07, right=0.94, top=0.91, bottom=0.11,
               hspace=0.05, wspace=0.04,
               width_ratios=[0.16, 1, 1, 1, 1, 0.07])

ROW_TITLES = ['Row 1\nInput', 'Row 2\nSeverity', 'Row 3\nError Analysis']
ROW_SUBS   = [['(a) True-colour RGB', '(b) dNBR change index'],
               ['(c) Ground truth (MTBS)', '(d) ResUNet prediction'],
               ['(e) Pixel-wise error map', '(f) Softmax confidence']]

for row in range(3):
    for fi, (fire_label, rgb, dnbr, gt, pred, err, conf, acc) in enumerate(fires):
        bc = 1 + fi * 2   # base column

        # Row label (only for first fire)
        if fi == 0:
            ax_l = fig.add_subplot(gs[row, 0])
            ax_l.text(0.5, 0.5, ROW_TITLES[row],
                      ha='center', va='center', fontsize=8,
                      fontweight='bold', color='#2c3e50',
                      transform=ax_l.transAxes)
            ax_l.axis('off')

        ax0 = fig.add_subplot(gs[row, bc])
        ax1 = fig.add_subplot(gs[row, bc + 1])

        if row == 0:
            ax0.imshow(rgb)
            im_d = ax1.imshow(dnbr, cmap='RdYlGn_r', vmin=-0.3, vmax=0.8)
            if fi == 1:
                _im_dnbr = im_d
        elif row == 1:
            ax0.imshow(gt,   cmap=cmap_sev, norm=norm_sev, interpolation='nearest')
            ax1.imshow(pred, cmap=cmap_sev, norm=norm_sev, interpolation='nearest')
        else:
            ax0.imshow(err,  cmap=cmap_err, vmin=0, vmax=1, interpolation='nearest')
            im_c = ax1.imshow(conf, cmap='plasma', vmin=0, vmax=1)
            ax0.set_title(f'Pixel accuracy = {acc:.1f}%',
                          fontsize=7, color='#333', pad=2)
            if fi == 1:
                _im_conf = im_c

        for ax, lbl in [(ax0, ROW_SUBS[row][0]), (ax1, ROW_SUBS[row][1])]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color('#bbbbbb')
            ax.text(0.02, 0.03, lbl, transform=ax.transAxes,
                    fontsize=6.5, color='white', fontweight='bold',
                    bbox=dict(facecolor='#1a1a2e', alpha=0.72,
                              pad=1.5, linewidth=0))

# Fire column headers
for fi, (fire_label, *_) in enumerate(fires):
    # Centre of the two sub-columns for this fire
    x_left  = gs.get_subplot_params().left
    x_right = gs.get_subplot_params().right
    total_w = x_right - x_left
    col_w   = total_w / 6     # 6 columns including label+cbar
    cx = x_left + (1 + fi * 2 + 1.0) * col_w   # centre of the pair
    fig.text(cx, gs.get_subplot_params().top + 0.012,
             fire_label, ha='center', va='bottom',
             fontsize=9, fontweight='bold', color='#1a1a2e')

# ── Colorbars ─────────────────────────────────────────────────────────────────
ax_cb_d = fig.add_subplot(gs[0, 5])
cb_d = ColorbarBase(ax_cb_d, cmap=plt.cm.get_cmap('RdYlGn_r'),
                     norm=mcolors.Normalize(-0.3, 0.8), orientation='vertical')
cb_d.set_label('dNBR', fontsize=6.5)
cb_d.set_ticks([-0.3, 0, 0.3, 0.6, 0.8])
cb_d.ax.tick_params(labelsize=5.5)

ax_cb_c = fig.add_subplot(gs[2, 5])
cb_c = ColorbarBase(ax_cb_c, cmap=plt.cm.plasma,
                     norm=mcolors.Normalize(0, 1), orientation='vertical')
cb_c.set_label('Confidence', fontsize=6.5)
cb_c.ax.tick_params(labelsize=5.5)

# ── Severity legend ───────────────────────────────────────────────────────────
handles = [mpatches.Patch(facecolor=SEV_COLORS[i], label=SEV_NAMES[i],
                           edgecolor='grey', linewidth=0.5)
           for i in range(4)]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8,
           frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.005),
           title='Severity legend (Rows 2 & 3)', title_fontsize=7.5)

out = (f'{BASE}/outputs/sentinel2_results_v2/figures/'
       f'resunet_severity_overview_v2.png')
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')

from PIL import Image
img = Image.open(out)
w, h = img.size
print(f'Dimensions: {w}x{h} px  |  at 3.39" wide → {h/(w/3.39):.2f}" tall')
