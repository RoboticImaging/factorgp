import numpy as np
from skimage.metrics import structural_similarity as _ssim_fn


def compute_gradient_rmse(pred_img, gt_img):
    gx_pr, gy_pr = np.gradient(pred_img)
    gx_gt, gy_gt = np.gradient(gt_img)
    pred_mag = np.sqrt(gx_pr ** 2 + gy_pr ** 2)
    gt_mag   = np.sqrt(gx_gt ** 2 + gy_gt ** 2)
    return float(np.sqrt(np.mean((pred_mag - gt_mag) ** 2)))


def compute_ssim(pred_img, gt_img, data_range, win_size=None):
    kwargs = dict(data_range=data_range)
    if win_size is not None:
        kwargs['win_size'] = win_size
    return float(_ssim_fn(gt_img, pred_img, **kwargs))


def print_results_table(rows, title="Final metrics"):
    """Print a formatted results table.

    Parameters
    ----------
    rows : list of dict, each with keys: label, rmse, smse, grmse, ssim3
    title : str
    """
    cw = 8
    nw = max(14, max(len(r['label']) for r in rows) + 2)
    hdr = f"  {'Model':<{nw}} {'RMSE':>{cw}} {'SMSE':>{cw}} {'GRMSE':>{cw}} {'SSIM3':>{cw}}"
    sep = "  " + "─" * (len(hdr) - 2)
    bar = "═" * len(hdr)
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    print(hdr)
    print(sep)
    for r in rows:
        print(f"  {r['label']:<{nw}} "
              f"{r['rmse']:>{cw}.4f} "
              f"{r['smse']:>{cw}.4f} "
              f"{r['grmse']:>{cw}.4f} "
              f"{r['ssim3']:>{cw}.4f}")
    print()
