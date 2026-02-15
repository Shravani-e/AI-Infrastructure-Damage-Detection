import os, argparse, cv2, json, math, numpy as np, pandas as pd
from skimage.morphology import skeletonize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import yaml

def pixel_to_mm(p, gsd): return p*gsd

def width_stats_mm(mask_bin, gsd):
    dist = cv2.distanceTransform((mask_bin>0).astype(np.uint8), cv2.DIST_L2, 3)
    vals = (dist[mask_bin>0]*2.0)*gsd
    if len(vals)==0: return 0.0,0.0,0.0
    return np.percentile(vals,50), np.percentile(vals,95), np.max(vals)

def skeleton_length_mm(mask_bin, gsd):
    sk = skeletonize((mask_bin>0).astype(bool))
    return pixel_to_mm(sk.sum(), gsd)

def area_mm2(mask_bin, gsd):
    return (mask_bin>0).sum()*(gsd**2)

def classify(depth, w95, length, thr):
    if depth>thr['severe']['depth_mm_gt'] or w95>thr['severe']['width_mm_gt'] or length>thr['severe']['length_mm_gt']:
        return "Severe"
    if depth>=thr['moderate']['depth_mm_range'][0] or w95>=thr['moderate']['width_mm_range'][0] or length>thr['moderate']['length_mm_gt']:
        return "Moderate"
    return "Minor"

def repair_plan(sev):
    if sev=="Severe":
        return ["Engineer review", "Epoxy injection (structural)", "Staple/dowel stitching", "Section repair/jacketing/FRP", "Mitigate load/moisture"]
    if sev=="Moderate":
        return ["Epoxy injection (medium pressure)", "Rout-and-seal; localized patch", "Address moisture; re-inspect"]
    return ["Surface sealant / low-pressure epoxy", "Clean & monitor"]

def main(cfg_path, images_dir, masks_dir, nde_csv, out_dir):
    cfg = yaml.safe_load(open(cfg_path,'r'))
    gsd = float(cfg['gsd_mm_per_px'])
    os.makedirs(out_dir, exist_ok=True)

    pairs = []
    for fn in sorted(os.listdir(images_dir)):
        if not fn.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff')): continue
        imgp = os.path.join(images_dir, fn)
        mskp = os.path.join(masks_dir, fn)
        if os.path.exists(mskp):
            pairs.append((fn, imgp, mskp))

    feats = []
    for fn, imgp, mskp in pairs:
        m = cv2.imread(mskp, cv2.IMREAD_GRAYSCALE)
        length = skeleton_length_mm(m, gsd)
        w50, w95, wmax = width_stats_mm(m, gsd)
        area = area_mm2(m, gsd)
        feats.append(dict(image=fn, length_mm=length, width_p50_mm=w50, width_p95_mm=w95, width_max_mm=wmax, area_mm2=area))
    df = pd.DataFrame(feats)

    nde = pd.read_csv(nde_csv)
    df_train = df.merge(nde, on='image', how='inner')

    X = df_train[['width_p95_mm','length_mm','area_mm2']].values
    y = df_train['depth_mm'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    mae = float(mean_absolute_error(yte, pred))
    r2  = float(r2_score(yte, pred))

    df['pred_depth_mm'] = rf.predict(df[['width_p95_mm','length_mm','area_mm2']].values)
    df['severity'] = df.apply(lambda r: classify(r['pred_depth_mm'], r['width_p95_mm'], r['length_mm'], cfg['severity']), axis=1)
    df['repairs'] = df['severity'].map(lambda s: "; ".join(repair_plan(s)))
    df['model_test_MAE_mm'] = mae
    df['model_test_R2'] = r2

    out_csv = os.path.join(out_dir, 'results.csv')
    df.to_csv(out_csv, index=False)

    for _,row in df.iterrows():
        img = cv2.imread(os.path.join(images_dir, row['image']), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(os.path.join(masks_dir, row['image']), cv2.IMREAD_GRAYSCALE)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        contours,_ = cv2.findContours((m>0).astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255,255,255), 1)
        y0=30
        for t in [
            f"Length: {row['length_mm']:.1f} mm",
            f"Width p95: {row['width_p95_mm']:.3f} mm (max {row['width_max_mm']:.3f} mm)",
            f"Area: {row['area_mm2']:.1f} mm^2",
            f"Pred depth: {row['pred_depth_mm']:.1f} mm",
            f"Severity: {row['severity']}",
        ]:
            cv2.putText(overlay, t, (20,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            y0+=28
        outp = os.path.join(out_dir, f"annot_{row['image']}")
        cv2.imwrite(outp, overlay)

    print("Saved:", out_csv, "and annotated images in", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--nde_csv', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    main(args.config, args.images, args.masks, args.nde_csv, args.out)
