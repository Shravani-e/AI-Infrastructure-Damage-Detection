import os, cv2, numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter

root = os.path.dirname(__file__)
images = os.path.join(root, "images")
masks  = os.path.join(root, "masks")
os.makedirs(images, exist_ok=True)
os.makedirs(masks, exist_ok=True)

def make_img(h=512, w=768, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(180, 10, size=(h,w)).astype(np.float32)
    base = gaussian_filter(base, 1.0)
    for _ in range(2000):
        y = rng.integers(0,h); x=rng.integers(0,w)
        base[y,x] -= rng.uniform(0,20)
    base = np.clip(base,0,255)

    y = int(h*0.4)
    crack = np.zeros((h,w), np.uint8)
    dy = 0
    for x in range(30, w-30):
        dy += rng.integers(-1,2); dy = int(np.clip(dy,-2,2))
        y = int(np.clip(y+dy, 10, h-10))
        width = int(1 + 6*(0.5 + 0.5*np.sin(x/60)))
        cv2.circle(crack, (x,y), width//2, 255, -1)

    img = base.copy()
    img[crack>0] -= 120
    img = np.clip(img,0,255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3,3), 0)
    return img, crack

rows = []
for i in range(30):
    img, m = make_img(seed=i)
    fn = f"demo_{i:03d}.png"
    cv2.imwrite(os.path.join(images, fn), img)
    cv2.imwrite(os.path.join(masks, fn), m)
    dist = cv2.distanceTransform((m>0).astype(np.uint8), cv2.DIST_L2, 3)
    w95_px = np.percentile(dist[m>0]*2.0, 95)
    length_px = (m>0).sum()/max(1, int(np.mean(dist[m>0]*2.0)))
    w95_mm = w95_px*0.2
    length_mm = length_px*0.2
    depth = np.clip(12*(0.6*(w95_mm**1.2) + 0.002*np.sqrt(max(length_mm,1)))+np.random.normal(0,0.5), 0, 30)
    rows.append(dict(image=fn, depth_mm=float(depth)))

pd.DataFrame(rows).to_csv(os.path.join(root,'nde_groundtruth.csv'), index=False)
print("Synthetic data written to", images, "and", masks)
print("NDE CSV at", os.path.join(root,'nde_groundtruth.csv'))
