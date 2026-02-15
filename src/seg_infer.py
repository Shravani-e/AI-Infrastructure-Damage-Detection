import os, argparse, cv2, torch, numpy as np
from utils import load_config, set_device, UNetSmall

def sigmoid(x): return 1/(1+np.exp(-x))

def main(cfg_path, images_dir, out_masks):
    cfg = load_config(cfg_path)
    device = set_device(cfg.get('device','auto'))
    os.makedirs(out_masks, exist_ok=True)

    ckpt = torch.load(cfg['infer']['weights'], map_location=device)
    model = UNetSmall(in_ch=1, out_ch=1).to(device)
    model.load_state_dict(ckpt['model']); model.eval()

    size = cfg['train']['img_size']
    thresh = float(cfg['infer']['thresh'])

    for fn in sorted(os.listdir(images_dir)):
        if not fn.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff')):
            continue
        path = os.path.join(images_dir, fn)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h,w = img.shape[:2]
        img_r = cv2.resize(img, (size,size))
        inp = (img_r[...,None]/255.0 - 0.5)/0.5
        inp = np.transpose(inp, (2,0,1))[None].astype(np.float32)
        x = torch.from_numpy(inp).to(device)
        with torch.no_grad():
            logits = model(x).cpu().numpy()[0,0]
        prob = sigmoid(logits)
        m = (prob > thresh).astype(np.uint8)*255
        m = cv2.resize(m, (w,h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_masks, fn), m)
        print("Saved mask:", os.path.join(out_masks, fn))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--out_masks', required=True)
    args = ap.parse_args()
    main(args.config, args.images, args.out_masks)
