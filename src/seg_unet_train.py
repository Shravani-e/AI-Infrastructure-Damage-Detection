import os, argparse, torch, numpy as np
from utils import load_config, set_device, seed_everything, UNetSmall, CrackDataset, list_pairs, bce_dice_loss
from sklearn.model_selection import train_test_split

def main(cfg_path):
    cfg = load_config(cfg_path)
    seed_everything(cfg.get('seed',42))
    device = set_device(cfg.get('device','auto'))
    tcfg = cfg['train']

    ims, mks = list_pairs(tcfg['images_dir'], tcfg['masks_dir'])
    from sklearn.model_selection import train_test_split
    Xtr, Xval, Ytr, Yval = train_test_split(ims, mks, test_size=tcfg.get('val_split',0.2), random_state=42)

    tr_ds = CrackDataset(Xtr, Ytr, size=tcfg['img_size'], aug=True)
    va_ds = CrackDataset(Xval, Yval, size=tcfg['img_size'], aug=False)

    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=tcfg['batch_size'], shuffle=True, num_workers=0)
    va_ld = torch.utils.data.DataLoader(va_ds, batch_size=tcfg['batch_size'], shuffle=False, num_workers=0)

    model = UNetSmall(in_ch=1, out_ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg['lr']))

    best = 1e9
    os.makedirs(tcfg['out_dir'], exist_ok=True)
    for epoch in range(tcfg['epochs']):
        model.train()
        tr_loss=0.0
        for x,y in tr_ld:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            loss.backward(); opt.step()
            tr_loss += float(loss.item())*x.size(0)
        tr_loss /= len(tr_ds)

        model.eval()
        va_loss=0.0
        with torch.no_grad():
            for x,y in va_ld:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                loss = bce_dice_loss(logits, y)
                va_loss += float(loss.item())*x.size(0)
        va_loss /= len(va_ds)

        print(f"Epoch {epoch+1}/{tcfg['epochs']}  train={tr_loss:.4f}  val={va_loss:.4f}")
        if va_loss < best:
            best = va_loss
            torch.save({'model':model.state_dict(), 'cfg':cfg}, os.path.join(tcfg['out_dir'],'best.pt'))
    print("Saved best to", os.path.join(tcfg['out_dir'],'best.pt'))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args.config)
