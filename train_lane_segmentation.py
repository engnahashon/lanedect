#!/usr/bin/env python3
"""
Reproducible binary lane-marking segmentation baseline for TuSimple and CULane.

Important:
- This script trains on real image/annotation files when those datasets are supplied.
- It reports pixel-mask metrics (precision, recall, F1, IoU), NOT the official
  TuSimple accuracy or official CULane lane-instance F1.
- Official benchmark scores require benchmark-specific prediction conversion and
  evaluation using the official evaluators.
- For TuSimple, the official training labels are the label_data_*.json files.
- For CULane, images are listed in list/train.txt or list/train_gt.txt and lane
  points are read from the corresponding .lines.txt annotation files.
"""
import argparse, json, random, time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def resolve_path(root, raw):
    p = Path(raw.strip())
    if p.is_absolute() and p.exists(): return p
    return (root / p.as_posix().lstrip("/")).resolve()

def parse_tusimple(root):
    files = sorted(root.glob("label_data*.json"))
    if not files:
        # Permit a common nested layout.
        files = sorted(root.rglob("label_data*.json"))
    records = []
    for jf in files:
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                raw = obj.get("raw_file", "")
                image = resolve_path(root, raw)
                if not image.exists():
                    # Some archives put clips/ directly beneath a nested root.
                    matches = list(root.rglob(Path(raw).name))
                    if matches: image = matches[0]
                if image.exists():
                    records.append({"image": image, "lanes": obj.get("lanes", []),
                                    "h_samples": obj.get("h_samples", []),
                                    "source": raw})
    if not records:
        raise FileNotFoundError(
            f"No TuSimple labelled images found under {root}. Expected label_data*.json and clips/."
        )
    return records

def read_culane_list(root, split, explicit=None):
    candidates = [Path(explicit)] if explicit else [
        root / "list" / f"{split}.txt",
        root / "list" / f"{split}_gt.txt",
        root / "lists" / f"{split}.txt",
        root / "lists" / f"{split}_gt.txt",
    ]
    listfile = next((p if p.is_absolute() else (Path.cwd()/p) for p in candidates if p.exists()), None)
    if listfile is None:
        raise FileNotFoundError(
            f"Could not find CULane {split} list. Tried: " + ", ".join(map(str, candidates)) +
            ". Supply --list-file or check the extracted CULane list directory."
        )
    records = []
    for rawline in listfile.read_text(encoding="utf-8", errors="replace").splitlines():
        line = rawline.strip()
        if not line or line.startswith("#"): continue
        toks = line.split()
        image_rel = toks[0]
        image = resolve_path(root, image_rel)
        if not image.exists():
            continue
        # If a list line explicitly names a .lines.txt file, use it.
        ann = None
        for tok in toks[1:]:
            if ".lines.txt" in tok:
                candidate = resolve_path(root, tok)
                if candidate.exists(): ann = candidate; break
        if ann is None:
            ann = image.with_suffix(".lines.txt")
            if not ann.exists():
                # Some copies store annotations under an annotations/ tree.
                matches = list(root.rglob(image.stem + ".lines.txt"))
                ann = matches[0] if matches else ann
        if ann.exists():
            records.append({"image": image, "annotation": ann, "source": image_rel})
    if not records:
        raise FileNotFoundError(
            f"No CULane image/lines pairs resolved from {listfile}. Check root paths and .lines.txt annotations."
        )
    return records

def draw_lane_mask(record, dataset, width, height, orig_w, orig_h):
    mask = np.zeros((height, width), dtype=np.uint8)
    if dataset == "tusimple":
        xs_list, ys = record["lanes"], record["h_samples"]
        for xs in xs_list:
            pts = []
            for x, y in zip(xs, ys):
                if x is None or x < 0: continue
                xx = int(round(float(x) * width / orig_w))
                yy = int(round(float(y) * height / orig_h))
                if 0 <= xx < width and 0 <= yy < height: pts.append((xx, yy))
            if len(pts) >= 2:
                cv2.polylines(mask, [np.asarray(pts, np.int32)], False, 1, thickness=5)
    else:
        text = record["annotation"].read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            vals = line.strip().split()
            if len(vals) < 4: continue
            try: nums = [float(v) for v in vals]
            except ValueError: continue
            pts = []
            for x, y in zip(nums[0::2], nums[1::2]):
                xx = int(round(x * width / orig_w))
                yy = int(round(y * height / orig_h))
                if 0 <= xx < width and 0 <= yy < height: pts.append((xx, yy))
            if len(pts) >= 2:
                cv2.polylines(mask, [np.asarray(pts, np.int32)], False, 1, thickness=5)
    return mask

class LaneDataset(Dataset):
    def __init__(self, records, dataset, size=(320, 800), orig_size=None, augment=False):
        self.records, self.dataset = records, dataset
        self.height, self.width = size
        self.augment = augment
        if orig_size is None:
            self.orig_w, self.orig_h = ((1280, 720) if dataset == "tusimple" else (1640, 590))
        else: self.orig_w, self.orig_h = orig_size
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        img = cv2.imread(str(r["image"]))
        if img is None: raise RuntimeError(f"Could not read image: {r['image']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        # Use actual image dimensions to map annotation coordinates.
        mask = draw_lane_mask(r, self.dataset, self.width, self.height, w0, h0)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if self.augment and random.random() < .5:
            img = cv2.convertScaleAbs(img, alpha=random.uniform(.65, 1.25), beta=random.randint(-20, 20))
        if self.augment and random.random() < .25:
            noise = np.random.normal(0, random.uniform(1, 8), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        x = torch.from_numpy(img.astype(np.float32).transpose(2,0,1) / 255.0)
        y = torch.from_numpy(mask[None].astype(np.float32))
        return x, y, str(r["image"])

def conv_block(a, b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                         nn.Conv2d(b,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
class SmallUNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1=conv_block(3,base); self.p1=nn.MaxPool2d(2)
        self.e2=conv_block(base,base*2); self.p2=nn.MaxPool2d(2)
        self.e3=conv_block(base*2,base*4); self.p3=nn.MaxPool2d(2)
        self.mid=conv_block(base*4,base*8)
        self.u3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=conv_block(base*8,base*4)
        self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=conv_block(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=conv_block(base*2,base)
        self.out=nn.Conv2d(base,1,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.p1(a)); c=self.e3(self.p2(b)); m=self.mid(self.p3(c))
        z=self.d3(torch.cat([self.u3(m),c],1))
        z=self.d2(torch.cat([self.u2(z),b],1))
        z=self.d1(torch.cat([self.u1(z),a],1))
        return self.out(z)

def calc_metrics(tp, fp, fn, tn):
    eps=1e-9
    p=tp/(tp+fp+eps); r=tp/(tp+fn+eps)
    f1=2*p*r/(p+r+eps); iou=tp/(tp+fp+fn+eps)
    return {"precision":p,"recall":r,"f1":f1,"iou":iou,
            "tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn)}

@torch.no_grad()
def evaluate(model, loader, device, threshold=.5, save_dir=None, max_visuals=24):
    model.eval(); totals=np.zeros(4,dtype=np.int64); per_image=[]
    if save_dir: Path(save_dir).mkdir(parents=True, exist_ok=True)
    saved=0
    for x,y,paths in tqdm(loader, desc="evaluate", leave=False):
        x=x.to(device); y=y.to(device)
        probs=torch.sigmoid(model(x))
        pred=(probs>=threshold); gt=(y>=.5)
        tp=(pred&gt).sum().item(); fp=(pred&~gt).sum().item()
        fn=(~pred&gt).sum().item(); tn=(~pred&~gt).sum().item()
        totals += np.array([tp,fp,fn,tn],dtype=np.int64)
        for j,path in enumerate(paths):
            pj=pred[j,0].cpu().numpy(); gj=gt[j,0].cpu().numpy()
            itp=np.logical_and(pj,gj).sum(); ifp=np.logical_and(pj,~gj).sum()
            ifn=np.logical_and(~pj,gj).sum(); itn=np.logical_and(~pj,~gj).sum()
            m=calc_metrics(itp,ifp,ifn,itn); m["image"]=path; per_image.append(m)
            if save_dir and saved < max_visuals:
                rgb=(x[j].detach().cpu().numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
                overlay=rgb.copy()
                overlay[gj] = (0.55*overlay[gj] + 0.45*np.array([0,255,0])).astype(np.uint8)
                overlay[pj & ~gj] = (0.55*overlay[pj & ~gj] + 0.45*np.array([255,0,0])).astype(np.uint8)
                Image.fromarray(overlay).save(Path(save_dir)/f"prediction_{saved:03d}.png")
                saved += 1
    m=calc_metrics(*totals)
    return m, per_image

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tusimple","culane"], required=True)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--list-file", default=None, help="Optional CULane list file")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=.1)
    ap.add_argument("--threshold", type=float, default=.5)
    ap.add_argument("--max-visuals", type=int, default=24)
    args=ap.parse_args()
    seed_everything(args.seed); args.output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset=="tusimple":
        records=parse_tusimple(args.data_root)
        # Deterministic image-level split. Grouping by source clip is preferred when clip IDs are available.
        rng=random.Random(args.seed); rng.shuffle(records)
        nval=max(1,int(len(records)*args.val_fraction))
        val_records, train_records=records[:nval],records[nval:]
    else:
        train_records=read_culane_list(args.data_root,"train",args.list_file)
        try: val_records=read_culane_list(args.data_root,"val")
        except FileNotFoundError:
            rng=random.Random(args.seed); rng.shuffle(train_records)
            nval=max(1,int(len(train_records)*args.val_fraction))
            val_records,train_records=train_records[:nval],train_records[nval:]
    if len(train_records)<2 or not val_records: raise RuntimeError("Need at least 2 training records and 1 validation record.")
    print(f"device={device}; dataset={args.dataset}; train={len(train_records)}; validation={len(val_records)}")
    train_ds=LaneDataset(train_records,args.dataset,(args.height,args.width),augment=True)
    val_ds=LaneDataset(val_records,args.dataset,(args.height,args.width),augment=False)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=(device.type=="cuda"))
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=(device.type=="cuda"))
    model=SmallUNet().to(device)
    # BCE is suitable for binary masks; positive-class weighting addresses sparse lane pixels.
    pos_weight=torch.tensor([8.0],device=device)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda"))
    history=[]; best_f1=-1; best_path=args.output_dir/"best_model.pt"
    start=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); loss_sum=0; seen=0
        for x,y,_ in tqdm(train_loader,desc=f"epoch {epoch}/{args.epochs}",leave=False):
            x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=(device.type=="cuda")):
                logits=model(x); loss=criterion(logits,y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += loss.item()*x.size(0); seen += x.size(0)
        val_metrics, _=evaluate(model,val_loader,device,args.threshold)
        row={"epoch":epoch,"train_loss":loss_sum/max(seen,1),**{f"val_{k}":v for k,v in val_metrics.items() if isinstance(v,(int,float))}}
        history.append(row); print(json.dumps(row))
        with (args.output_dir/"history.json").open("w") as f: json.dump(history,f,indent=2)
        if val_metrics["f1"]>best_f1:
            best_f1=val_metrics["f1"]
            torch.save({"model_state_dict":model.state_dict(),"dataset":args.dataset,
                        "height":args.height,"width":args.width,"threshold":args.threshold,
                        "seed":args.seed,"epoch":epoch,"validation_metrics":val_metrics},best_path)
    ckpt=torch.load(best_path,map_location=device,weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final_metrics, per_image=evaluate(model,val_loader,device,args.threshold,args.output_dir/"visualizations",args.max_visuals)
    summary={"dataset":args.dataset,"data_root":str(args.data_root.resolve()),"device":str(device),
             "train_images":len(train_records),"validation_images":len(val_records),
             "epochs_requested":args.epochs,"best_epoch":ckpt["epoch"],"seed":args.seed,
             "input_height":args.height,"input_width":args.width,"threshold":args.threshold,
             "elapsed_seconds":time.time()-start,"validation_pixel_metrics":final_metrics,
             "metric_warning":"Pixel-mask metrics are not official TuSimple or CULane benchmark scores."}
    (args.output_dir/"results.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (args.output_dir/"per_image_metrics.jsonl").open("w",encoding="utf-8") as f:
        for row in per_image: f.write(json.dumps(row)+"\n")
    print("\nFINAL SUMMARY\n"+json.dumps(summary,indent=2))
    print(f"Saved checkpoint: {best_path}")

if __name__=="__main__": main()
