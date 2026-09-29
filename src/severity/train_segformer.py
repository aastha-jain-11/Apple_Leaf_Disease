import json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, get_cosine_schedule_with_warmup
from severity_config import *
from severity_dataset import SeveritySegmentationDataset

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def metrics(pred, target):
    out = {}; ds=[]; ios=[]
    for c, name in enumerate(CLASS_NAMES):
        p, t = pred == c, target == c
        tp = np.logical_and(p,t).sum(); fp = np.logical_and(p,~t).sum(); fn = np.logical_and(~p,t).sum()
        dd, di = 2*tp+fp+fn, tp+fp+fn
        d = 1.0 if dd == 0 else 2*tp/dd
        io = 1.0 if di == 0 else tp/di
        out[f"dice_{name}"] = float(d); out[f"iou_{name}"] = float(io)
        ds.append(d); ios.append(io)
    out["mean_dice"] = float(np.mean(ds)); out["mean_iou"] = float(np.mean(ios))
    return out

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); pp=[]; tt=[]; loss=0
    for b in loader:
        x=b["pixel_values"].to(device); y=b["labels"].to(device)
        o=model(pixel_values=x, labels=y); loss += o.loss.item()*x.size(0)
        z=torch.nn.functional.interpolate(o.logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        pp.append(z.argmax(1).cpu().numpy()); tt.append(y.cpu().numpy())
    m=metrics(np.concatenate(pp), np.concatenate(tt)); m["loss"]=loss/len(loader.dataset); return m

def main():
    seed_everything(SEED); MODEL_ROOT.mkdir(parents=True,exist_ok=True); OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr=SeveritySegmentationDataset(SPLIT_CSV,"train",True); va=SeveritySegmentationDataset(SPLIT_CSV,"validation",False)
    tl=DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=device.type=="cuda")
    vl=DataLoader(va,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=device.type=="cuda")
    model=SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_CHECKPOINT,num_labels=NUM_CLASSES,
        id2label=dict(enumerate(CLASS_NAMES)),label2id={x:i for i,x in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    steps=EPOCHS*max(1,len(tl))
    sch=get_cosine_schedule_with_warmup(opt,max(1,steps//20),steps)
    scaler=torch.amp.GradScaler("cuda") if device.type=="cuda" else None
    best=-1; patience=0; hist=[]
    for ep in range(1,EPOCHS+1):
        model.train(); run=0; start=time.time()
        for b in tl:
            x=b["pixel_values"].to(device); y=b["labels"].to(device); opt.zero_grad(set_to_none=True)
            if scaler:
                with torch.autocast(device_type="cuda",dtype=torch.float16): o=model(pixel_values=x,labels=y)
                scaler.scale(o.loss).backward(); scaler.step(opt); scaler.update()
            else:
                o=model(pixel_values=x,labels=y); o.loss.backward(); opt.step()
            sch.step(); run += o.loss.item()*x.size(0)
        vm=evaluate(model,vl,device); row={"epoch":ep,"train_loss":run/len(tl.dataset),**{f"val_{k}":v for k,v in vm.items()},"seconds":time.time()-start}; hist.append(row)
        print(f"Epoch {ep:02d}: train={row['train_loss']:.4f} mIoU={vm['mean_iou']:.4f} lesionIoU={vm['iou_lesion']:.4f} lesionDice={vm['dice_lesion']:.4f}")
        score=vm["dice_lesion"]
        if score>best:
            best=score; patience=0; model.save_pretrained(MODEL_ROOT/"best")
        else: patience+=1
        if patience>=EARLY_STOPPING_PATIENCE: break
    pd.DataFrame(hist).to_csv(OUTPUT_ROOT/"training_history.csv",index=False)
    with open(OUTPUT_ROOT/"training_config.json","w") as f: json.dump({"checkpoint":SEGFORMER_CHECKPOINT,"classes":CLASS_NAMES,"best_lesion_dice":best},f,indent=2)
    print("Best model:", MODEL_ROOT/"best")

if __name__=="__main__": main()
