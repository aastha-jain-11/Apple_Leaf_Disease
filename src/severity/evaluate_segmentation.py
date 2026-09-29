import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from severity_config import SPLIT_CSV, MODEL_ROOT, OUTPUT_ROOT, CLASS_NAMES
from severity_dataset import SeveritySegmentationDataset

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SegformerForSemanticSegmentation.from_pretrained(MODEL_ROOT/"best").to(device).eval()
    ds=SeveritySegmentationDataset(SPLIT_CSV,"test",False)
    dl=DataLoader(ds,batch_size=16,shuffle=False,num_workers=4)
    pp=[]; tt=[]
    with torch.no_grad():
        for b in dl:
            x=b["pixel_values"].to(device); y=b["labels"].to(device)
            z=model(pixel_values=x).logits
            z=torch.nn.functional.interpolate(z,size=y.shape[-2:],mode="bilinear",align_corners=False)
            pp.append(z.argmax(1).cpu().numpy()); tt.append(y.cpu().numpy())
    p=np.concatenate(pp); t=np.concatenate(tt); out={}
    for c,n in enumerate(CLASS_NAMES):
        a=p==c; b=t==c; tp=np.logical_and(a,b).sum(); fp=np.logical_and(a,~b).sum(); fn=np.logical_and(~a,b).sum()
        out[f"dice_{n}"]=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 1.0
        out[f"iou_{n}"]=float(tp/(tp+fp+fn)) if tp+fp+fn else 1.0
    out["mean_dice"]=float(np.mean([out[f"dice_{n}"] for n in CLASS_NAMES]))
    out["mean_iou"]=float(np.mean([out[f"iou_{n}"] for n in CLASS_NAMES]))
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    with open(OUTPUT_ROOT/"test_metrics.json","w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
