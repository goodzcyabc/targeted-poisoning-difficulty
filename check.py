import os
from collections import Counter

DATA_PATH = "./data/tiny-imagenet-200"

def load_wnids(data_path):
    with open(os.path.join(data_path, "wnids.txt"), "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

# build official mapping
wnids = load_wnids(DATA_PATH)
class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
idx_to_class = {i: wnid for wnid, i in class_to_idx.items()}

# read val_annotations.txt into image -> wnid
ann = {}
ann_path = os.path.join(DATA_PATH, "val", "val_annotations.txt")
with open(ann_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        img_name, wnid = parts[0], parts[1]
        ann[img_name] = wnid

# reconstruct exactly how your current val dataset should look
samples = []
img_root = os.path.join(DATA_PATH, "val", "images")
with open(ann_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        img_name, wnid = parts[0], parts[1]
        path = os.path.join(img_root, img_name)
        y = class_to_idx[wnid]
        samples.append((path, y))

# check yt=0
yt = 0
picked = [(i, p, y) for i, (p, y) in enumerate(samples) if y == yt]

print("yt =", yt)
print("official wnid for yt=0 =", idx_to_class[0])
print("num picked =", len(picked))

for i, p, y in picked[:10]:
    img_name = os.path.basename(p)
    true_wnid = ann[img_name]
    print(i, img_name, "label=", y, "true_wnid=", true_wnid)

print("unique wnids among picked:")
print(Counter(ann[os.path.basename(p)] for i, p, y in picked))