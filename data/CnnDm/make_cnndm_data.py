# import cnn-dm dataset and generate target-source format
SRC_PATH = "../../../progressive-generation/data/cnn/{}.cnndm_source"
TGT_PATH = "../../../progressive-generation/data/cnn/{}.cnndm_target"
for split in ("train","valid","test"):
    print(f"processing {split} split from {SRC_PATH.format(split)}")
    lines = []
    with open(SRC_PATH.format(split)) as prompt, open(TGT_PATH.format(split)) as txt:
        pt_list = [l.strip() for l in prompt.readlines()]
        txt_list = [l.strip() for l in txt.readlines()]
    for pt, txt in zip(pt_list, txt_list):
        lines.append(f"{pt}\t{txt}")
    with open(f"cnndm_{split}.txt",'w',encoding='utf8') as fout:
        fout.write("\n".join(lines) + "\n")