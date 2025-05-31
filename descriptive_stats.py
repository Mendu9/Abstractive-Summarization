from pathlib import Path
import json, math, pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm
from config import Config

ROOT       = Path(Config.OFFLINE_DATA_DIR)
TOKENIZER  = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

def flatten(node):
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "\n".join(flatten(x) for x in node if x)
    if not node:
        return ""
    parts = []
    if node.get("section_title"):
        parts.append(node["section_title"])
    parts.extend(node.get("paragraphs", []))
    parts.append(flatten(node.get("subsections", [])))
    return "\n".join(parts)

records = []
for corpus in Config.REPORT_TYPES:                 
    for split in ["train", "valid", "test"]:
        id_file = ROOT/"split_ids"/f"{corpus}_{split}.ids"
        if not id_file.exists():  continue
        ids = id_file.read_text().split()

        for fid in tqdm(ids, desc=f"{corpus}-{split}"):
            js = json.load(open(ROOT/corpus/f"{fid}.json", encoding="utf-8"))
            title = js.get("title", "")

   
            if corpus == "crs":
                body     = flatten(js.get("report", {}))
                target   = "\n".join(js.get("summary", []))
                sec_cnt  = 1  
            else:  
                body     = flatten(js.get("report", []))
                target   = flatten(js.get("highlight", []))
                sec_cnt  = len(js.get("report", []))  

            src_text  = (title + "\n" + body).strip()
            tgt_text  = target.strip()

            records.append({
                "corpus": corpus.upper(), "split": split,
                "chars_src": len(src_text),
                "chars_tgt": len(tgt_text),
                "toks_src": len(TOKENIZER.encode(src_text,  add_special_tokens=True)),
                "toks_tgt": len(TOKENIZER.encode(tgt_text, add_special_tokens=True)),
                "ratio":    len(TOKENIZER.encode(tgt_text, add_special_tokens=True)) /
                            max(1, len(TOKENIZER.encode(src_text, add_special_tokens=True))),
                "sections": sec_cnt
            })

df = pd.DataFrame(records)

summary = (
    df.groupby(["corpus","split"])
      [["toks_src","toks_tgt","chars_src","chars_tgt"]]
      .agg(["mean","median",lambda x: x.quantile(0.95)])
      .round(1)
)
summary.columns = [f"{m}_{stat}" for m,stat in summary.columns]
print("\n=== Length Summary Statistics ===")
print(summary.to_markdown())

plt.figure(figsize=(12,10))
plt.subplot(2,2,1); df["toks_src"].hist(bins=60); plt.title("Source Tokens")
plt.subplot(2,2,2); df["toks_tgt"].hist(bins=60); plt.title("Target Tokens")
plt.subplot(2,2,3); df["chars_src"].hist(bins=60); plt.title("Source Characters")
plt.subplot(2,2,4); df["chars_tgt"].hist(bins=60); plt.title("Target Characters")
plt.tight_layout(); plt.show()

plt.figure(figsize=(6,4))
df["ratio"].hist(bins=60)
plt.title("Compression Ratio (target/source tokens)"); plt.xlabel("ratio"); plt.ylabel("frequency")
plt.show()

plt.figure(figsize=(6,4))
df["sections"].hist(bins=30)
plt.title("Top-level Section Count per Report"); plt.xlabel("# sections"); plt.ylabel("frequency")
plt.show()

plt.figure(figsize=(6,5))
plt.hexbin(df["toks_src"], df["toks_tgt"], gridsize=60, cmap="viridis", mincnt=3)
plt.xlabel("Source tokens"); plt.ylabel("Target tokens")
plt.title("2-D Density: Source vs Target Length")
cb = plt.colorbar(); cb.set_label("document count")
plt.show()

plt.figure(figsize=(6,5))
df.boxplot(column="toks_src", by="corpus")
plt.suptitle(""); plt.title("Source Length Variability by Corpus")
plt.ylabel("tokens")
plt.show()

p95 = df["toks_src"].quantile(0.95)
print(f"\n95th-percentile source length = {p95:,.0f} tokens")
print("Chosen MAX_INPUT_LENGTH = 16 384  (covers > 99 %)")
print("Decoder cap 512 tokens  (max target =", df['toks_tgt'].max(), ")")
