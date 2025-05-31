import os, json
from torch.utils.data import Dataset
from config import Config

class GovReportDataset(Dataset):
    def __init__(self, split: str, tokenizer, report_type: str):
        self.tokenizer = tokenizer
        self.max_in = Config.MAX_INPUT_LENGTH
        self.max_out = Config.MAX_TARGET_LENGTH

        data_dir = Config.OFFLINE_DATA_DIR
        ids_path = os.path.join(data_dir, "split_ids", f"{report_type}_{split}.ids")
        with open(ids_path) as f:
            ids = [l.strip() for l in f if l.strip()]

        self.samples = []
        for id_ in ids:
            fp = os.path.join(data_dir, report_type, f"{id_}.json")
            if not os.path.exists(fp): continue
            obj = json.load(open(fp))
            title = obj.get("title", "")
            if report_type == "crs":
                doc = self._flatten_section(obj.get("report", {}))
                summ = "\n".join(obj.get("summary", []))
            else:
                doc = self._flatten_section(obj.get("report", []))
                summ = self._flatten_section(obj.get("highlight", []))
            self.samples.append((title, doc, summ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        title, doc, summ = self.samples[i]
        src = (title + "\n" + doc).strip()
        tgt = summ.strip()

        enc = self.tokenizer(
            src,
            max_length=self.max_in,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        dec = self.tokenizer(
            tgt,
            max_length=self.max_out,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = enc.input_ids.squeeze()
        attention_mask = enc.attention_mask.squeeze()
        labels = dec.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def _flatten_section(self, node):
        if isinstance(node, str):
            return node
        if isinstance(node, list):
            return "\n".join(self._flatten_section(x) for x in node if x)
        if not node:
            return ""
        out = []
        if node.get("section_title"):
            out.append(node["section_title"])
        out.extend(node.get("paragraphs", []))
        out.append(self._flatten_section(node.get("subsections", [])))
        return "\n".join(out)
