from transformers import AutoTokenizer
from data.dataset import GovReportDataset
from config import Config

def inspect_dataset(split="train", report_type="crs"):
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    dataset = GovReportDataset(split=split, tokenizer=tokenizer, report_type=report_type)
    print(f"Total examples in split '{split}' ({report_type.upper()}): {len(dataset)}\n")
    sample = dataset.samples[0]  # (title, doc, summ)
    print("===== RAW TEXT =====")
    print("TITLE:", sample[0])
    print("\nDocument (first 1000 chars):")
    print(sample[1][:1000])
    print("\nSummary (first 500 chars):")
    print(sample[2][:500])
    src_ids = tokenizer.encode((sample[0] + "\n" + sample[1]).strip(), add_special_tokens=True)
    tgt_ids = tokenizer.encode(sample[2].strip(), add_special_tokens=True)
    print("\n===== TOKENIZED LENGTHS (no truncation) =====")
    print("Source tokens:", len(src_ids))
    print("Target tokens:", len(tgt_ids))
    print("First 10 source token IDs:", src_ids[:10])
    print("First 10 target token IDs:", tgt_ids[:10])

if __name__ == "__main__":
    print("=== Inspecting CRS sample ===")
    inspect_dataset(split="train", report_type="crs")
    print("\n=== Inspecting GAO sample ===")
    inspect_dataset(split="train", report_type="gao")
