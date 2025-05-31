import torch
import pytorch_lightning as pl
from utils.metrics import compute_rouge      
from evaluate import load as load_metric
from torch.optim import AdamW
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    T5ForConditionalGeneration, T5TokenizerFast,
    BartForConditionalGeneration, BartTokenizerFast,
    get_cosine_schedule_with_warmup
)
from config import Config

class SummarizationModel(pl.LightningModule):
    def __init__(self, model_type: str, args):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        if model_type == "t5":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        else:
            self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
            self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
        self.args = args

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        self.log("train_loss", out.loss, prog_bar=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        self.log("val_loss", out.loss, prog_bar=True)
        return out.loss

    def on_test_start(self):
        self.generated_greedy, self.generated_beam, self.references = [], [], []
    
    def test_step(self, batch, batch_idx):
        out = self(**batch)    
        
        greedy = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=Config.MAX_TARGET_LENGTH
        )
        beam = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=Config.MAX_TARGET_LENGTH,
            num_beams=4, early_stopping=True,
        )
        self.generated_greedy.extend(self.tokenizer.batch_decode(greedy, skip_special_tokens=True))
        self.generated_beam.extend(self.tokenizer.batch_decode(beam,   skip_special_tokens=True))
        self.references.extend(self.tokenizer.batch_decode(batch["labels"].mask_fill(batch["labels"]==-100, self.tokenizer.pad_token_id),
                                                           skip_special_tokens=True))

    def on_test_end(self):
        rouge_g  = compute_rouge(self.generated_greedy, self.references)
        rouge_b4 = compute_rouge(self.generated_beam,   self.references)
        self.print(f"\nGreedy ROUGE: {rouge_g}")
        self.print(f"Beam-4 ROUGE: {rouge_b4}")

        bertscore = load_metric("bertscore")
        p,r,f1 = bertscore.compute(predictions=self.generated_beam,
                                   references=self.references,
                                   lang="en")["f1"]
        self.print(f"Beam-4 BERTScore F1: {sum(f1)/len(f1):.4f}")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        train_loader = self.trainer.datamodule.train_dataloader()
        total_batches = len(train_loader) // self.args.GRAD_ACCUM
        total_steps = total_batches * self.args.EPOCHS
        scheduler = {
            'scheduler': get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=Config.WARMUP_STEPS,
                num_training_steps=total_steps
            ),
            'interval': 'step'
        }
        return [optimizer], [scheduler]
