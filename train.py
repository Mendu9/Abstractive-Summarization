
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from config import Config
from data.datamodule import SummarizationDataModule
from models.model import SummarizationModel

def main():
    pl.seed_everything(42)
    model_type   = "t5"
    batch_size   = Config.BATCH_SIZE
    epochs       = Config.EPOCHS
    grad_accum   = Config.GRAD_ACCUM
    gpus         = Config.GPUS
    fp16         = Config.FP16
    num_workers  = Config.NUM_WORKERS
    report_types = Config.DEFAULT_REPORT_TYPES
    output_dir   = Config.OUTPUT_DIR

    dummy = SummarizationModel(model_type, Config)
    tokenizer = dummy.tokenizer
    dm = SummarizationDataModule(
        tokenizer,
        report_types=report_types,
        batch_size=batch_size,
        num_workers=num_workers
    )
    dm.setup()
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(output_dir, model_type),
        filename="best-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss", mode="min", save_top_k=1
    )
    lrmon = LearningRateMonitor(logging_interval="step")
    accelerator = "gpu" if gpus > 0 else "cpu"
    devices     = gpus if gpus > 0 else None
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        precision=16 if fp16 else 32,
        accumulate_grad_batches=grad_accum,
        callbacks=[ckpt_cb, lrmon],
        default_root_dir=output_dir,
    )
    model = SummarizationModel(model_type, Config)
    trainer.fit(model, dm)
    trainer.test(model, dm)

if __name__ == "__main__":
    main()
