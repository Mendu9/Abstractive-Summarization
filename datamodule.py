import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from data.dataset import GovReportDataset

class SummarizationDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, report_types, batch_size, num_workers):
        super().__init__()
        self.tokenizer    = tokenizer
        self.report_types = report_types
        self.batch_size   = batch_size
        self.num_workers  = num_workers

    def setup(self, stage=None):
        trains, vals, tests = [], [], []
        for rpt in self.report_types:
            trains.append(GovReportDataset("train", self.tokenizer, rpt))
            vals.append(  GovReportDataset("valid", self.tokenizer, rpt))
            tests.append( GovReportDataset("test",  self.tokenizer, rpt))
        self.train_ds = ConcatDataset(trains)
        self.val_ds   = ConcatDataset(vals)
        self.test_ds  = ConcatDataset(tests)

        print(f">>> Loaded datasets:")
        print(f"    train: {len(self.train_ds)} samples")
        print(f"    valid: {len(self.val_ds)} samples")
        print(f"    test : {len(self.test_ds)} samples")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0
        )
