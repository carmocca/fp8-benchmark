import argparse
import time
from datetime import datetime

import torch
from lightning.pytorch import Trainer, seed_everything, LightningModule
from lightning.pytorch.demos import WikiText2, Transformer


class LanguageModel(LightningModule):
    def __init__(self, vocab_size, log_interval):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)
        self.log_interval = log_interval

    def on_train_start(self):
        self.val_loss = torch.tensor(0.0, device=self.trainer.strategy.root_device)
        self.train_n = len(self.trainer.train_dataloader.dataset)
        self.val_n = len(self.trainer.val_dataloaders.dataset)

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.t0 = time.perf_counter()

        data, target = batch
        output = self.model(data, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))

        if batch_idx % self.log_interval == 0 and batch_idx != 0:
            t1 = time.perf_counter()
            elapsed = t1 - self.t0
            tprint(
                f"Train Epoch {self.current_epoch}: [{batch_idx * len(data)}/{self.train_n} "
                f"({100. * batch_idx / len(self.trainer.train_dataloader):.0f}%)] "
                f"{elapsed / self.log_interval:.05f} sec/batch. Loss: {loss.item():.6f}"
            )
            self.t0 = t1

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data, target)
        self.val_loss += torch.nn.functional.nll_loss(output, target.view(-1), reduction="sum")

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def on_train_epoch_start(self):
        torch.cuda.reset_peak_memory_stats()

    def on_validation_epoch_start(self):
        tprint(f"Train {self.current_epoch} memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()

    def on_validation_epoch_end(self):
        t1 = time.perf_counter()
        tprint(f"Validate: Average loss: {self.val_loss / self.val_n:.4f}, in {t1 - self.t0:.05f} seconds")
        tprint(f"Validate {self.current_epoch} memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    seed_everything(args.seed)

    dataset = WikiText2()
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    model = LanguageModel(dataset.vocab_size, args.log_interval)

    trainer = Trainer(
        accelerator="cuda",
        devices=1,
        gradient_clip_val=0.25,
        max_epochs=args.epochs,
        barebones=True,
        precision="8-mixed",
    )

    device = trainer.strategy.root_device
    print(torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device))
    print(f"Train bs: {args.batch_size}, validate bs: {args.val_batch_size}")

    trainer.fit(model, train_dataloader, val_dataloader)


def tprint(msg):
    now = datetime.utcnow().strftime("%F %T.%f")[:-3]
    print(f"[{now}] {msg}")


if __name__ == "__main__":
    main()
