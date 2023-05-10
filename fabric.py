import argparse
import time
from datetime import datetime

import torch
from lightning.fabric import Fabric, seed_everything
from lightning.pytorch.demos import WikiText2, Transformer


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    n = len(train_loader.dataset)

    t0 = time.perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        loss.backward(loss)
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.25)
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % args.log_interval == 0 and batch_idx != 0:
            t1 = time.perf_counter()
            elapsed = t1 - t0
            tprint(
                f"Train Epoch {epoch}: [{batch_idx * len(data)}/{n} ({100. * batch_idx / len(train_loader):.0f}%)] "
                f"{elapsed / args.log_interval:.05f} sec/batch. Loss: {loss.item():.6f}"
            )
            t0 = t1


@torch.no_grad()
def validate(model, device, loader):
    model.eval()
    loss = torch.tensor(0.0, device=device)

    t0 = time.perf_counter()
    for data, target in loader:
        output = model(data, target)
        loss += torch.nn.functional.nll_loss(output, target.view(-1), reduction="sum")
    t1 = time.perf_counter()

    n = len(loader.dataset)
    tprint(f"Validate: Average loss: {loss / n:.4f}, in {t1 - t0:.05f} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    seed_everything(args.seed)
    fabric = Fabric(accelerator="cuda", devices=1)  # , precision="8-mixed")

    print(torch.cuda.get_device_name(fabric.device), torch.cuda.get_device_capability(fabric.device))
    print(f"Train bs: {args.batch_size}, validate bs: {args.val_batch_size}")

    dataset = WikiText2()
    train_dataset, val_dataloader = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataloader, batch_size=args.val_batch_size, shuffle=False)
    model = Transformer(vocab_size=dataset.vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    for epoch in range(1, args.epochs + 1):
        torch.cuda.reset_peak_memory_stats()
        train(args, model, train_dataloader, optimizer, epoch)
        tprint(f"Train {epoch} memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        torch.cuda.reset_peak_memory_stats()
        validate(model, fabric.device, val_dataloader)
        tprint(f"Validate {epoch} memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


def tprint(msg):
    now = datetime.utcnow().strftime("%F %T.%f")[:-3]
    print(f"[{now}] {msg}")


if __name__ == "__main__":
    main()
