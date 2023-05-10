import argparse
import time

import torch
from datetime import datetime

from lightning.pytorch.demos import WikiText2, Transformer


from transformer_engine import pytorch as te


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    n = len(train_loader.dataset)

    t0 = time.perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        with te.fp8_autocast(enabled=True):
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
        data, target = data.to(device), target.to(device)
        with te.fp8_autocast(enabled=True):
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

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device))
    print(f"Train bs: {args.batch_size}, validate bs: {args.val_batch_size}")

    dataset = WikiText2()
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
    model = Transformer(vocab_size=dataset.vocab_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(1, args.epochs + 1):
        torch.cuda.reset_peak_memory_stats()
        train(args, model, device, train_dataloader, optimizer, epoch)
        tprint(f"Train {epoch} memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        torch.cuda.reset_peak_memory_stats()
        validate(model, device, val_dataloader)
        tprint(f"Validate {epoch} memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


def tprint(msg):
    now = datetime.utcnow().strftime("%F %T.%f")[:-3]
    print(f"[{now}] {msg}")


if __name__ == "__main__":
    main()
