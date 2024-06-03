import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from utils import epoch
from networks import (
    LinearPcDEQ1,
    LinearPcDEQ2,
    SingleConvPcDEQ1,
    SingleConvPcDEQ2,
    MultiConvPcDEQ1,
    MultiConvPcDEQ2,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="DEQ name")
    parser.add_argument("-activ", type=str, required=True, help="activation function")
    parser.add_argument("-lr", type=float, default=7e-4, required=True)
    parser.add_argument("-epochs", type=int, default=50, required=True)
    parser.add_argument("-wd", type=float, default=1e-2, required=True)
    parser.add_argument("-b", type=int, default=64, required=True)
    parser.add_argument("-m", type=int, required=True)
    args = parser.parse_args()

    p = Path(__file__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = f"{p.parent}/weights"
    Path(weights_path).mkdir(parents=True, exist_ok=True)

    match args.net:
        case "linear_pcdeq_1":
            model = LinearPcDEQ1(80, args.activ)
        case "linear_pcdeq_2":
            model = LinearPcDEQ2(80, args.activ)
        case "single_conv_pcdeq_1":
            model = SingleConvPcDEQ1(28, 1, 82, args.activ)
        case "single_conv_pcdeq_2":
            model = SingleConvPcDEQ2(28, 1, 82, args.activ)
        case "multi_conv_pcdeq_1":
            model = MultiConvPcDEQ1(1, 12, 24, 48, args.activ)
        case "multi_conv_pcdeq_2":
            model = MultiConvPcDEQ2(1, 12, 24, 48, args.activ)
        case _:
            raise NotImplementedError(
                f"Network of name '{args.net}' currently is not supported"
            )

    model.to(device)
    mnist_train = datasets.MNIST(
        ".", train=True, download=True, transform=transforms.ToTensor()
    )

    mnist_test = datasets.MNIST(
        ".", train=False, download=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader(
        mnist_train, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        mnist_test, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True
    )
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[args.m], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for i in range(1, args.epochs + 1):
        train_acc, train_loss, train_fiter, train_biter = epoch(
            train_loader, model, device, criterion, opt
        )
        test_acc, test_loss, test_fiter, _ = epoch(
            test_loader, model, device, criterion
        )
        scheduler.step()
        print(
            f"Epoch: {i} | Train acc: {train_acc:.4f}, Loss: {train_loss:.4f}, For Iters: {train_fiter:.2f}, Back Iters: {train_biter:.2f} | "
            + f"Test acc: {test_acc:.4f}, Loss: {test_loss:.4f}, For Iters: {test_fiter:.2f}"
        )
        if best_acc < test_acc:
            torch.save(
                model.state_dict(), f"{weights_path}/mnist_{args.net}_{args.activ}.pth"
            )


if __name__ == "__main__":
    main()
