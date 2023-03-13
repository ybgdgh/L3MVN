import os
import pickle
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F


from dataset import FinetuningDataset, create_room_splits
from models import ContrastiveNet, FeedforwardNet

import sys 
sys.path.append("..") 
from constants import mp3d_category_id, category_to_id
from utils.tensorboard_utils import TensorboardWriter


def train_job(lm, label_set, use_gt, epochs, batch_size, seed=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    writer = TensorboardWriter('tb', flush_secs=30)


    def ff_loss(pred, label):
        return F.cross_entropy(pred, label)

    # Create datasets
    suffix = lm + "_" + label_set + "_useGT_" + str(use_gt) + "_502030"
    path_to_data = os.path.join("./data/", suffix)
    train_ds, val_ds, test_ds = create_room_splits(path_to_data, device="cuda")

    # Create dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    # test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

    output_size = len(category_to_id)

    ff_net = FeedforwardNet(1024, output_size)
    ff_net.to(device)

    # 63.42, lr=0.00001, wd=0.001, ss=50, g=0.1
    # 64.49, lr=0.0001, wd=0.001, ss=10, g=0.5
    optimizer = torch.optim.Adam(ff_net.parameters(),
                                 lr=0.0001,
                                 weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=0.5)

    loss_fxn = ff_loss

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    desc = ""
    with trange(epochs) as pbar:
        for epoch in pbar:
            train_epoch_loss = []
            val_epoch_loss = []
            train_epoch_acc = []
            val_epoch_acc = []
            for batch_idx, (query_em, _, label) in enumerate(train_dl):
                pred = ff_net(query_em)
                loss = loss_fxn(pred, label)

                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())

                accuracy = ((torch.argmax(pred, dim=1) == label) * 1.0).mean()
                train_epoch_acc.append(accuracy)

                if batch_idx % 100 == 0:
                    pbar.set_description((desc).rjust(20))

            scheduler.step()
            train_losses.append(torch.mean(torch.tensor(train_epoch_loss)))
            train_acc.append(torch.mean(torch.tensor(train_epoch_acc)))

            for batch_idx, (query_em, _, label) in enumerate(val_dl):
                with torch.no_grad():
                    pred = ff_net(query_em)
                    loss = loss_fxn(pred, label)
                    val_epoch_loss.append(loss.item() * len(label))

                    accuracy = ((torch.argmax(pred, dim=1) == label) *
                                1.0).mean()
                    val_epoch_acc.append(accuracy * len(label))
                    if batch_idx % 100 == 0:
                        desc = (f"{loss.item():6.4}" + ", " +
                                f"{accuracy.item():6.4}")
                        pbar.set_description((desc).rjust(20))
            val_losses.append(
                torch.sum(torch.tensor(val_epoch_loss)) / len(val_ds))
            val_acc.append(
                torch.sum(torch.tensor(val_epoch_acc)) / len(val_ds))
            if epoch == 0:
                best_val_acc = val_acc[-1]
                torch.save(ff_net.state_dict(),
                           "./checkpoints/best_ff_" + suffix + ".pt")
            elif val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]
                torch.save(ff_net.state_dict(),
                           "./checkpoints/best_ff_" + suffix + ".pt")

            writer.add_scalar(suffix+"train_losses", np.mean(train_epoch_loss), epoch)
            writer.add_scalar(suffix+"train_acc", torch.mean(torch.tensor(train_epoch_acc)).cpu().numpy(), epoch)
            writer.add_scalar(suffix+"val_losses", np.mean(val_epoch_loss), epoch)
            writer.add_scalar(suffix+"val_acc", torch.mean(torch.tensor(val_epoch_acc)).cpu().numpy(), epoch)

    ff_net.load_state_dict(
        torch.load("./checkpoints/best_ff_" + suffix + ".pt"))
    ff_net.eval()
    test_loss, test_acc = [], []
    # for batch_idx, (query_em, _, label) in enumerate(test_dl):
    #     pred = ff_net(query_em)
    #     loss = loss_fxn(pred, label)
    #     test_loss.append(loss.item())

    #     accuracy = ((torch.argmax(pred, dim=1) == label) * 1.0).mean()
    #     test_acc.append(accuracy)

    writer.close()
    print("test loss:", torch.mean(torch.tensor(test_loss)))
    print("test acc:", torch.mean(torch.tensor(test_acc)))
    return train_losses, val_losses, train_acc, val_acc, test_loss, test_acc


if __name__ == "__main__":
    (
        train_losses_list,
        val_losses_list,
        train_acc_list,
        val_acc_list,
        test_loss_list,
        test_acc_list,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for lm in ["RoBERTa-large", "BERT-large"]:
        for label_set in ["mpcat40"]:
            for use_gt in [True, False]:
                print("Starting:", lm, label_set, "use_gt =", use_gt)
                (
                    train_losses,
                    val_losses,
                    train_acc,
                    val_acc,
                    test_loss,
                    test_acc,
                ) = train_job(lm, label_set, use_gt, 200, 512)
                train_losses_list.append(train_losses)
                val_losses_list.append(val_losses)
                train_acc_list.append(train_acc)
                val_acc_list.append(val_acc)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)

    pickle.dump(train_losses_list, open("./ff_results/train_losses.pkl", "wb"))
    pickle.dump(train_acc_list, open("./ff_results/train_acc.pkl", "wb"))
    pickle.dump(val_losses_list, open("./ff_results/val_losses.pkl", "wb"))
    pickle.dump(val_acc_list, open("./ff_results/val_acc.pkl", "wb"))
    pickle.dump(test_loss_list, open("./ff_results/test_loss.pkl", "wb"))
    pickle.dump(test_acc_list, open("./ff_results/test_acc.pkl", "wb"))
