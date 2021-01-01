import torch
import pickle
import logging
import argparse
import torch.nn as nn
from tqdm import tqdm
from model import UNet
import matplotlib.pyplot as plt
from utils import count_parameters
from prepare_data import get_dataloader

class SoftDiceLoss(nn.Module):
    '''
    Soft Dice Loss
    '''        
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = torch.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))


class CombinedLoss(nn.Module):
    def __init__(self, criterion_list):
        super(CombinedLoss, self).__init__()
        self.criterion_list = criterion_list

    def forward(self, logits, target):
        loss = 0
        for criterion in self.criterion_list:
            loss += criterion(logits, target)
        return loss


if __name__ == "__main__":
    logging.basicConfig(
        filename="logfile.log",
        format="%(levelname)s %(asctime)s %(message)s",
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Original UNet Paper Replication!")
    parser.add_argument("--batch_size", help="Default=8", type=int, default=8)
    parser.add_argument("--lr", help="Default=0.00001", type=float, default=0.00001)
    parser.add_argument("--epochs", help="Default=200", type=int, default=200)
    parser.add_argument(
        "--train_img_dir",
        help="Default=data/train_imgs",
        type=str,
        default="data/train_imgs",
    )
    parser.add_argument(
        "--train_label_dir",
        help="data/train_labels",
        type=str,
        default="data/train_labels",
    )
    parser.add_argument('-print_dataset', help="Default=False", action='store_true', default=False)
    args = parser.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    logger.info("Device used: {}".format(dev))

    logger.info("Reading data initialized...")
    batch_size = args.batch_size
    dataloader = get_dataloader(image_dir=args.train_img_dir, labels_dir=args.train_label_dir, print_dataset=args.print_dataset, batch_size=batch_size, input_img_size=(572, 572), output_img_size=(388, 388))
    logger.info("Reading data finished...")
    
    net = UNet().to(device)
    count_parameters(net)

    # criterion = CombinedLoss([torch.nn.BCEWithLogitsLoss(), SoftDiceLoss()])
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 1e-16, last_epoch=-1, verbose=True)

    net.train()
    counter = []
    for epoch in tqdm(range(args.epochs)):
        net_loss = 0
        for i, batch in enumerate(tqdm(dataloader)):
            X, y = batch
            X = X.view(batch_size, 1, 572, 572).float()
            y = y.view(batch_size, 1, 388, 388).float()
            # data is a batch of featuresets and labels
            optimizer.zero_grad()
            outputs = net(X.to(device))
            loss = criterion(outputs, y.to(device))
            net_loss += loss
            loss.backward()
            optimizer.step()
        scheduler.step()
        net_loss /= (i + 1)
        plt.imshow(torch.round(torch.sigmoid(outputs[0][0])).cpu().detach().numpy())
        plt.savefig("figure_{}.jpg".format(epoch))

        plt.imshow(torch.round(outputs[0][0]).cpu().detach().numpy())
        plt.savefig("figure_raw_{}.jpg".format(epoch))

        logger.info("Epoch={}     loss={}".format(epoch, net_loss))
        counter.append(net_loss)

    with open("loss_list.txt", "wb") as output_file:
        pickle.dump(counter, output_file)
