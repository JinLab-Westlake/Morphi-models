import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from datasets.cell_datasets import Synapse_dataset, RandomGenerator
import torch.nn.functional as F
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


class ResizePadGenerator:
    def __init__(self, output_size=[224, 224]):
        self.target_size = output_size[0]

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Make sure the input is a PIL image
        image = TF.to_pil_image(image)
        label = TF.to_pil_image(label)

        iw, ih = image.size

        # Determine if resizing is necessary
        if iw > self.target_size or ih > self.target_size:
            # Proportional scaling (long =target_size)
            scale = self.target_size / max(iw, ih)
            new_w, new_h = int(iw * scale), int(ih * scale)
            image = image.resize((new_w, new_h), resample=Image.BILINEAR)
            label = label.resize((new_w, new_h), resample=Image.NEAREST)
        else:
            # No scaling
            new_w, new_h = iw, ih

        # padding to object size
        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top

        image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        label = TF.pad(label, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

        # Turn back to Tensor
        image = TF.to_tensor(image)
        label = torch.as_tensor(np.array(label), dtype=torch.long)

        return {'image': image, 'label': label}


def generate_soft_weight_from_label(label_tensor):
    """
    label_tensor: [B, H, W] or [B, 1, H, W], label==2 is the dead zone
    return: soft_weight [B, 1, H, W],Normalized distance transform
    """
    if label_tensor.ndim == 4:
        label_tensor = label_tensor.squeeze(1)
    batch_size, H, W = label_tensor.shape
    soft_weight = torch.zeros((batch_size, 1, H, W), dtype=torch.float32)

    for i in range(batch_size):
        label_np = label_tensor[i].cpu().numpy().astype(np.uint8)
        mask = (label_np == 2).astype(np.uint8)
        dist_map = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
        if dist_map.max() > 0:
            dist_map = dist_map / dist_map.max()
        soft_weight[i, 0] = torch.tensor(dist_map)

    return soft_weight.cuda()


def get_live_dead_soft_edge(label_batch, max_dist=5):
    labels = label_batch.detach().cpu().numpy().astype(np.uint8)
    B, H, W = labels.shape
    soft_weights = []

    kernel = np.ones((3, 3), dtype=np.uint8)

    for i in range(B):
        label = labels[i]

        live_mask = label == 1
        dead_mask = (label == 2).astype(np.uint8)

        dead_dilated = cv2.dilate(
            dead_mask,
            kernel,
            iterations=1,
        )

        # Live-side interface pixels
        boundary_focus = live_mask & (dead_dilated > 0)

        if np.any(boundary_focus):
            # OpenCV computes distances from nonzero pixels to zero pixels,
            # so boundary seeds must be assigned zero.
            dt_input = np.ones((H, W), dtype=np.uint8)
            dt_input[boundary_focus] = 0

            dist = cv2.distanceTransform(
                dt_input,
                cv2.DIST_L2,
                5,
            )

            soft_weight = np.clip(
                1.0 - dist / float(max_dist),
                0.0,
                1.0,
            )

            # Retain only the live side
            soft_weight[~live_mask] = 0.0
        else:
            soft_weight = np.zeros(
                (H, W),
                dtype=np.float32,
            )

        soft_weights.append(
            torch.from_numpy(soft_weight).unsqueeze(0)
        )

    return torch.stack(soft_weights, dim=0).float()


# cutmix
def rand_bbox(size, lam):
    """Randomly generate the coordinates of a rectangular box"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # The center point of the rectangle box is selected uniformly at random
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Boundary clipping
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def trainer_patches_refinement(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # some geometric transformation
    data_transforms = transforms.Compose([
        ResizePadGenerator(output_size=[args.img_size, args.img_size]),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% prob horizontal flip
        transforms.RandomVerticalFlip(p=0.5),  # 50% prob vertical flip
        transforms.RandomRotation(degrees=45),  # Rotate randomly from -45 to 45 degrees
        # Add color jitter
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose([
                                   ResizePadGenerator(output_size=[args.img_size, args.img_size])]))

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    dice_loss = DiceLoss(num_classes)
    boundary_weight = 0.1

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:

        # Branches join cutmix only for the first 100 epochs
        cutmix_active = epoch_num < 100

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # #z-score
            # mean = image_batch.mean()
            # std = image_batch.std()
            # image_batch = (image_batch-mean)/ (std+ 1e-8)

            """
            normal way
            """
            # output logits + death_prob(two branch)
            logits, death_prob = model(image_batch)

            # segmentation loss
            loss_ce = ce_loss(logits, label_batch.long())
            loss_dice = dice_loss(logits, label_batch, softmax=True)

            # edge soft weight
            soft_weight = generate_soft_weight_from_label(label_batch)
            loss_prob = F.mse_loss(death_prob, soft_weight)

            # death boundary binary loss
            with torch.no_grad():
                boundary_mask = get_live_dead_soft_edge(label_batch).cuda()
            prob = torch.softmax(logits, dim=1)
            onehot_label = F.one_hot(label_batch, num_classes=num_classes).permute(0, 3, 1, 2).float()
            loss_boundary = F.binary_cross_entropy(prob * boundary_mask, onehot_label * boundary_mask)

            loss = 0.4 * loss_ce + 0.6 * loss_dice + 0.1 * loss_boundary + 0.3 * loss_prob

            # loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                pred_label = torch.argmax(prob, dim=1, keepdim=True)
                writer.add_image('train/Prediction', pred_label[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

                # Add; with soft weight and death prob
                writer.add_image('train/SoftWeight', soft_weight[1, ...], iter_num)
                writer.add_image('train/DeathProb', death_prob[1, ...], iter_num)

        save_interval = 30
        if epoch_num > int(max_epoch / 6) and (epoch_num + 6) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"