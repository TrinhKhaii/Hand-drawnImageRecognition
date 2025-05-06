"""
@author: Trinh Khai Truong 
"""
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from model import QuickDraw
from dataset import MyDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def get_args():
    parser = argparse.ArgumentParser(description="Train a QuickDraw model")
    parser.add_argument("--data-path", type=str, default="dataset", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--image_size", type=int, default=28, help="Image size for training")
    parser.add_argument("--ratio", type=float, default=0.8, help="the ratio between training and test sets")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")
    parser.add_argument("--checkpoint_dir", type=str, default="trained_models2", help="Path to the checkpoint for resuming training")
    parser.add_argument("--tensorboard_dir", type=str, default="tensorboard2", help="Path to save tensorboard logs")
    args = parser.parse_args()
    return args


def train(args):
    path = args.data_path
    isTrain = True

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    if not os.path.isdir(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    writer = SummaryWriter(args.tensorboard_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = QuickDraw()
    model.to(device)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == "adamw": 
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    criterion  = nn.CrossEntropyLoss()

    train_dataset = MyDataset(path, isTrain, ratio=args.ratio)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = False,
        num_workers = args.num_workers,
        pin_memory=True
    )

    val_dataset = MyDataset(path, not isTrain, ratio=args.ratio)
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = False,
        num_workers = args.num_workers,
        pin_memory=True
    )

    best_accuracy = 0.0
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        progess_train_bar = tqdm(train_dataloader, colour='green')
        all_losses = []
        all_train_labels = []
        all_train_predictions = []
        for iter, (images, labels) in enumerate(progess_train_bar):
            images = images.to(device)
            labels = labels.to(device).squeeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_losses.append(loss.item())
            val_loss = np.mean(all_losses)
            train_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            all_train_labels.extend(labels.cpu().tolist())
            all_train_predictions.extend(predicted.cpu().tolist())

            writer.add_scalar('Train/Loss', val_loss, epoch * len(train_dataloader) + iter)

        avg_train_loss = train_loss / len(train_dataloader)
        train_accuracy = accuracy_score(all_train_labels, all_train_predictions)
        writer.add_scalar('Train/Accuracy', train_accuracy, epoch)

        model.eval()
        val_loss = 0.0
        all_losses = []
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for iter, (images, labels) in enumerate(val_dataloader):
                images = images.to(device)
                labels = labels.to(device).squeeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                all_losses.append(loss.item())
                val_loss = np.mean(all_losses)

                predicted = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(predicted.cpu().tolist())

                writer.add_scalar('Validation/Loss', val_loss, epoch * len(train_dataloader) + iter)

            val_accuracy = accuracy_score(all_labels, all_predictions)
            val_precision = precision_score(all_labels, all_predictions, average='macro')
            val_recall = recall_score(all_labels, all_predictions, average='macro')
               
            scheduler.step(val_accuracy)

            print("Epoch {}/{} - Train Loss: {:.4f} - Train Accuracy: {:.4f} - Val Loss: {:.4f} - Val Accuracy: {:.4f} = Val Precision: {:.4f} - Val Recall: {:.4f}".format(
                epoch + 1,
                args.num_epochs,
                avg_train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                val_precision,
                val_recall
            ))

            writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
            writer.add_scalar('Validation/Precision', val_precision, epoch)
            writer.add_scalar('Validation/Recall', val_recall, epoch)
            
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'last_checkpoint.pt'))
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_checkpoint.pt'))
                print("Best model saved with accuracy: {:.4f}".format(best_accuracy))
                
    writer.close()



if __name__ == '__main__':
    args = get_args()
    train(args)