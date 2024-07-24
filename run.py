#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train import train_prednet
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--cls', default=6, type=int, help='number of cycles')
parser.add_argument('--model', default='PredNet', help= 'models to train')
parser.add_argument('--dataset', default='MNIST', help= 'dataset to train')
parser.add_argument('--gpunum', default=2, type=int, help='number of gpu used to train the model')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
args = parser.parse_args()

train_prednet(model=args.model, cls=args.cls, gpunum=args.gpunum, lr=args.lr, dataset=args.dataset)

