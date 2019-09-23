# coding=utf-8

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
from torchvision import transforms

import utils
from data_image import KnowITImageData
from data_concepts import KnowITConceptsData
from data_facial import KnowITFacesData
from model import VR_ImageFeatures, VR_ImageBOW

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

np.set_printoptions(threshold=sys.maxsize)


def get_params():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_dir", default='Data/', type=str)
    parser.add_argument('--csvtrain', default='data_full_train.csv', help='Training set data file')
    parser.add_argument('--csvval', default='data_full_val.csv', help='Dataset val data file')
    parser.add_argument('--csvtest', default='data_full_test_qtypes.csv', help='Dataset test data file')
    parser.add_argument('--bertembds_ftrain', default='Features/language_bert_train.pckl')
    parser.add_argument('--bertembds_fval', default='Features/language_bert_val.pckl')
    parser.add_argument('--bertembds_ftest', default='Features/language_bert_test.pckl')
    parser.add_argument('--framesdir', default='Data/Frames/')
    parser.add_argument('--idsframes', default='ids_frames.csv')
    parser.add_argument('--list_vcps_objs', default='Concepts/objects_vocab.txt')
    parser.add_argument('--vcpsframes', default='Concepts/knowit_resnet101_faster_rcnn_genome_vcps_all.tsv')
    parser.add_argument('--list_faces_names', default='Faces/people.txt')
    parser.add_argument('--facesframes', default='Faces/knowit_knn_cnn_th060.tsv')

    # Data params
    parser.add_argument('--vision', default='image', help='image | concepts | facial | caption')
    parser.add_argument('--numframes', type=int, default=5)
    parser.add_argument('--img_space', type=int, default=512)
    parser.add_argument('--bert_emb_size', type=int, default=768)

    # Training params
    parser.add_argument('--seed', type=int, default=181)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--device", default='cuda', type=str, help="cuda, cpu")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--workers", default=8)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--nepochs', default=100, help='Number of epochs', type=int)
    parser.add_argument('--patience', default=15, type=int)
    parser.add_argument('--freeVision', default=False, type=bool)
    parser.add_argument('--freeLanguage', default=True, type=bool)
    parser.add_argument('--patParamGroups', default=1, type=int)
    parser.add_argument('--no_cuda', action='store_true')

    return parser.parse_args()


def accuracy_perclass(df, out, label, index):

    qtypes = df['QType'].to_list()

    acc_vis, acc_text, acc_tem, acc_know = 0, 0, 0, 0
    num_vis, num_text, num_tem, num_know = 0, 0, 0, 0

    for o, l, i in zip(out, label, index):

        qtype = qtypes[i]

        if qtype == 'visual':
            num_vis += 1
            if o == l:
                acc_vis += 1
        elif qtype == 'textual':
            num_text += 1
            if o == l :
                acc_text += 1
        elif qtype == 'temporal':
            num_tem += 1
            if o == l:
                acc_tem += 1
        elif qtype == 'knowledge':
            num_know += 1
            if o == l:
                acc_know += 1

    acc_vis = acc_vis / num_vis
    acc_text = acc_text / num_text
    acc_tem = acc_tem / num_tem
    acc_know = acc_know / num_know

    logger.info("Acc visual samples\t%.03f", acc_vis)
    logger.info("Acc textual samples\t%.03f", acc_text)
    logger.info("Acc temporal samples\t%.03f", acc_tem)
    logger.info("Acc knowledge samples\t%.03f", acc_know)


def trainEpoch(train_loader, model, criterion, optimizer, epoch):

    losses = utils.AverageMeter()
    model.train()
    for batch_idx, (input, target) in enumerate(train_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        output = model(*input_var)

        # Compute loss
        train_loss = criterion(output, target_var[0])
        losses.update(train_loss.data.cpu().numpy(), input[0].size(0))

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Print info
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss=losses))

    # Plot loss after all mini-batches have finished
    plotter.plot('loss', 'train', 'Class Loss', epoch, losses.avg)


def valEpoch(val_loader, model, criterion, epoch):

    losses = utils.AverageMeter()
    model.eval()
    for batch_idx, (input, target) in enumerate(val_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        with torch.no_grad():
            output = model(*input_var)

        _, predicted = torch.max(output, 1)

        # Compute loss
        train_loss = criterion(output, target_var[0])
        losses.update(train_loss.data.cpu().numpy(), input[0].size(0))

        # Save predictions to compute accuracy
        if batch_idx == 0:
            out = predicted.data.cpu().numpy()
            label = target[0].cpu().numpy()
        else:
            out = np.concatenate((out,predicted.data.cpu().numpy()),axis=0)
            label = np.concatenate((label,target[0].cpu().numpy()),axis=0)

    # Accuracy
    acc = np.sum(out == label) / len(out)
    logger.info('Validation set: Average loss: {:.4f}\t'
          'Accuracy {acc}'.format(losses.avg, acc=acc))
    plotter.plot('loss', 'val', 'Class Loss', epoch, losses.avg)
    plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)
    return acc


def train(args, outdir, modelname):

    # Set GPU
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(args.device, n_gpu))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Create training directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Model
    if args.vision == 'image':
        train_transforms = transforms.Compose([
            transforms.Resize(256),  # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(256),  # we get only the center of that rescaled
            transforms.RandomCrop(224),  # random crop within the center crop (data augmentation)
            transforms.RandomHorizontalFlip(),  # random horizontal flip (data augmentation)
            transforms.ToTensor(),  # to pytorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                                 std=[0.229, 0.224, 0.225])
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(256),  # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(224),  # we get only the center of that rescaled
            transforms.ToTensor(),  # to pytorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                                 std=[0.229, 0.224, 0.225])
        ])
        trainDataObject = KnowITImageData(args, split='train', transform=train_transforms)
        valDataObject = KnowITImageData(args, split='val', transform=val_transforms)
        model = VR_ImageFeatures(args)
    elif args.vision == 'concepts':
        trainDataObject = KnowITConceptsData(args, split='train')
        valDataObject = KnowITConceptsData(args, split='val')
        num_concepts = trainDataObject.get_num_objects()
        model = VR_ImageBOW(args, num_concepts)
    elif args.vision == 'facial':
        trainDataObject = KnowITFacesData(args, split='train')
        valDataObject = KnowITFacesData(args, split='val')
        num_people = trainDataObject.get_num_people()
        model = VR_ImageBOW(args, num_people)

    if args.device == "cuda":
        model.cuda()

    # Optimizer
    if args.vision == 'image':
        vision_params = list(map(id, model.resnet.parameters()))
        base_params   = filter(lambda p: id(p) not in vision_params, model.parameters())
        optimizer = torch.optim.SGD([
                    {'params': base_params},
                    {'params': model.resnet.parameters(), 'lr': args.lr*args.freeVision },
                ], lr=args.lr*args.freeLanguage, momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Loss
    class_loss = nn.CrossEntropyLoss().cuda()

    # Data Loaders for training and validation
    train_loader = torch.utils.data.DataLoader(trainDataObject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    logger.info('Training loader with %d samples' % train_loader.__len__())
    val_loader = torch.utils.data.DataLoader(valDataObject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    logger.info('Validation loader with %d samples' % val_loader.__len__())

    # Now, let's start the training process!
    logger.info('Training...')
    valtrack = 0 # measures patience
    valgrouptrack = 0
    best_val = 0
    for epoch in range(0, args.nepochs):

        # Compute a training epoch
        trainEpoch(train_loader, model, class_loss, optimizer, epoch)

        # Compute a validation epoch
        current_val = valEpoch(val_loader, model, class_loss, epoch)

        # check patience
        is_best = current_val > best_val
        best_val = max(current_val, best_val)
        if not is_best:
            valtrack += 1
        else:
            valtrack = 0
        if valtrack >= args.patience:
            break

        logger.info('** Validation information: %f (this accuracy) - %f (best accuracy) - %d (patience valtrack)' % (current_val, best_val, valtrack))

        # check if change group params fine-tunning
        if args.vision == 'image':
            if valgrouptrack >= args.patParamGroups:
                args.freeVision = args.freeComment
                args.freeComment = not (args.freeVision)
                optimizer.param_groups[0]['lr'] = args.lr * args.freeComment
                optimizer.param_groups[1]['lr'] = args.lr * args.freeVision
                logger.info('Initial base params lr: %f' % optimizer.param_groups[0]['lr'])
                logger.info('Initial vision lr: %f' % optimizer.param_groups[1]['lr'])
                args.patParamGroups = 3
                valgrouptrack = 0

        if is_best:
            state =  {'state_dict': model.state_dict(),
                'best_val': best_val,
                'optimizer': optimizer.state_dict(),
                'valtrack': valtrack,
                'curr_val': current_val}
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            filename = os.path.join(outdir, modelname)
            torch.save(state, filename)


def evaluate(args, modeldir, modelname):

    # Model
    if args.vision == 'image':
        test_transforms = transforms.Compose([
            transforms.Resize(256),  # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(224),  # we get only the center of that rescaled
            transforms.ToTensor(),  # to pytorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                                 std=[0.229, 0.224, 0.225])
        ])
        testDataObject = KnowITImageData(args, split='test', transform=test_transforms)
        model = VR_ImageFeatures(args)
    elif args.vision == 'concepts':
        testDataObject = KnowITConceptsData(args, split='test')
        num_concepts = testDataObject.get_num_objects()
        model = VR_ImageBOW(args, num_concepts)
    elif args.vision == 'facial':
        testDataObject = KnowITFacesData(args, split='test')
        num_people = testDataObject.get_num_people()
        model = VR_ImageBOW(args, num_people)

    if args.device == "cuda":
        model.cuda()

    # Load best model
    logger.info("=> loading checkpoint from '{}'".format(modeldir))
    checkpoint = torch.load(os.path.join(modeldir, modelname))
    model.load_state_dict(checkpoint['state_dict'])

    # Data Loader
    test_loader = torch.utils.data.DataLoader(testDataObject, batch_size=args.batch_size, shuffle=False, pin_memory=(not args.no_cuda), num_workers=args.workers)
    logger.info('Evaluation loader with %d samples' % test_loader.__len__())

    # Switch to evaluation mode & compute test samples embeddings
    model.eval()
    for i, (input, target) in enumerate(test_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        with torch.no_grad():
            output = model(*input_var)
        _, predicted = torch.max(output, 1)

        # Store outpputs
        if i==0:
            out = predicted.data.cpu().numpy()
            label = target[0].cpu().numpy()
            index = target[1].cpu().numpy()
        else:
            out = np.concatenate((out,predicted.data.cpu().numpy()),axis=0)
            label = np.concatenate((label,target[0].cpu().numpy()),axis=0)
            index = np.concatenate((index, target[1].cpu().numpy()), axis=0)

    # Compute Accuracy
    acc = np.sum(out == label)/len(out)
    logger.info('*' *20)
    logger.info('Model in %s' %modeldir)
    df = pd.read_csv('Data/data_full_test_qtypes.csv', delimiter='\t')
    accuracy_perclass(df, out, label, index)
    logger.info('Overall Accuracy\t%.03f' % acc)
    logger.info('*' * 20)



if __name__ == "__main__":

    # Parameters
    args = get_params()
    assert args.vision in ['image', 'concepts', 'facial'], "Incorrect image features."

    # Model name and path
    train_name = 'AnswerPrediction_%s' % (args.vision)
    outdir = os.path.join('Training/VideoReasoning/', train_name)
    if args.vision == 'image':
        modelname = 'ROCK-image-weights.pth.tar'
    elif args.vision == 'concepts':
        modelname = 'ROCK-concepts-weights.pth.tar'
    elif args.vision == 'facial':
        modelname = 'ROCK-concepts-weights.pth.tar'

    # Training
    if not os.path.isfile(os.path.join(outdir, modelname)):
        global plotter
        plotter = utils.VisdomLinePlotter(env_name=train_name)
        train(args, outdir, modelname)

    # Evaluation
    evaluate(args, outdir, modelname)
