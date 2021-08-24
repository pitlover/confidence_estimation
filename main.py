from model import resnet
from model import densenet_BC
from model import vgg

import losses
import time
import data as dataset
import crl_utils
import metrics
import utils
import train
import data
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Confidence Aware Learning')
parser.add_argument('--epochs', default=300, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='original : 128 Batch size for training')
parser.add_argument('--data', default='cifar10', type=str, help='Dataset name to use [cifar10, cifar100, svhn]')
parser.add_argument('--model', default='res', type=str, help='Models name to use [res, dense, vgg]')
parser.add_argument('--loss', default='CE', type=str, help='Loss to use [CE, CRL, Focal, MS, Contrastive, Triplet, NPair, Avg]')
parser.add_argument('--cal', default='Default', type=str, help='Calculate Correctness, Confidence')
parser.add_argument('--rank_target', default='softmax', type=str, help='Rank_target name to use [softmax, margin, entropy]')
parser.add_argument('--rank_weight', default=0.0, type=float, help='Rank loss weight')
parser.add_argument('--lr', default = 1e-4, type =float,help = 'Learning rate setting')
parser.add_argument('--weight-decay', default = 1e-4, type =float, help = 'Weight decay setting')
parser.add_argument('--lr-decay-step', default = 10, type =int, help = 'Learning decay step setting')
parser.add_argument('--lr-decay-gamma', default = 0.5, type =float, help = 'Learning decay gamma setting')
parser.add_argument('--data_path', default='/mnt/hdd0/jiizero/', type=str, help='Dataset directory')
parser.add_argument('--save_path', default='./test/', type=str, help='Savefiles directory')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--sort', action='store_true', help='sample sort-> hard sample and easy sample')
parser.add_argument('--sort_mode', type=float, default=0, help='sample sort-> 0: acc/conf 1: conf')
parser.add_argument('--valid', action='store_true', help='is_use_validset')
parser.add_argument('--calibrate', action='store_true', help='is_use_validset')
parser.add_argument('--b', type=float, default=None, help='Flood level')
parser.add_argument('--ts', type=float, default=None, help='Temperature Scaling')
parser.add_argument('--mixup', type=float, default=None, help='Mixup with alpha')
parser.add_argument('--rot', action='store_true', help='RotNet')
parser.add_argument('--ji_conf', action='store_true', help='b*(1.5-conf)')
parser.add_argument('--minus_1_conf', action='store_true', help='b*(1/conf)')
parser.add_argument('--ji_acc_conf', action='store_true', help='b*(acc/conf)')
parser.add_argument('--ji_wj', type=float, default=0, help='ce + |soft-acc|')
parser.add_argument('--mode', type=float, default=0, help='batch : 0 (default), sample : 1')
args = parser.parse_args()

def main():
    file_name = "./flood_graph/150_250/128/500/ji_sort/1_conf/sample-wised/default/{}/".format(args.b)
    start = time.time()
    # set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    # check save path
    save_path = file_name
    # save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # make dataloader
    if args.valid == True:
        train_loader, valid_loader, test_loader, test_onehot, test_label = dataset.get_valid_loader(args.data, args.data_path,
                                                                                args.batch_size)

    else:
        train_loader, train_onehot, train_label, test_loader, test_onehot, test_label = dataset.get_loader(args.data, args.data_path, args.batch_size)

    # set num_class
    if args.data == 'cifar100':
        num_class = 100
    else:
        num_class = 10

    # set num_classes
    model_dict = {
        "num_classes": num_class,
    }

    # set model
    if args.model == 'res':
        model = resnet.resnet110(**model_dict).cuda()
    elif args.model == 'dense':
        model = densenet_BC.DenseNet3(depth=100,
                                      num_classes=num_class,
                                      growth_rate=12,
                                      reduction=0.5,
                                      bottleneck=True,
                                      dropRate=0.0).cuda()
    elif args.model == 'vgg':
        model = vgg.vgg16(**model_dict).cuda()


    # set criterion
    if args.loss == 'MS':
        cls_criterion = losses.MultiSimilarityLoss().cuda()
    elif args.loss == 'Contrastive':
        cls_criterion = losses.ContrastiveLoss().cuda()
    elif args.loss == 'Triplet':
        cls_criterion = losses.TripletLoss().cuda()
    elif args.loss == 'NPair':
        cls_criterion = losses.NPairLoss().cuda()
    elif args.loss == 'Focal':
        cls_criterion = losses.FocalLoss(gamma=3.0).cuda()
    else:
        if args.mode == 0:
            cls_criterion = nn.CrossEntropyLoss().cuda()
        else:
            cls_criterion = nn.CrossEntropyLoss(reduction="none").cuda()



    ranking_criterion = nn.MarginRankingLoss(margin=0.0).cuda()

    # set optimizer (default:sgd)
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=5e-4,
                          # weight_decay=0.0001,
                          nesterov=False)

    # optimizer = optim.SGD(model.parameters(),
    #                       lr=float(args.lr),
    #                       momentum=0.9,
    #                       weight_decay=args.weight_decay,
    #                       nesterov=False)

    # set scheduler
    # scheduler = MultiStepLR(optimizer,
    #                         milestones=[500, 750],
    #                         gamma=0.1)

    scheduler = MultiStepLR(optimizer,
                            milestones=[150, 250],
                            gamma=0.1)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    # make logger
    train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
    result_logger = utils.Logger(os.path.join(save_path, 'result.log'))

    # make History Class
    correctness_history = crl_utils.History(len(train_loader.dataset))

    ## define matrix
    if args.data == 'cifar':
        matrix_idx_confidence = [[_] for _ in range(50000)]
        matrix_idx_iscorrect = [[_] for _ in range(50000)]
    else:
        matrix_idx_confidence = [[_] for _ in range(73257)]
        matrix_idx_iscorrect = [[_] for _ in range(73257)]


    # write csv
    #'''
    import csv
    f = open('{}/logs_{}_{}.txt'.format(file_name, args.b, args.epochs),'w', newline='')
    f.write("location = {}\n\n".format(file_name)+str(args))

    f0 = open('{}/Test_confidence_{}_{}.csv'.format(file_name, args.b, args.epochs), 'w', newline='')
    # f0 = open('./baseline_graph/150_250/128/500/Test_confidence_{}_{}.csv'.format(args.b, args.epochs), 'w', newline='')
    # f0 = open('./CRL_graph/150_250/Test_confidence_{}_{}.csv'.format(args.b, args.epochs), 'w', newline='')

    wr_conf_test = csv.writer(f0)
    header = [_ for _ in range(args.epochs + 1)]
    header[0] = 'Epoch'
    wr_conf_test.writerows([header])

    f1 = open('{}/Train_confidence_{}_{}.csv'.format(file_name, args.b, args.epochs), 'w', newline='')
    # f1 = open('./baseline_graph/150_250/128/500/Train_confidence_{}_{}.csv'.format(args.b, args.epochs), 'w', newline='')
    # f1 = open('./CRL_graph/150_250/Train_confidence_{}_{}.csv'.format(args.b, args.epochs), 'w', newline='')

    wr = csv.writer(f1)
    header = [_ for _ in range(args.epochs + 1)]
    header[0] = 'Epoch'
    wr.writerows([header])

    f2 = open('{}/Train_Flood_{}_{}_{}.csv'.format(file_name, args.data, args.b, args.epochs), 'w', newline='')
    # f2 = open('./baseline_graph/150_250/128/500/Train_Base_{}_{}_{}.csv'.format(args.data, args.b, args.epochs), 'w', newline='')
    # f2 = open('./CRL_graph/150_250/Train_Flood_{}_{}_{}.csv'.format(args.data, args.b, args.epochs), 'w', newline='')

    wr_train = csv.writer(f2)
    header = [_ for _ in range(args.epochs+1)]
    header[0] = 'Epoch'
    wr_train.writerows([header])

    f3 = open('{}/Test_Flood_{}_{}_{}.csv'.format(file_name, args.data, args.b, args.epochs), 'w', newline='')
    # f3 = open('./baseline_graph/150_250/128/500/Test_Base_{}_{}_{}.csv'.format(args.data, args.b, args.epochs), 'w', newline='')
    # f3 = open('./CRL_graph/150_250/Test_Flood_{}_{}_{}.csv'.format(args.data, args.b, args.epochs), 'w', newline='')

    wr_test = csv.writer(f3)
    header = [_ for _ in range(args.epochs+1)]
    header[0] = 'Epoch'
    wr_test.writerows([header])
    #'''

    # start Train
    best_valid_acc = 0
    test_ece_report = []
    test_acc_report = []
    test_nll_report = []
    test_over_con99_report = []
    test_e99_report = []
    test_cls_loss_report = []

    train_ece_report = []
    train_acc_report = []
    train_nll_report = []
    train_over_con99_report = []
    train_e99_report = []
    train_cls_loss_report = []
    train_rank_loss_report = []
    train_total_loss_report = []

    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        matrix_idx_confidence, matrix_idx_iscorrect, idx, iscorrect, confidence, target, cls_loss_tr, rank_loss_tr, batch_correctness, total_confidence, total_correctness = \
            train.train(matrix_idx_confidence, matrix_idx_iscorrect, train_loader,
                    model,
                    wr,
                    cls_criterion,
                    ranking_criterion,
                    optimizer, 
                    epoch,
                    correctness_history,
                    train_logger,
                    args)


        if args.rank_weight != 0.0:
            print("RANK ", rank_loss_tr)
            total_loss_tr = cls_loss_tr + rank_loss_tr

        if args.valid == True:
            idx, iscorrect, confidence, target, cls_loss_val, acc = train.valid(valid_loader,
                                                             model,
                                                             cls_criterion,
                                                             ranking_criterion,
                                                             optimizer,
                                                             epoch,
                                                             correctness_history,
                                                             train_logger,
                                                             args)
            if acc > best_valid_acc:
                best_valid_acc = acc
                print("*** Update Best Acc ***")



        # save model
        if epoch == args.epochs:
            torch.save(model.state_dict(),
                       os.path.join(save_path, 'model.pth'))

        print("########### Train ###########")
        acc_tr, aurc_tr, eaurc_tr, aupr_tr, fpr_tr, ece_tr, nll_tr, brier_tr, E99_tr, over_99_tr, cls_loss_tr = metrics.calc_metrics(train_loader,
                                                                                          train_label,
                                                                                           train_onehot,
                                                                                          model,
                                                                                          cls_criterion, args)

        if args.sort == True and epoch == 260:
        #if args.sort == True:
            train_loader = dataset.sort_get_loader(args.data, args.data_path,
                                                    args.batch_size, idx, np.array(target),
                                                    iscorrect, batch_correctness, total_confidence, total_correctness, np.array(confidence), epoch, args)

        train_acc_report.append(acc_tr)
        train_nll_report.append(nll_tr*10)
        train_ece_report.append(ece_tr)
        train_over_con99_report.append(over_99_tr)
        train_e99_report.append(E99_tr)
        train_cls_loss_report.append(cls_loss_tr)

        if args.rank_weight != 0.0:
            train_total_loss_report.append(total_loss_tr)
            train_rank_loss_report.append(rank_loss_tr)
        print("CLS ", cls_loss_tr)

        # finish train
        print("########### Test ###########")
        # calc measure
        acc_te, aurc_te, eaurc_te, aupr_te, fpr_te, ece_te, nll_te, brier_te, E99_te, over_99_te, cls_loss_te = metrics.calc_metrics(test_loader,
                                                                            test_label,
                                                                            test_onehot,
                                                                            model,
                                                                            cls_criterion, args)
        test_ece_report.append(ece_te)
        test_acc_report.append(acc_te)
        test_nll_report.append(nll_te*10)
        test_over_con99_report.append(over_99_te)
        test_e99_report.append(E99_te)
        test_cls_loss_report.append(cls_loss_te)

        print("CLS ", cls_loss_te)
        print("############################")

    # for idx in matrix_idx_confidence:
    #     wr.writerow(idx)

    #'''
    # draw graph
    df = pd.DataFrame()
    df['epoch'] = [i for i in range(1, args.epochs + 1)]
    df['test_ece'] = test_ece_report
    df['train_ece'] = train_ece_report
    fig_loss = plt.figure(figsize=(35, 35))
    fig_loss.set_facecolor('white')
    ax = fig_loss.add_subplot()

    ax.plot(df['epoch'], df['test_ece'], df['epoch'], df['train_ece'], linewidth=10)
    ax.legend(['Test', 'Train'], loc = 2, prop={'size': 60})
    plt.title('[FL] ECE per epoch', fontsize=80)
    # plt.title('[BASE] ECE per epoch', fontsize=80)
    # plt.title('[CRL] ECE per epoch', fontsize=80)
    plt.xlabel('Epoch', fontsize=70)
    plt.ylabel('ECE', fontsize=70)
    plt.ylim([0, 1])
    plt.setp(ax.get_xticklabels(), fontsize=30)
    plt.setp(ax.get_yticklabels(), fontsize=30)
    plt.savefig('{}/{}_{}_ECE_lr_{}.png'.format(file_name, args.model, args.b, args.epochs))
    # plt.savefig('./baseline_graph/150_250/128/500/{}_{}_ECE_lr_{}.png'.format(args.model, args.b, args.epochs))
    # plt.savefig('./CRL_graph/150_250/{}_{}_ECE_lr_{}.png'.format(args.model, args.b, args.epochs))

    df2 = pd.DataFrame()
    df2['epoch'] = [i for i in range(1, args.epochs + 1)]
    df2['test_acc'] = test_acc_report
    df2['train_acc'] = train_acc_report
    fig_acc = plt.figure(figsize=(35, 35))
    fig_acc.set_facecolor('white')
    ax = fig_acc.add_subplot()

    ax.plot(df2['epoch'], df2['test_acc'], df2['epoch'], df2['train_acc'], linewidth=10)
    ax.legend(['Test', 'Train'], loc = 2, prop={'size': 60})
    plt.title('[FL] Accuracy per epoch', fontsize=80)
    # plt.title('[BASE] Accuracy per epoch', fontsize=80)
    # plt.title('[CRL] Accuracy per epoch', fontsize=80)
    plt.xlabel('Epoch', fontsize=70)
    plt.ylabel('Accuracy', fontsize=70)
    plt.ylim([0, 100])
    plt.setp(ax.get_xticklabels(), fontsize=30)
    plt.setp(ax.get_yticklabels(), fontsize=30)
    plt.savefig('{}/{}_{}_acc_lr_{}.png'.format(file_name, args.model, args.b, args.epochs))
    # plt.savefig('./baseline_graph/150_250/128/500/{}_{}_acc_lr_{}.png'.format(args.model, args.b, args.epochs))
    # plt.savefig('./CRL_graph/150_250/{}_{}_acc_lr_{}.png'.format(args.model, args.b, args.epochs))

    df3 = pd.DataFrame()
    df3['epoch'] = [i for i in range(1, args.epochs + 1)]
    df3['test_nll'] = test_nll_report
    df3['train_nll'] = train_nll_report
    fig_acc = plt.figure(figsize=(35, 35))
    fig_acc.set_facecolor('white')
    ax = fig_acc.add_subplot()

    ax.plot(df3['epoch'], df3['test_nll'], df3['epoch'], df3['train_nll'], linewidth=10)
    ax.legend(['Test', 'Train'], loc = 2, prop={'size': 60})
    plt.title('[FL] NLL per epoch', fontsize=80)
    # plt.title('[BASE] NLL per epoch', fontsize=80)
    # plt.title('[CRL] NLL per epoch', fontsize=80)
    plt.xlabel('Epoch', fontsize=70)
    plt.ylabel('NLL', fontsize=70)
    plt.ylim([0, 45])
    plt.setp(ax.get_xticklabels(), fontsize=30)
    plt.setp(ax.get_yticklabels(), fontsize=30)
    plt.savefig('{}/{}_{}_nll_lr_{}.png'.format(file_name, args.model, args.b, args.epochs))
    # plt.savefig('./baseline_graph/150_250/128/500/{}_{}_nll_lr_{}.png'.format(args.model, args.b, args.epochs))
    # plt.savefig('./CRL_graph/150_250/{}_{}_nll_lr_{}.png'.format(args.model, args.b, args.epochs))

    df4 = pd.DataFrame()
    df4['epoch'] = [i for i in range(1, args.epochs + 1)]
    df4['test_over_con99'] = test_over_con99_report
    df4['train_over_con99'] = train_over_con99_report
    fig_acc = plt.figure(figsize=(35, 35))
    fig_acc.set_facecolor('white')
    ax = fig_acc.add_subplot()

    ax.plot(df4['epoch'], df4['test_over_con99'], df4['epoch'], df4['train_over_con99'], linewidth=10)
    ax.legend(['Test', 'Train'], loc=2, prop={'size': 60})
    plt.title('[FL] Over conf99 per epoch', fontsize=80)
    # plt.title('[BASE] Over conf99 per epoch', fontsize=80)
    # plt.title('[CRL] Over conf99 per epoch', fontsize=80)
    plt.xlabel('Epoch', fontsize=70)
    plt.ylabel('Over con99', fontsize=70)
    if args.data == 'cifar10' or args.data == 'cifar100':
        plt.ylim([0, 50000])
    else:
        plt.ylim([0, 73257])

    plt.setp(ax.get_xticklabels(), fontsize=30)
    plt.setp(ax.get_yticklabels(), fontsize=30)
    plt.savefig('{}/{}_{}_over_conf99_lr_{}.png'.format(file_name, args.model, args.b, args.epochs))
    # plt.savefig('./baseline_graph/150_250/128/500/{}_{}_over_conf99_lr_{}.png'.format(args.model, args.b, args.epochs))
    # plt.savefig('./CRL_graph/150_250/{}_{}_over_conf99_lr_{}.png'.format(args.model, args.b, args.epochs))

    df5 = pd.DataFrame()
    df5['epoch'] = [i for i in range(1, args.epochs + 1)]
    df5['test_e99'] = test_e99_report
    df5['train_e99'] = train_e99_report
    fig_acc = plt.figure(figsize=(35, 35))
    fig_acc.set_facecolor('white')
    ax = fig_acc.add_subplot()

    ax.plot(df5['epoch'], df5['test_e99'], df5['epoch'], df5['train_e99'], linewidth=10)
    ax.legend(['Test', 'Train'], loc=2, prop={'size': 60})
    plt.title('[FL] E99 per epoch', fontsize=80)
    # plt.title('[BASE] E99 per epoch', fontsize=80)
    # plt.title('[CRL] E99 per epoch', fontsize=80)
    plt.xlabel('Epoch', fontsize=70)
    plt.ylabel('E99', fontsize=70)
    plt.ylim([0, 0.2])
    plt.setp(ax.get_xticklabels(), fontsize=30)
    plt.setp(ax.get_yticklabels(), fontsize=30)
    plt.savefig('{}/{}_{}_E99_flood_lr_{}.png'.format(file_name,args.model, args.b, args.epochs))
    # plt.savefig('./baseline_graph/150_250/128/500/{}_{}_E99_flood_lr_{}.png'.format(args.model, args.b, args.epochs))
    # plt.savefig('./CRL_graph/150_250/{}_{}_E99_flood_lr_{}.png'.format(args.model, args.b, args.epochs))

    df5 = pd.DataFrame()
    df5['epoch'] = [i for i in range(1, args.epochs + 1)]
    df5['test_cls_loss'] = test_cls_loss_report
    df5['train_cls_loss'] = train_cls_loss_report
    fig_acc = plt.figure(figsize=(35, 35))
    fig_acc.set_facecolor('white')
    ax = fig_acc.add_subplot()

    ax.plot(df5['epoch'], df5['test_cls_loss'], df5['epoch'], df5['train_cls_loss'], linewidth=10)
    ax.legend(['Test', 'Train'], loc=2, prop={'size': 60})
    plt.title('[FL] CLS_loss per epoch', fontsize=80)
    # plt.title('[BASE] CLS_loss per epoch', fontsize=80)
    # plt.title('[CRL] CLS_loss per epoch', fontsize=80)
    plt.xlabel('Epoch', fontsize=70)
    plt.ylabel('Loss', fontsize=70)
    plt.ylim([0, 5])
    plt.setp(ax.get_xticklabels(), fontsize=30)
    plt.setp(ax.get_yticklabels(), fontsize=30)
    plt.savefig('{}/{}_{}_cls_loss_flood_lr_{}.png'.format(file_name, args.model, args.b, args.epochs))
    # plt.savefig('./baseline_graph/150_250/128/500/{}_{}_cls_loss_flood_lr_{}.png'.format(args.model, args.b, args.epochs))
    # plt.savefig('./CRL_graph/150_250/{}_{}_cls_loss_flood_lr_{}.png'.format(args.model, args.b, args.epochs))

    if args.rank_weight != 0.0:
        df6 = pd.DataFrame()
        df6['epoch'] = [i for i in range(1, args.epochs + 1)]
        df6['train_cls_loss'] = train_cls_loss_report
        df6['train_rank_loss'] = train_rank_loss_report
        df6['train_total_loss'] = train_total_loss_report
        fig_acc = plt.figure(figsize=(35, 35))
        fig_acc.set_facecolor('white')
        ax = fig_acc.add_subplot()

        ax.plot(df6['epoch'], df6['train_cls_loss'], df6['epoch'], df6['train_rank_loss'], df6['epoch'], df6['train_total_loss'], linewidth=10)
        ax.legend(['CLS', 'Rank', 'Total'], loc=2, prop={'size': 60})
        plt.title('[FL] CLS_loss per epoch', fontsize=80)
        plt.xlabel('Epoch', fontsize=70)
        plt.ylabel('Loss', fontsize=70)
        # plt.ylim([0, 5])
        plt.setp(ax.get_xticklabels(), fontsize=30)
        plt.setp(ax.get_yticklabels(), fontsize=30)
        plt.savefig('./CRL_graph/150_250/{}_{}_cls_loss_flood_lr_{}.png'.format(args.model, args.b, args.epochs))

    test_acc_report.insert(0, 'ACC')
    test_ece_report.insert(0, 'ECE')
    test_nll_report.insert(0, 'NLL')
    test_over_con99_report.insert(0, 'Over_conf99')
    test_e99_report.insert(0, 'E99')
    test_cls_loss_report.insert(0, 'CLS')
    wr_test.writerow(test_acc_report)
    wr_test.writerow(test_ece_report)
    wr_test.writerow(test_nll_report)
    wr_test.writerow(test_over_con99_report)
    wr_test.writerow(test_e99_report)
    wr_test.writerow(test_cls_loss_report)

    train_acc_report.insert(0, 'ACC')
    train_ece_report.insert(0, 'ECE')
    train_nll_report.insert(0, 'NLL')
    train_over_con99_report.insert(0, 'Over_conf99')
    train_e99_report.insert(0, 'E99')
    train_cls_loss_report.insert(0, 'CLS')

    wr_train.writerow(train_acc_report)
    wr_train.writerow(train_ece_report)
    wr_train.writerow(train_nll_report)
    wr_train.writerow(train_over_con99_report)
    wr_train.writerow(train_e99_report)
    wr_train.writerow(train_cls_loss_report)


    if args.rank_weight != 0.0:
        train_rank_loss_report.insert(0, 'Rank')
        train_total_loss_report.insert(0, 'Total')
        wr_train.writerow(train_rank_loss_report)
        wr_train.writerow(train_total_loss_report)

    #'''


    # result write
    result_logger.write([acc_te, aurc_te*1000, eaurc_te*1000, aupr_te*100, fpr_te*100, ece_te*100, nll_te*10, brier_te*100, E99_te*100])
    if args.valid == True:
        print("Best Valid Acc : {}".format(acc))
    print("Flood Level: {}".format(args.b))
    print("Sort : {}".format(args.sort))
    print("Sort Mode : {}".format(args.sort_mode))
    print("TIME : ", time.time() - start)
if __name__ == "__main__":
    main()



