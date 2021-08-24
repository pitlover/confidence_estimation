import crl_utils
import utils
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def train(matrix_idx_confidence, matrix_idx_iscorrect, loader, model, wr, criterion_cls, criterion_ranking, optimizer, epoch, history, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    cls_losses = utils.AverageMeter()          ## cross entropy loss
    ranking_losses = utils.AverageMeter()      ## marginranking loss
    ji_wj_losses = utils.AverageMeter()        ## JI_WJ
    end = time.time()

    print("*** Training ***")
    model.train()

    all_idx = []
    all_iscorrect = []
    all_confidence = []
    all_target = []

    ## 원본 이미지, 라벨 저장

    for i, (input, target, idx) in enumerate(loader):   ## batchsize = 128
    # for i, (input, target) in enumerate(loader):   ## batchsize = 128
        data_time.update(time.time() - end)
        input, target = input.cuda(), target.cuda()
        confidence = []
        all_idx.extend(idx.tolist())
        all_target.extend(target.tolist())

        ##mixup
        if args.mixup is not None:
            input, target_a, target_b, lam = utils.mixup_data(input, target, args.mixup, True)
            input, target_a, target_b = map(Variable, (input, target_a, target_b))

        output = model(input)

        if args.ts is not None:
            temp = torch.nn.Parameter(torch.ones(1) * args.ts)
            ts = temp.unsqueeze(1).expand(output.size(0), output.size(1)).cuda()

            output = output / ts
        # NaN alert
        assert torch.all(output == output)

        for a in range(len(input)):
            wr.writerow([str(idx[a].item()), str(target[a].item())])

        # record loss and accuracy
        prec, correct = utils.accuracy(output, target)


        # # compute ranking target value normalize (0 ~ 1) range
        # max(softmax)
        if args.rank_target == 'softmax':
            conf = F.softmax(output, dim=1)
            confidence, prediction = conf.max(dim=1)        ## predictin : 예측 class, confidence : 그때의 confidence

        # entropy
        elif args.rank_target == 'entropy':
            if args.data == 'cifar100':
                value_for_normalizing = 4.605170
            else:
                value_for_normalizing = 2.302585
            confidence = crl_utils.negative_entropy(output,
                                                    normalize=True,
                                                    max_value=value_for_normalizing)

        # margin
        elif args.rank_target == 'margin':
            conf, _ = torch.topk(F.softmax(output), 2, dim=1)
            conf[:,0] = conf[:,0] - conf[:,1]
            confidence = conf[:, 0]

        # correctness count update
        if args.loss == "CRL" or args.cal == "Cor":
            history.correctness_update(idx, correct, output)

        # Avg confidence update
        if args.cal == "Conf":
            history.confidence_update(idx, confidence, output)

        for a in range(len(input)):
            matrix_idx_confidence[idx[a]].append(confidence[a].item())

        all_iscorrect.extend(map(int, correct))
        all_confidence.extend(confidence.tolist())


        # make input pair
        rank_input1 = confidence
        rank_input2 = torch.roll(confidence, -1)
        idx2 = torch.roll(idx, -1)

        # calc target, margin
        rank_target, rank_margin, acc, correctness= history.get_target_margin(idx, idx2) ## rank_target : 누가 더 크냐 1, 0, -1 / rank_margin : 옳게 맞춘 횟수의 차이
        # print(rank_target, rank_margin)

        rank_target_nonzero = rank_target.clone()
        # print("rank_target_nonzero", rank_target_nonzero)
        rank_target_nonzero[rank_target_nonzero == 0] = 1 ## rank_target 에서 0을 다 1로 바꿈
        # print("rank_target_nonzero", rank_target_nonzero)
        rank_input2 = rank_input2 + rank_margin / rank_target_nonzero
        # print(rank_input2)
        # ranking loss // margin rankingloss
        ranking_loss = criterion_ranking(rank_input1,
                                         rank_input2,
                                         rank_target)
        # total loss
        ji_loss = 0
        if args.mixup is not None:
            cls_loss = utils.mixup_criterion(criterion_cls, output, target_a, target_b, lam)

        else:
            cls_loss = criterion_cls(output, target) # (128, 1)
            if args.b != None:
                # print("******************************")
                # print("Conf = ", confidence.sum().item()/len(confidence))
                if args.mode == 0: ## batch-wised
                    if args.ji_conf == True and cls_loss <= args.b:
                        print("*** Adjusting b(1.5-conf) ***")
                        print("[Before]", cls_loss.item())
                        cls_loss = abs(cls_loss - args.b * (1.5 - confidence.mean())) + args.b * (1.5 - confidence.mean())
                        print("[After]", cls_loss.item())
                    elif args.minus_1_conf == True and cls_loss <= args.b:
                        print("*** Adjusting b(1/conf) ***")
                        print("[Before]", cls_loss.item())
                        cls_loss = abs(cls_loss - args.b * (1 / confidence.mean())) + args.b * (1 / confidence.mean())
                        print("[After]", cls_loss.item())
                    elif args.ji_acc_conf == True and cls_loss <= args.b:
                        print("*** Adjusting b(acc/conf) ***")
                        print("[Before]", cls_loss.item())
                        acc_conf = (torch.from_numpy(correctness).to(torch.device("cuda")) / confidence).mean()
                        cls_loss = abs(cls_loss - args.b * acc_conf) + args.b * acc_conf
                        print("[After]", cls_loss.item())
                        print("--------------------------------------------")
                    elif args.ji_wj != 0 and cls_loss <= args.b:
                        print("*** Adjusting wj")
                        l1loss = nn.L1Loss(reduction="mean").cuda()
                        ji_wj_loss = l1loss(confidence, torch.from_numpy(correctness / epoch).to(torch.device("cuda")))
                    else:
                        if cls_loss.item() <= args.b:
                            print("*** Adjusting b(Flood) ***")
                        cls_loss = abs(cls_loss - args.b) + args.b

                else: ## sample-wised
                    if args.ji_conf == True and cls_loss.mean().item() <= args.b:
                        print("*** Adjusting b(1.5-conf) ***")
                        print("[Before]", cls_loss.mean().item())
                        cls_loss = abs(cls_loss - args.b * (1.5 - confidence)) + args.b * (1.5 - confidence)
                        print("[After]", cls_loss.mean().item())
                        cls_loss = cls_loss.mean()
                    elif args.minus_1_conf == True and cls_loss.mean().item() <= args.b:
                        print("*** Adjusting b(1/conf) ***")
                        print("[Before]", cls_loss.mean().item())
                        cls_loss = abs(cls_loss - args.b * (1 / confidence)) + args.b * (1 / confidence)
                        print("[After]", cls_loss.mean().item())
                        cls_loss = cls_loss.mean()
                    elif args.ji_acc_conf == True and cls_loss.mean().item() <= args.b:
                        print("*** Adjusting b(acc/conf) ***")
                        print("[Before]", cls_loss.mean().item())
                        acc_conf = torch.from_numpy(correctness).to(torch.device("cuda")) / confidence
                        cls_loss = abs(cls_loss - args.b * acc_conf) + args.b * acc_conf
                        cls_loss = cls_loss.mean()
                        print("[After]", cls_loss.mean().item())
                        print("--------------------------------------------")
                    elif args.ji_wj != 0 and cls_loss <= args.b:
                        print("*** Adjusting wj")
                        l1loss = nn.L1Loss(reduction="mean").cuda()
                        ji_wj_loss = l1loss(confidence, torch.from_numpy(correctness / epoch).to(torch.device("cuda")))
                    else:
                        if cls_loss.mean().item() <= args.b:
                            print("*** Adjusting b(Flood) ***")
                        cls_loss = abs(cls_loss - args.b) + args.b
                        cls_loss = cls_loss.mean()



        ranking_loss = args.rank_weight * ranking_loss

        if args.loss == "Margin":
            loss = ranking_loss
        elif args.ji_wj != 0:
            if cls_loss <= args.b:
                loss = cls_loss + args.ji_wj * ji_wj_loss
            else:
                loss = cls_loss
        else:
            loss = cls_loss + ranking_loss
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print("prec", prec)
        # print("correct", correct)
        for a in range(len(idx)):
            if correct[a].item() == False:
                matrix_idx_iscorrect[idx[a]].append(0)
            else:
                matrix_idx_iscorrect[idx[a]].append(1)
        total_losses.update(loss.item(), input.size(0))
        cls_losses.update(cls_loss.mean().item(), input.size(0))
        # cls_losses.update(cls_loss.item(), input.size(0))
        if args.ji_wj != 0 and cls_loss <= args.b:
            ji_wj_losses.update(ji_wj_loss.item(), input.size(0))
        ranking_losses.update(ranking_loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('[{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f}({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}({loss.avg:.4f}) '
                  'CLS Loss {cls_loss.val:.4f}({cls_loss.avg:.4f}) '
                  'Rank Loss {rank_loss.val:.4f}({rank_loss.avg:.4f}) '
                  'JI_WJ Loss {ji_wj_loss.val:.4f}({ji_wj_loss.avg:.4f}) '
                  'Prec {top1.val:.2f}%({top1.avg:.2f}%)'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=total_losses, cls_loss=cls_losses,
                   rank_loss=ranking_losses, ji_wj_loss = ji_wj_losses, top1=top1))



    # max correctness update
    history.max_correctness_update(epoch)
    logger.write([epoch, total_losses.avg, cls_losses.avg, ranking_losses.avg, top1.avg])

    cur_confidence = history.get_confidence()
    cur_correctness = history.get_correctness()

    if args.rank_weight != 0.0:
        return matrix_idx_confidence, matrix_idx_iscorrect, all_idx, all_iscorrect, all_confidence, all_target, cls_losses.avg, ranking_losses.avg, correctness, cur_confidence, cur_correctness

    else:
        return matrix_idx_confidence, matrix_idx_iscorrect, all_idx, all_iscorrect, all_confidence, all_target, total_losses.avg, 0, correctness, cur_confidence, cur_correctness


def valid(loader, model, criterion_cls, criterion_ranking, optimizer, epoch, history, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    cls_losses = utils.AverageMeter()          ## cross entropy loss
    ranking_losses = utils.AverageMeter()      ## marginranking loss
    end = time.time()

    print("*** Valid ***")
    model.eval()

    all_idx = []
    all_iscorrect = []
    all_confidence = []
    all_target = []

    ## 원본 이미지, 라벨 저장

    for i, (input, target, idx) in enumerate(loader):   ## batchsize = 128
    # for i, (input, target) in enumerate(loader):   ## batchsize = 128
        with torch.no_grad():
            data_time.update(time.time() - end)
            input, target = input.cuda(), target.cuda()
            confidence = []
            all_idx.extend(idx.tolist())
            all_target.extend(target.tolist())

            ##mixup
            if args.mixup is not None:
                input, target_a, target_b, lam = utils.mixup_data(input, target, args.mixup, True)
                input, target_a, target_b = map(Variable, (input, target_a, target_b))

            output = model(input)

            # NaN alert
            assert torch.all(output == output)

            # compute ranking target value normalize (0 ~ 1) range
            # max(softmax)
            if args.rank_target == 'softmax':
                conf = F.softmax(output, dim=1)
                confidence, prediction = conf.max(dim=1)        ## predictin : 예측 class, confidence : 그때의 confidence

            # entropy
            elif args.rank_target == 'entropy':
                if args.data == 'cifar100':
                    value_for_normalizing = 4.605170
                else:
                    value_for_normalizing = 2.302585
                confidence = crl_utils.negative_entropy(output,
                                                        normalize=True,
                                                        max_value=value_for_normalizing)
            # margin
            elif args.rank_target == 'margin':
                conf, _ = torch.topk(F.softmax(output), 2, dim=1)
                conf[:,0] = conf[:,0] - conf[:,1]
                confidence = conf[:,0]

            # make input pair
            rank_input1 = confidence
            rank_input2 = torch.roll(confidence, -1)
            idx2 = torch.roll(idx, -1)

            # calc target, margin
            rank_target, rank_margin, norm_cor = history.get_target_margin(idx, idx2) ## rank_target : 누가 더 크냐 1, 0, -1 / rank_margin : 옳게 맞춘 횟수의 차이

            rank_target_nonzero = rank_target.clone()
            rank_target_nonzero[rank_target_nonzero == 0] = 1 ## rank_target 에서 0을 다 1로 바꿈
            rank_input2 = rank_input2 + rank_margin / rank_target_nonzero
            ranking_loss = criterion_ranking(rank_input1,
                                             rank_input2,
                                             rank_target)

            # total loss
            if args.mixup is not None:
                cls_loss = utils.mixup_criterion(criterion_cls, output, target_a, target_b, lam)
            else:
                cls_loss = criterion_cls(output, target)

            ranking_loss = args.rank_weight * ranking_loss
            loss = cls_loss + ranking_loss

        # record loss and accuracy
        prec, correct = utils.accuracy(output, target)

        all_iscorrect.extend(map(int, correct))
        all_confidence.extend(confidence.tolist())
        total_losses.update(loss.item(), input.size(0))
        cls_losses.update(cls_loss.item(), input.size(0))
        ranking_losses.update(ranking_loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('[{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f}({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}({loss.avg:.4f}) '
                  'CLS Loss {cls_loss.val:.4f}({cls_loss.avg:.4f}) '
                  'Rank Loss {rank_loss.val:.4f}({rank_loss.avg:.4f}) '
                  'Prec {top1.val:.2f}%({top1.avg:.2f}%)'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=total_losses, cls_loss=cls_losses,
                   rank_loss=ranking_losses,top1=top1))

        # history.confidence_update(idx, correct, output)


    # max correctness update
    # history.max_correctness_update(epoch)
    logger.write([epoch, total_losses.avg, cls_losses.avg, ranking_losses.avg, top1.avg])

    return all_idx, all_iscorrect, all_confidence, all_target, total_losses, prec.item()