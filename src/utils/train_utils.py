import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    
    return res

def train(model, train_loader, optimizer, criterion, total_epoch, epoch):
    losses = []
    # ensure model is in training mode
    model.train()    
    
    iterator = tqdm(train_loader, desc="Training epoch {}/{}".format(epoch+1,total_epoch), leave=True)
    for (inputs, target) in iterator:   

        inputs = inputs.cuda()
        target = target.cuda()

        # forward pass
        output = model(inputs)
        loss = criterion(output, target)

        # backward pass + run optimizer to update weights
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        # keep track of loss value
        losses.append(loss.cpu().detach().numpy())

        # set postfix
        pred = output.argmax(dim=1, keepdim=True)
        y_true=target.tolist() 
        y_pred=pred.reshape(-1).tolist()
        accuracy = accuracy_score(y_true, y_pred)
        iterator.set_postfix(loss="{:.2f}".format(losses[-1]),acc="{:.2f}".format(accuracy))
           
    return losses

def test(model, test_loader):    
    # ensure model is in eval mode
    model.eval() 
    y_true = []
    y_pred = []
   
    with torch.no_grad():
        for (inputs, target) in tqdm(test_loader, desc="Test...", leave=False):
            inputs = inputs.cuda()
            target = target.cuda()
            output = model(inputs)
            pred = output.argmax(dim=1, keepdim=True)
            y_true.extend(target.tolist()) 
            y_pred.extend(pred.reshape(-1).tolist())
        
    return accuracy_score(y_true, y_pred)

def display_loss_plot(losses, title="Training loss", xlabel="Iterations", ylabel="Loss"):
    x_axis = [i for i in range(len(losses))]
    plt.plot(x_axis,losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def kd_loss_function(output, target_output, temperature):
    output = output / temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    
    return loss_kd

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    
    return torch.abs(loss).sum()

def sdn_train(model,train_loader,optimizer,criterion,temperature,total_epoch=None, epoch=None):
    model.train()
    alpha = 0.1
    beta = 1e-6
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    middle1_losses = AverageMeter()
    middle2_losses = AverageMeter()
    middle3_losses = AverageMeter()
    losses1_kd = AverageMeter()
    losses2_kd = AverageMeter()
    losses3_kd = AverageMeter()
    feature_losses_1 = AverageMeter()
    feature_losses_2 = AverageMeter()
    feature_losses_3 = AverageMeter()
    total_losses = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()
    desc = "Training "
    if total_epoch is not None and epoch is not None:
        desc+="epoch [%d/%d]"%(epoch,total_epoch)
    iterator = tqdm(train_loader,desc=desc,leave=True)
    end_time = time.time()
    for i, (input,target) in enumerate(iterator):
        data_time.update(time.time() - end_time)
        
        target = target.squeeze().long().cuda()
        input = input.cuda()
        
        output, middle_output1, middle_output2, middle_output3, \
        final_fea, middle1_fea, middle2_fea, middle3_fea = model(input)

        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        middle1_loss = criterion(middle_output1, target)
        middle1_losses.update(middle1_loss.item(), input.size(0))
        middle2_loss = criterion(middle_output2, target)
        middle2_losses.update(middle2_loss.item(), input.size(0))
        middle3_loss = criterion(middle_output3, target)
        middle3_losses.update(middle3_loss.item(), input.size(0))

        temp4 = output / temperature
        temp4 = torch.softmax(temp4, dim=1)
        
        loss1by4 = kd_loss_function(middle_output1, temp4.detach(), temperature) * (temperature**2)
        losses1_kd.update(loss1by4, input.size(0))

        loss2by4 = kd_loss_function(middle_output2, temp4.detach(), temperature) * (temperature**2)
        losses2_kd.update(loss2by4, input.size(0))

        loss3by4 = kd_loss_function(middle_output3, temp4.detach(), temperature) * (temperature**2)
        losses3_kd.update(loss3by4, input.size(0))

        feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach()) 
        feature_losses_1.update(feature_loss_1, input.size(0))
        feature_loss_2 = feature_loss_function(middle2_fea, final_fea.detach()) 
        feature_losses_2.update(feature_loss_2, input.size(0))
        feature_loss_3 = feature_loss_function(middle3_fea, final_fea.detach()) 
        feature_losses_3.update(feature_loss_3, input.size(0))

        total_loss = (1 - alpha) * (loss + middle1_loss + middle2_loss + middle3_loss) + \
                    alpha * (loss1by4 + loss2by4 + loss3by4) + \
                    beta * (feature_loss_1 + feature_loss_2 + feature_loss_3)
        total_losses.update(total_loss.item(), input.size(0))

        prec1 = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], input.size(0))
        
        middle1_prec1 = accuracy(middle_output1.data, target, topk=(1,))
        middle1_top1.update(middle1_prec1[0], input.size(0))
        middle2_prec1 = accuracy(middle_output2.data, target, topk=(1,))
        middle2_top1.update(middle2_prec1[0], input.size(0))
        middle3_prec1 = accuracy(middle_output3.data, target, topk=(1,))
        middle3_top1.update(middle3_prec1[0], input.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        show_info = "Time {batch_time.val:.3f} ({batch_time.avg:.3f}), "\
                    "Data {data_time.val:.3f} ({data_time.avg:.3f}), "\
                    "Loss {loss.val:.3f} ({loss.avg:.3f}), "\
                    "Prec@1 {top1.val:.2f}% ({top1.avg:.2f}%), ".format(
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=total_losses,
                    top1=top1)
        iterator.set_postfix({"Msg":show_info})
    return batch_time.avg, data_time.avg, middle1_losses.avg,middle2_losses.avg,middle3_losses.avg,losses.avg

def sdn_validate(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()
    end = time.time()
    for input, target in tqdm(test_loader):

        target = target.squeeze().long().cuda()
        input = input.cuda()

        output, middle_output1, middle_output2, middle_output3, \
        final_fea, middle1_fea, middle2_fea, middle3_fea = model(input)
            
        prec1 = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], input.size(0))
        middle1_prec1 = accuracy(middle_output1.data, target, topk=(1,))
        middle1_top1.update(middle1_prec1[0], input.size(0))
        middle2_prec1 = accuracy(middle_output2.data, target, topk=(1,))
        middle2_top1.update(middle2_prec1[0], input.size(0))
        middle3_prec1 = accuracy(middle_output3.data, target, topk=(1,))
        middle3_top1.update(middle3_prec1[0], input.size(0))
        
    return middle1_top1.avg, middle2_top1.avg, middle3_top1.avg, top1.avg

def dsn_train(model,train_loader,optimizer,criterion,total_epoch=None, epoch=None):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    middle1_losses = AverageMeter()
    middle2_losses = AverageMeter()
    middle3_losses = AverageMeter()
    total_losses = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()
    desc = "Training "
    if total_epoch is not None and epoch is not None:
        desc+="epoch [%d/%d]"%(epoch,total_epoch)
    iterator = tqdm(train_loader,desc=desc,leave=True)
    end_time = time.time()
    for i, (input,target) in enumerate(iterator):
        data_time.update(time.time() - end_time)
        
        target = target.squeeze().long().cuda()
        input = input.cuda()
        
        out1,out2,out3,out4 = model(input)

        loss = criterion(out4, target)
        losses.update(loss.item(), input.size(0))

        middle1_loss = criterion(out1, target)
        middle1_losses.update(middle1_loss.item(), input.size(0))
        middle2_loss = criterion(out2, target)
        middle2_losses.update(middle2_loss.item(), input.size(0))
        middle3_loss = criterion(out3, target)
        middle3_losses.update(middle3_loss.item(), input.size(0))


        total_loss = loss + middle1_loss + middle2_loss + middle3_loss
        total_losses.update(total_loss.item(), input.size(0))

        prec1 = accuracy(out4.data, target, topk=(1,))
        top1.update(prec1[0], input.size(0))
        
        middle1_prec1 = accuracy(out1.data, target, topk=(1,))
        middle1_top1.update(middle1_prec1[0], input.size(0))
        middle2_prec1 = accuracy(out2.data, target, topk=(1,))
        middle2_top1.update(middle2_prec1[0], input.size(0))
        middle3_prec1 = accuracy(out3.data, target, topk=(1,))
        middle3_top1.update(middle3_prec1[0], input.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        show_info = "Time {batch_time.avg:.3f}s, "\
                    "Data {data_time.avg:.3f}s, "\
                    "Loss ({loss1.avg:.3f}|{loss2.avg:.3f}|{loss3.avg:.3f}|{loss4.avg:.3f}), "\
                    "Top1 ({acc1.avg:.2f}%|{acc2.avg:.2f}%|{acc3.avg:.2f}%|{acc4.avg:.2f}%|)".format(
                    batch_time=batch_time, data_time=data_time,
                    loss1=middle1_losses, loss2=middle2_losses, loss3=middle3_losses, loss4=losses,
                    acc1=middle1_top1,acc2=middle2_top1,acc3=middle3_top1,acc4=top1)
        iterator.set_postfix({"Msg":show_info})
    return batch_time.avg, data_time.avg, middle1_losses.avg,middle2_losses.avg,middle3_losses.avg,losses.avg

def dsn_validate(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()
    end = time.time()
    for input, target in tqdm(test_loader):

        target = target.squeeze().long().cuda()
        input = input.cuda()

        out1,out2,out3,out4 = model(input)
            
        prec1 = accuracy(out4.data, target, topk=(1,))
        top1.update(prec1[0], input.size(0))
        middle1_prec1 = accuracy(out1.data, target, topk=(1,))
        middle1_top1.update(middle1_prec1[0], input.size(0))
        middle2_prec1 = accuracy(out2.data, target, topk=(1,))
        middle2_top1.update(middle2_prec1[0], input.size(0))
        middle3_prec1 = accuracy(out3.data, target, topk=(1,))
        middle3_top1.update(middle3_prec1[0], input.size(0))
        
    return middle1_top1.avg, middle2_top1.avg, middle3_top1.avg, top1.avg