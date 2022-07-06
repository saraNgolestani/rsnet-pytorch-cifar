'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as ptl
from utils.score_utils import Statistics, compute_scores, compute_scores_and_th


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(ptl.LightningModule):
    def __init__(self, block, num_blocks, num_classes=80, lr=5e-4):
        super(ResNet, self).__init__()
        self.best_th = 0.45
        self.val_step_counter = 0
        self.all_val_pred = []
        self.all_val_actual = []
        self.all_train_pred = []
        self.all_train_actual = []
        self.val_stats = Statistics()
        self.train_stats = Statistics()
        self.test_stats = Statistics()
        self.lr = lr
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def bcewithlogits_loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=14000,
            gamma=0.9
        ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1}
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.bcewithlogits_loss(logits, y.float())
        step_preds = logits.detach().cpu()
        step_actuals = y.cpu()
        #preds = (logits.detach() >= 0.45)
        current_loss = loss.item() * x.size(0)
        scores, _ = compute_scores_and_th(step_preds, step_actuals, self.best_th)
        self.all_train_pred.extend(step_preds.tolist())
        self.all_train_actual.extend(step_actuals.tolist())
        #scores = compute_scores(preds.cpu(), y.cpu())
        self.train_stats.update(loss=float(current_loss), precision=scores)
        self.log('train mAP', 100 * self.train_stats.precision())
        self.log('train loss', self.train_stats.loss())
        return loss

    def training_epoch_end(self, outputs):
        self.log('train mAP on epoch', 100 * self.train_stats.precision())
        self.log('train loss on epoch', self.train_stats.loss())
        scores, self.best_th = compute_scores_and_th(self.all_train_pred, self.all_train_actual)
        self.log('train mAP on epoch with best TH', 100 * (sum(scores) / len(scores)))

        self.all_train_pred = []
        self.all_train_actual = []
        self.val_step_counter = 0
        self.train_stats = Statistics()

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.bcewithlogits_loss(logits, y.float())
        #preds = (logits.detach() >= 0.45)
        current_loss = loss.item() * x.size(0)
        step_preds = logits.detach().cpu()
        step_actuals = y.cpu()
        scores, _ = compute_scores_and_th(step_preds, step_actuals, self.best_th)

        self.all_val_pred.extend(step_preds.tolist())
        self.all_val_actual.extend(step_actuals.tolist())
        self.val_stats.update(loss=float(current_loss), precision=scores, best_th=self.best_th)
        return loss

    def validation_epoch_end(self, outputs):
        print(f'validation epoch end on device:{self.device}')

        self.log('val mAP on epoch', 100 * self.val_stats.precision())
        self.log('val loss on epoch', self.val_stats.loss())
        self.log('val best TH on epoch', self.best_th)

        if self.all_val_pred and self.all_val_actual:
            scores, self.best_th = compute_scores_and_th(self.all_val_pred, self.all_val_actual)
            if scores is not None:
                self.log('val mAP on epoch with best TH', 100 * (sum(scores) / len(scores)))
        self.all_val_pred = []
        self.all_val_actual = []
        self.val_step_counter = 0
        self.val_stats = Statistics()


def ResNet18(args):
    return ResNet(BasicBlock, [2, 2, 2, 2], lr=args.lr)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
