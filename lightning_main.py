
from models.resnet import ResNet18
import argparse
from utils.dataset_utils import COCODatasetLightning
from utils.uav_utils import UAVDatasetLightning
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch
import os
import warnings
import numpy as np
import random
from  pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb
warnings.filterwarnings('always')

# torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch TResNet ImageNet Inference')
parser.add_argument('--val_dir')
parser.add_argument('--checkpoint_name')
parser.add_argument('--save_path', default='saved_models')
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--max_epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--num_nodes', type=int, default=1)
parser.add_argument('--dataset_sampling_ratio', default=1, type=float, help="sampling ratio of dataset")
parser.add_argument('--seed', default=0, type=int, help="seed for randomness")
parser.add_argument('--wandb_name', default='resnet_uav_mgpu')
parser.add_argument('--load_from_chkp', default=False, type=bool, help="load from check point")
parser.add_argument('--train', default=True, type=bool, help="load from check point")
parser.add_argument('--num_devices', type=int, default=2)



args = parser.parse_args()
checkpoint_callback = ModelCheckpoint(
    monitor='val mAP on epoch with best TH',
    dirpath=args.save_path,
    filename='model-{epoch:03d}-{val mAP on UAV with best TH:.2f}',
    save_top_k=2,
    mode='max'
)


def set_seed(seed=0):
    seed_everything(seed)
    np.random.seed(seed)
    random.seed(seed)


def run():
    torch.multiprocessing.freeze_support()
    print('loop')


if __name__ == '__main__':
    args = parser.parse_args()
    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger(project='resnet-18', name=args.wandb_name, entity="sara_ngln")

    wandb_logger.experiment.config.update({
        "val_zoom_factor": args.val_zoom_factor,
        "batch_size": args.batch_size,
        "num_gpus": args.num_gpu,
        "num_nodes": args.max_epochs,
        "dataset_sampling_ratio": args.dataset_sampling_ratio,
    })

    set_seed(args.seed)
    run()
    model = ResNet18(args)
    trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], max_epochs=args.num_epochs,
                         num_nodes=args.num_nodes, gpus=args.num_gpu, accelerator="gpu", devices=args.num_devices, precision=32,
                         strategy='ddp')
    train_dl = COCODatasetLightning(args).train_dataloader()
    val_dl = COCODatasetLightning(args).val_dataloader()
    # train_dl = UAVDatasetLightning(args).train_dataloader()
    # val_dl = UAVDatasetLightning(args).val_dataloader()
    if args.load_from_chkp and args.train:
        trainer.fit(model, train_dl, val_dl, ckpt_path=os.path.join(args.save_path, args.checkpoint_name))
    elif args.train:
        trainer.fit(model, train_dl, val_dl)

    if not args.train:
        data_module = UAVDatasetLightning(args)
        trainer.test(model=model, datamodule=data_module, ckpt_path=os.path.join(args.save_path, args.checkpoint_name))






