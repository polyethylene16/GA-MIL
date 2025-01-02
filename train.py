from utils import inference_one_epoch, train_one_epoch
from datasets import Mildataset, SubMildataset
from utils import group_argtopk, set_seed, load_pkl
from utils import cal_metric
from models import CustomCLIP
from utils import TransformFixMatch
import torch
import torch.utils.data as data
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms as transforms
import pandas as pd
import os
import math
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import time
from sklearn.model_selection import StratifiedKFold


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def parse_args():
    parser = argparse.ArgumentParser(description="Training for instance-level MIL with multi-modal fine-tuning")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--initial', type=str, default='/root/autodl-tmp/PFM/plip', help='path of pretrained CLIP-wise model')

    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes')
    # parser.add_argument('--patience', type=float, default=0.8, help='patience of early stopping')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--w_decay', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--train_lib', type=str, default='/root/autodl-tmp/PFM/data/processed/train/info_for_training.pkl', help='description and details of train data')
    parser.add_argument('--output', type=str, default="/root/autodl-tmp/PFM/output", help="folder to save weight and log file")
    parser.add_argument('--n_folds', default=5, type=int, help='number of folds')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')

    parser.add_argument('--alpha', type=float, default=0.2, help='coefficient of residual connection')

    parser.add_argument('--k', type=int, default=5, help="top k tiles in slides will be backwarded")
    parser.add_argument('--T', type=float, default=1., help='temperature')
    parser.add_argument('--threshold', type=float, default=0.95, help='pseudo label threshold')
    parser.add_argument('--lambda_u', type=float, default=1, help='coefficient of unlabeled loss')
    return parser.parse_args()


def main(args):
    n_epochs = args.n_epochs
    eps = 1e-4
    classnames = ['This is a histopathological photograph with normal cells.',  
                  'This image shows the occurrence of metastatic carcinoma in anterior breast lymph nodes.']

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    lib = pd.DataFrame(load_pkl(args.train_lib))
    X = lib["slides"].to_list()
    y = lib["labels"].to_list()

    transform_labeled = transforms.Compose([transforms.Resize(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomCrop(size=224,
                                                                 padding=int(224 * 0.125),
                                                                 padding_mode='reflect'),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    transform_val = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                           std=[0.26862954, 0.26130258, 0.27577711])])
    transform_unlabeled = TransformFixMatch(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])


    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"----------------* start training: fold {fold + 1} *----------------")

        save_fold = os.path.join(args.output, 'fold_{}'.format(fold + 1))
        
        if not os.path.exists(save_fold):
            RuntimeError("output folder does not exist! Please run MO Model first!")
        
        else:
            ckpt = torch.load(os.path.join(save_fold, "best_model.pt"), map_location=device)
        
        args.fold = fold

        train_lib = lib.iloc[list(train_idx)]
        val_lib = lib.iloc[list(val_idx)]
        train_dset = Mildataset(train_lib, transform=transform_val)
        val_dset = Mildataset(val_lib, transform=transform_val)

        train_loader = data.DataLoader(train_dset, 
                                       batch_size=args.batch_size, 
                                       shuffle=False, 
                                       num_workers=args.n_workers, 
                                       pin_memory=False)
        
        val_loader = data.DataLoader(val_dset,
                                     batch_size=args.batch_size, 
                                     shuffle=False, 
                                     num_workers=args.n_workers, 
                                     pin_memory=False)
        if args.initial:
            clip_model = CLIPModel.from_pretrained(args.initial)
            processor = CLIPProcessor.from_pretrained(args.initial)

        else:
            clip_model = CLIPModel.from_pretrained("vinid/plip")
            processor = CLIPProcessor.from_pretrained("vinid/plip")

        clip_model.float()
        clip_model.to(device)

        model = CustomCLIP(classnames=classnames,
                           processor=processor,
                           clip_model=clip_model,
                           ratio=4,
                           alpha=args.alpha, 
                           learnable=False)
        
        model.to(device)
        model.load_state_dict(ckpt['model'])
        
        for name, param in model.named_parameters():
            if 'adapter' not in name and 'alpha' not in name:
                param.requires_grad_(False)
        
        # pg = [p for p in model.parameters() if p.requires_grad]
        # optimizer = torch.optim.AdamW(pg, lr = args.lr, weight_decay=args.w_decay)
        no_decay = ["bias", 'bn']
        grouped_parameters = [
            {
                "params":[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": args.w_decay
            },
            {
                "params":[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0
            }
        ]
        optimizer = torch.optim.AdamW(grouped_parameters, lr = args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
 
        best_score = 0.0
        # current_patience = 0

        log = os.path.join(save_fold, 'log.txt')
        with open(log, 'a') as f:
            f.write('\n')
            f.write('M1 Stage Starting ...\n')


        for epoch in range(n_epochs):
            args.epoch = epoch
            time_start = time.time()
            train_dset.set_mode(1, transform=transform_val)
            probs = inference_one_epoch(model=model,
                                        loader=train_loader,
                                        num_classes=args.n_classes,
                                        args=args)
            
            w = np.arange(args.n_classes) / np.arange(args.n_classes).sum()
            scores = probs @ w.T
            topk = group_argtopk(np.array(train_dset.slide_idx), scores, args.k)
            labeled_ids = topk[args.k - 1::args.k]
            unlabeled_ids = [i for i in topk if i not in labeled_ids]
            train_labeled_dset = SubMildataset(train_lib, indexs=labeled_ids, transform=transform_labeled)
            train_unlabeled_dset = SubMildataset(train_lib, indexs=unlabeled_ids, transform=transform_unlabeled)
            train_labeled_loader = data.DataLoader(train_labeled_dset,
                                           batch_size=args.batch_size, 
                                           num_workers=args.n_workers,
                                           shuffle=True,
                                           drop_last=True)
            train_unlabeled_loader = data.DataLoader(train_unlabeled_dset,
                                           batch_size=args.batch_size * (args.k - 1), 
                                           num_workers=args.n_workers,
                                           shuffle=True,
                                           drop_last=True)
            
            # scheduler.step()
            
            val_dset.set_mode(1, transform=transform)
            probs, uncertainty = inference_one_epoch_joint(model=model,
                                        loader=val_loader,
                                        num_classes=args.n_classes,
                                        args=args,
                                        return_uncertainty=True)
            
            logit = probs[group_argtopk(np.array(val_dset.slide_idx), probs @ w.T, 1)]
            pred = np.argmax(logit, axis=-1)
            gt = np.array(val_dset.labels)

            acc, auc, f1, kappa, co_matrix = cal_metric(logit, pred, gt, args)
            harm_mean = 4 / (1 / (acc - eps) + 1 / (auc - eps) + 1 / (f1 + eps) + 1 / (kappa + eps))

            performance = {'Kappa': '{:.6f}'.format(kappa), 
                           'F1': '{:.6f}'.format(f1),
                           'AUC': '{:.6f}'.format(auc),
                           'Acc': '{:.6f}'.format(acc),
                           'Uncertainty': '{:.6f}'.format(uncertainty.sum()),
                           f'fold': f'{fold+1}'}
            
            with open(log, 'a') as file:
                file.write(str(performance) + '\n')

            if harm_mean >= best_score:
                best_score = harm_mean
                state_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "fold": fold + 1
                }
                print("best model saving ...\t")
                torch.save(state_dict, f"{save_fold}/best_model_M1.pt")
                # current_patience = 0
                best_perform = performance
            
            # else:
            #     current_patience += 1
            #     if current_patience >= args.patience:
            #         print("Early stopping triggered, stopping training ...")
            #         with open(log, 'a') as file:
            #             file.write('Early stopping triggered. Stopping training.\n')
            #         break
        print(f'"----------------* finish training: fold {fold + 1} *----------------')
        with open(log, "a") as file:
            file.write('Training finished.\n') 
            file.write('Best performance:\t')
            file.write(str(best_perform))

    
    time_end = time.time()
    print(f'Full time: {time_end - time_start} seconds.')

if __name__ == '__main__':
    args = parse_args()
    main(args)
