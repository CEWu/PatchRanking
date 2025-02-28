from dassl.data import DatasetWrapper
from datasets.eurosat import EuroSAT, OxfordPets
import os, argparse
from dassl.config import get_cfg_default
from dassl.data.transforms import build_transform
import torch 
import timm
from clip import clip
import json
from dassl.data import DataManager
import torch.nn as nn
import pdb
from dassl.optim import build_optimizer, build_lr_scheduler
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import scipy.stats as stats
from timm.models.mlp_mixer import MixerBlock

import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from dassl.utils import setup_logger, set_random_seed, collect_env_info

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.COOP.PRUNING_LOC = [] 
    cfg.TRAINER.COOP.NUM_TOKEN= []
    cfg.TRAINER.COOP.EVAL_RANK = False
    cfg.TRAINER.COOP.EVAL_PRUNE = False
    cfg.TRAINER.COOP.EVAL_MASK = False
    cfg.TRAINER.COOP.TRAIN_RANK = False
    cfg.TRAINER.COOP.TOKENS_SCORE = None
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 64

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for only vision side prompting
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    cfg.freeze()

    return cfg

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="", help="path to dataset")
parser.add_argument("--output-dir", type=str, default="", help="output directory")
parser.add_argument("--config-file", type=str, default="", help="path to config file")
parser.add_argument(
    "--dataset-config-file",
    type=str,
    default="",
    help="path to config file for dataset setup",
)
parser.add_argument("--num-shot", type=int, default=1, help="number of shots")
parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="which split")
parser.add_argument("--trainer", type=str, default="", help="name of trainer")
parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
parser.add_argument("--head", type=str, default="", help="name of head")
parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
parser.add_argument("--eval-only", action="store_true", help="evaluation only")
args = parser.parse_args()
cfg = setup_cfg(args)

if args.seed >= 0:
    print("Setting fixed seed: {}".format(args.seed))
    set_random_seed(args.seed)

print(cfg)
# dataset = eval(cfg.DATASET.NAME)(cfg)
# dataset_input = dataset.train_x

train_preprocess = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]
)

test_preprocess = transforms.Compose([
    transforms.Resize(size=224, interpolation=INTERPOLATION_MODES['bicubic'], max_size=None, antialias=None),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]
)
        
dm = DataManager(cfg, custom_tfm_train=test_preprocess, custom_tfm_test=test_preprocess)
train_loader_x = dm.train_loader_x

test_loader = dm.test_loader
# tfm_train = build_transform(cfg, is_train=False)
# data_loader = torch.utils.data.DataLoader(
#     DatasetWrapper(cfg, dataset_input, transform=tfm_train, is_train=False),
#     batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
#     sampler=None,
#     shuffle=False,
#     num_workers=cfg.DATALOADER.NUM_WORKERS,
#     drop_last=False,
#     pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
# )

import random

def compute_margin_ranking_loss(features, patch_index, sample_size=1000):
    ranking_loss = nn.MarginRankingLoss(margin=1)
    
    # Generate sampled pairwise comparisons
    sampled_pairs = random.sample([(i, j) for i in range(len(patch_index)) for j in range(i+1, len(patch_index))], sample_size)
    
    # Extract features using advanced indexing
    i_indices, j_indices = zip(*sampled_pairs)
    patch_a_features = features[0, [patch_index[i] for i in i_indices]]
    patch_b_features = features[0, [patch_index[j] for j in j_indices]]
    
    # Determine ranking
    y = torch.tensor([1 if i < j else -1 for i, j in sampled_pairs], dtype=torch.float32, device=features.device)
    
    # Compute the loss
    losses = ranking_loss(patch_a_features, patch_b_features, y)
    average_loss = torch.mean(losses)
    
    return average_loss

def compute_batch_margin_ranking_loss(features, patch_indices, sample_size=1000):
    batch_size = features.size(0)
    
    # Using a list comprehension here
    batch_losses = [compute_margin_ranking_loss(features[b].unsqueeze(0), patch_indices[b], sample_size) for b in range(batch_size)]
    
    # Convert batch_losses to a tensor and then compute the mean
    average_batch_loss = torch.stack(batch_losses).mean()
    
    return average_batch_loss

def ranking_to_prob_distribution(ranking):
    """
    Convert a ranking (list of indices) to a probability distribution.
    
    Args:
    - ranking (list): List of indices arranged in descending order of importance.
    
    Returns:
    - torch.Tensor: Probability distribution.
    """
    # Convert ranking to inverse scores such that higher-ranked items get higher scores
    scores = [0 for _ in ranking]
    scores = torch.tensor(scores, dtype=torch.float32)
    for index, rank in enumerate(ranking):
        if index >= len(ranking)-50:
            scores[rank] =  -5 - index * 0.1
        else:
            scores[rank] =  0
        #scores[rank] = 1/(index + 1)
    #sscores *= 100
    #scores = torch.tensor([1 / (rank + 1) for rank in range(len(ranking))], dtype=torch.float32)
    
    # Normalize the scores to get a probability distribution
    prob_distribution = F.softmax(scores, dim=-1)
    
    return prob_distribution

def compute_kl_divergence_loss(predicted_scores, true_rankings):
    """
    Compute the KL-divergence loss given predicted scores and true rankings.
    
    Args:
    - predicted_scores (torch.Tensor): Scores predicted by the model.
    - true_rankings (list of lists): List of rankings, one for each instance in the batch.
    
    Returns:
    - torch.Tensor: Average KL-divergence loss over the batch.
    """
    batch_size = len(true_rankings)
    kl_losses = []

    for i in range(batch_size):
        true_probs = ranking_to_prob_distribution(true_rankings[i]).to(predicted_scores.device)
        predicted_probs = F.softmax(predicted_scores[i], dim=0)
        
        # Align predicted probabilities with the ground truth ranking order for the current instance
        #aligned_predicted_probs = torch.index_select(predicted_probs, dim=0, index=torch.tensor(true_rankings[i], dtype=torch.long, device=predicted_scores.device))
        
        kl_loss = F.kl_div(predicted_probs.log(), true_probs, reduction='sum')
        kl_losses.append(kl_loss)
    
    # Average the KL-divergence losses over the batch
    avg_loss = sum(kl_losses) / batch_size
    return avg_loss
import numpy as np
def top_n_match(A, B, N=-50):
    A_top = set(A[N:].tolist())
    B_top = set(B[N:])
    intersection = A_top.intersection(B_top)
    return (len(intersection) / 50) * 100

'''
def compute_margin_ranking_loss(features, patch_index, sample_size=1000):
    """
    Compute the Margin Ranking Loss for given features and patch_index using sampling.
    
    Args:
    - features (torch.Tensor): Feature tensor of shape (batch_size, 196).
    - patch_index (list): List of patch indices indicating their importance.
    - sample_size (int): Number of pairs to sample.
    
    Returns:
    - float: Average Margin Ranking Loss over the sampled pairs.
    """
    ranking_loss = nn.MarginRankingLoss(margin=1)
    loss_values = []
    
    # Generate sampled pairwise comparisons
    sampled_pairs = random.sample([(i, j) for i in range(len(patch_index)) for j in range(i+1, len(patch_index))], sample_size)
    
    for (i, j) in sampled_pairs:
        patch_a_idx = patch_index[i]
        patch_b_idx = patch_index[j]
        
        # Extract features for the patches
        patch_a_feature = features[0, patch_a_idx].unsqueeze(0)
        patch_b_feature = features[0, patch_b_idx].unsqueeze(0)
        
        # If patch A is ranked higher than patch B, y = 1, else y = -1
        y = 1 if i < j else -1
        y_tensor = torch.tensor([y], dtype=torch.float32, device=features.device)
        # Compute the loss for the pair
        loss = ranking_loss(patch_a_feature, patch_b_feature, y_tensor)
        loss_values.append(loss)
    
    # Average the loss values for the instance
    average_loss = sum(loss_values) / len(loss_values)
    return average_loss


def compute_batch_margin_ranking_loss(features, patch_index, sample_size=1000):
    """
    Compute the Margin Ranking Loss for the entire batch given features and patch_index using sampling.
    
    Args:
    - features (torch.Tensor): Feature tensor of shape (batch_size, 196).
    - patch_index (list): List of patch indices indicating their importance.
    - sample_size (int): Number of pairs to sample.
    
    Returns:
    - float: Average Margin Ranking Loss over the batch.
    """
    batch_losses = []
    for b in range(features.size(0)):
        # normalize the features
        batch_loss = compute_margin_ranking_loss(features[b].unsqueeze(0), patch_index[b], sample_size)
        batch_losses.append(batch_loss)
    
    # Average the loss values for the entire batch
    average_batch_loss = sum(batch_losses) / len(batch_losses)
    return average_batch_loss
'''
'''
batch_size=2
# Create a dummy feature tensor of shape (batch_size, 196)
features = torch.rand((batch_size, 196))
patch_index = torch.tensor([139, 136, 137, 144, 147, 149, 154, 146, 143, 145, 150, 153, 138, 141, 155, 151, 170, 142, 157, 159, 176, 179, 148, 156, 173, 177, 184, 164, 152, 168, 171, 180, 163, 183, 172, 175, 161, 178, 181, 39, 98, 158, 27, 185, 35, 78, 165, 182, 162, 33, 37, 13, 26, 8, 73, 120, 34, 61, 65, 77, 86, 127, 134, 0, 7, 12, 24, 59, 75, 83, 95, 129, 1, 18, 30, 51, 66, 111, 115, 126, 4, 17, 90, 102, 114, 6, 21, 22, 23, 32, 46, 94, 99, 2, 20, 79, 122, 135, 3, 10, 133, 105, 112, 43, 72, 93, 60, 125, 11, 80, 92, 116, 124, 31, 106, 100, 128, 131, 29, 121, 132, 68, 41, 55, 19, 40, 186, 187, 189, 190, 194, 193, 188, 191, 192, 195, 44, 82, 88, 36, 38, 42, 49, 96, 108, 130, 70, 85, 47, 53, 118, 9, 84, 91, 119, 140, 74, 167, 174, 48, 169, 50, 58, 103, 166, 97, 54, 52, 64, 15, 81, 101, 113, 45, 71, 107, 109, 67, 87, 123, 16, 57, 56, 69, 110, 117, 76, 25, 104, 5, 62, 160, 14, 28, 63, 89]).long()
# Compute the Margin Ranking Loss for the dummy features and patch_index list using sampling
loss_value = compute_batch_margin_ranking_loss(features, patch_index)
loss_value
'''
file_name = "test/oxford_pets_tokens_rank_10token_6iter_avg_train.json"
with open(file_name, "r") as file:
    tokens_rank_dict = json.load(file)
file_name = "test/oxford_pets_tokens_rank_10token_6iter_avg_test.json"
with open(file_name, "r") as file:
    test_tokens_rank_dict = json.load(file)
EPOCH = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

feature_extractor = timm.create_model("timm/vit_base_patch16_clip_224", pretrained=True, num_classes=0).to(device)
print(feature_extractor)

class Predictor(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_embed_dim=768, out_embed_dim=196):
        super().__init__()
        '''
        self.in_conv = nn.Sequential(
            nn.AdaptiveAvgPool1d(in_embed_dim),
            nn.Linear(in_embed_dim, out_embed_dim),
            nn.LayerNorm(out_embed_dim),
            nn.GELU(),
            nn.Linear(out_embed_dim, out_embed_dim),
            nn.LayerNorm(out_embed_dim),
            nn.GELU(),
            nn.Linear(out_embed_dim, 1),
        )
        '''
        self.adaptive_pool = nn.AdaptiveAvgPool1d(in_embed_dim)
        #encoder_layer = nn.TransformerEncoderLayer(d_model=in_embed_dim, nhead=8)
        #self.in_conv = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.in_conv = nn.Sequential(
            MixerBlock(in_embed_dim, 197)
        )
        self.fc = nn.Linear(in_embed_dim, 1)
        #self.fc = nn.Linear(768, 196)
        #self.cls_project = nn.Linear(768, self.in_embed_dim)

    def forward(self, x):
        # x: [B, 196, 768]
        x = self.adaptive_pool(x)
        x = self.in_conv(x)
        #x = self.fc(x)
        x = x[:, 1:,:] # [B, 768]
        x = self.fc(x) # [B, 196, 1]
        return x

predictor = Predictor(in_embed_dim=64, out_embed_dim=256).to(device)
#optimizer = optim.SGD(predictor.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-4)
# Directory to save checkpoints
checkpoint_dir = './checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
feature_extractor.eval()
for epoch in range(EPOCH):
    predictor.train()
    for batch_idx, batch in enumerate(train_loader_x):
    # for index, batch in enumerate(data_loader):
        im_patch_index_list = []
        image, label, impath = batch["img"], batch["label"], batch['impath']
        image = image.cuda()
        label = label.cuda()
        with torch.no_grad():
            feature = feature_extractor._intermediate_layers(image, n=10)
        feature_block3 = feature[-1]
        #feature_block3 = (feature[0] + feature[1] + feature[2])/3.0
        patch_feature = feature_block3[:,:,:]
        pred_patch_feature = predictor(patch_feature).squeeze(dim=-1)#.softmax(dim=-1)
        
        for imp in impath:         
            im_patch_index = tokens_rank_dict[imp]['patch_index']
            im_patch_index_list.append(im_patch_index)
        #print(len(im_patch_index_list))
        #loss = compute_batch_margin_ranking_loss(pred_patch_feature, im_patch_index_list, sample_size=10000)
        loss = compute_kl_divergence_loss(pred_patch_feature, im_patch_index_list)
        distance = 0
        for index in range(len(im_patch_index_list)):
            scores = pred_patch_feature[index].detach().cpu().numpy()
            ranking_of_scores = scores.argsort()[::-1]
            if torch.rand(1) < 0.0:
                # show the ranking 
                print(ranking_of_scores[:15])
                print(im_patch_index_list[index][:15])
            distance += top_n_match(ranking_of_scores, im_patch_index_list[index], -50)
            #distance += stats.kendalltau(ranking_of_scores, im_patch_index_list[index])[0]
            #distance += wasserstein_distance(, im_patch_index_list[index])
        distance = distance/len(im_patch_index_list)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}, Distance: {distance}")
    lr_scheduler.step()

    # evaluate
    predictor.eval()
    avg_distance = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            im_patch_index_list = []
            image, label, impath = batch["img"], batch["label"], batch['impath']
            image = image.cuda()
            
            feature = feature_extractor._intermediate_layers(image, n=10)
            feature_block3 = feature[-1]
            patch_feature = feature_block3[:,:,:]
            pred_patch_feature = predictor(patch_feature).squeeze(dim=-1)
            
            for imp in impath:         
                im_patch_index = test_tokens_rank_dict[imp]['patch_index']
                im_patch_index_list.append(im_patch_index)
                
            distance = 0
            for index in range(len(im_patch_index_list)):
                scores = pred_patch_feature[index].detach().cpu().numpy()
                ranking_of_scores = scores.argsort()[::-1]
                distance += top_n_match(ranking_of_scores, im_patch_index_list[index], -50)
                if torch.rand(1) < 0.0:
                    # show the ranking 
                    print(ranking_of_scores[:15])
                    print(im_patch_index_list[index][:15])
            distance = distance/len(im_patch_index_list)
            avg_distance += distance
        avg_distance = avg_distance/len(test_loader)
        print(f"--------------  Evaluation: Epoch: {epoch}, Batch: {batch_idx}, Distance: {avg_distance}  --------------")
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.ckpt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': predictor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
        