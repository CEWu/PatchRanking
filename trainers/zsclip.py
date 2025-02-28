import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from tqdm import tqdm
import json
import pdb
import re
import torch.nn.functional as F
from dassl.data import DataManager
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.cuda.amp import GradScaler, autocast
from dassl.metrics import compute_accuracy
is_debug=False

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


def increment_iter_in_filename(filename):
    # Define a regular expression pattern to match the '_<number>iter' part
    pattern = r'_(\d+)iter\.json$'
    
    # Define a function that will be used to increment the matched integer
    def repl(match):
        num = int(match.group(1))
        return f"_{num + 1}iter.json"
    
    # Use re.sub() to replace the matched part of the string using the repl function
    return re.sub(pattern, repl, filename)

# def custom_avg_pool(input_tensor, k):
#     """
#     Apply a custom average pooling operation.
    
#     Arguments:
#     - input_tensor (torch.Tensor): The input tensor with shape [H, W], where H and W are height and width respectively.
#     - k (int): Kernel size for average pooling.
    
#     Returns:
#     - expanded_similarities (torch.Tensor): Tensor after custom average pooling operation.
#     """

#     # Reshape to 7x7
#     pdb.set_trace()
#     reshaped_similarities = input_tensor.view(7, 7)

#     # Expand each value to a 2x2 block
#     expanded_similarities = reshaped_similarities.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)    
    
#     height, width = expanded_similarities.shape
#     k = 2  # 2x2 window
#     for i in range(height - k + 1):  # Subtracting k to avoid running over the edge
#         for j in range(width - k + 1):
#             # Compute the average for the 2x2 block
#             block_avg = expanded_similarities[i:i+k, j:j+k].mean()
            
#             # Assign the block average to the new tensor's corresponding block
#             expanded_similarities[i:i+k, j:j+k] = block_avg
#     expanded_similarities = expanded_similarities.view(14*14)
#     return expanded_similarities

def custom_avg_pool(input_tensor, k):
    """
    Apply a custom average pooling operation.
    
    Arguments:
    - input_tensor (torch.Tensor): The input tensor with shape [H, W], where H and W are height and width respectively.
    - k (int): Kernel size for average pooling.
    
    Returns:
    - torch.Tensor: Tensor after custom average pooling operation.
    """
    
    # Assuming the indices form a square grid, reshape to 2D
    dim = int(input_tensor.size(0) ** 0.5)
    reshaped_similarities = input_tensor.reshape(dim, dim)

    
    # if block_size % k != 0:
    #     raise ValueError(f"Input tensor size ({block_size}) is not divisible by kernel size ({k}).")
    
    # Expand each value to a kxk block
    expanded_similarities = reshaped_similarities.repeat_interleave(k, dim=0).repeat_interleave(k, dim=1)
    height, width = expanded_similarities.shape
    for i in range(0, height - k + 1, k):  # Increment by k to avoid overlap
        for j in range(0, width - k + 1, k):
            # Compute the average for the kxk block
            block_avg = expanded_similarities[i:i+k, j:j+k].mean()
            
            # Assign the block average to the tensor's corresponding block
            expanded_similarities[i:i+k, j:j+k] = block_avg
    
    expanded_similarities = expanded_similarities.view(14*14)
    return expanded_similarities



@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model

@TRAINER_REGISTRY.register()
class ZeroshotCLIP_rank(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        backbone_name = cfg.MODEL.BACKBONE.NAME
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        try:
            with open(cfg.TRAINER.COOP.TOKENS_SCORE, 'r') as f:
                self.img_tokens_rank_dict = json.load(f)
                # Convert string keys to integer keys for each nested dictionary
                for impath, impath_data in self.img_tokens_rank_dict.items():
                    self.img_tokens_rank_dict[impath] = {int(token_index): token_data for token_index, token_data in impath_data.items()}
        except FileNotFoundError:
            split = cfg.TRAINER.COOP.SPLIT
            self.img_tokens_rank_dict = self.build_init_rank_dict(split=split)
            # # Save the rank_dict as a JSON file
            # with open(cfg.TRAINER.COOP.TOKENS_SCORE, 'w') as json_file:
            #     json.dump(self.img_tokens_rank_dict, json_file, indent=4)  # indent=4 will format the JSON for better readability
            
        design_details = {"trainer": 'CoOp',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0,
                        "is_vision": False,
                        "rank_mode": cfg.TRAINER.COOP.RANK_MODE,
                        "is_prune": cfg.TRAINER.COOP.IS_PRUNE,
                        "pruning_loc": cfg.TRAINER.COOP.PRUNING_LOC,
                        "num_token": cfg.TRAINER.COOP.NUM_TOKEN,
                        "eval_rank": cfg.TRAINER.COOP.EVAL_RANK,
                        "eval_prune": cfg.TRAINER.COOP.EVAL_PRUNE,
                        "eval_mask": cfg.TRAINER.COOP.EVAL_MASK,
                        "train_rank": cfg.TRAINER.COOP.TRAIN_RANK,
                        "block_size": cfg.TRAINER.COOP.BLOCK_SIZE,
                        "tokens_score": self.img_tokens_rank_dict}
        self.design_details = design_details
        clip_model = clip.build_model(state_dict or model.state_dict(), design_details)
        clip_model.to(self.device)
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        impath = batch["impath"]
    
        input = input.to(self.device)
        label = label.to(self.device)

        return input, label, impath
    
    @torch.no_grad()
    def build_init_rank_dict(self, split=None):
        print('Build init rank dict...')
        """A generic testing pipeline."""
        print('Using', split, 'split')
        self.set_model_mode("eval")
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader_x
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
            
        img_rank_dict = {}
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if is_debug:
                if batch_idx == 10:
                    break
            input, label, impath = self.parse_batch_test(batch)
            # Construct the nested dictionary for this impath
            impath_dict = {
                token_index: {
                    'iter': 1,
                    'score': -1, 
                    'is_keep': 1
                } for token_index in range(196)  # Loop for each token_index from 0 to 195
            }
            img_rank_dict[impath[0]] = impath_dict
        return img_rank_dict

    @torch.no_grad()
    def test_rank(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print('val')
        elif split == "train":
            data_loader = self.train_loader_x
            print('train')
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
            print('test')

        print(f"Evaluate on the *{split}* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if is_debug:
                if batch_idx == 10:
                    break
            input, label, impath = self.parse_batch_test(batch)
            if self.design_details['rank_mode']=='similarity':
                output, similarities = self.model_inference(input, impath)
                scores_list = similarities.tolist()
            else:
                output = self.model_inference(input, impath)
                if self.design_details['rank_mode'] == 'max':
                    tokens_confidence = output[:,output.max(1)[1][0].item()]
                elif self.design_details['rank_mode'] == 'label':           
                    tokens_confidence = output[:,label.item()]
                # Convert tokens_confidence tensor to a list
                scores_list = tokens_confidence.tolist()
                
            label = label.repeat(output.size(0))
            self.evaluator.process(output, label)

            # Update the 'score' field in self.img_tokens_rank_dict[impath[0]] for each index
            for idx, score in enumerate(scores_list):
                if self.img_tokens_rank_dict[impath[0]][idx]['is_keep'] == 1:
                    self.img_tokens_rank_dict[impath[0]][idx]['score'] = score
        results = self.evaluator.evaluate()

        with open(self.cfg.TRAINER.COOP.TOKENS_SCORE, "w") as outfile:
            json.dump(self.img_tokens_rank_dict, outfile, indent=4)
        
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        return list(results.values())[0]

    @torch.no_grad()
    def test_prune(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader_x
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
        print(f"Evaluate on the *{split}* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if is_debug:
                if batch_idx == 10:
                    break
            input, label, impath = self.parse_batch_test(batch)
            output = self.model_inference(input, impath, is_prune=self.design_details['is_prune'])
            self.evaluator.process(output, label)
        results = self.evaluator.evaluate()
        
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
            new_file_dir = increment_iter_in_filename(self.cfg.TRAINER.COOP.TOKENS_SCORE)
        with open(new_file_dir, "w") as outfile:
            json.dump(self.img_tokens_rank_dict, outfile, indent=4)
        return list(results.values())[0]

    def model_inference(self, image, impath=None, is_prune=False):
        if self.design_details['rank_mode']=='similarity' and not is_prune:
            image_features, full_image_features = self.clip_model.encode_image(image, impath)
            image_features_fp32 = image_features.to(dtype=torch.float32)
            full_image_features_fp32 = full_image_features.to(dtype=torch.float32)
            similarities = F.cosine_similarity(image_features_fp32, full_image_features_fp32, dim=1)
            k = self.design_details['block_size']
            moving_similarities = custom_avg_pool(similarities, k)
        else:           
            image_features = self.clip_model.encode_image(image, impath)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        if self.design_details['rank_mode']=='similarity' and not is_prune:
            return logits, moving_similarities
        else:
            return logits

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        preprocess = transforms.Compose([
            transforms.Resize(size=224, interpolation=INTERPOLATION_MODES['bicubic'], max_size=None, antialias=None),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]
        )
        dm = DataManager(self.cfg, custom_tfm_train=preprocess)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    @torch.no_grad()
    def test_mask(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        self.design_details
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if is_debug:
                if batch_idx == 10:
                    break
            input, label, impath = self.parse_batch_test(batch)
            output = self.model_inference(input, impath)
            self.evaluator.process(output, label)
        results = self.evaluator.evaluate()
        
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        # with open(self.cfg.TRAINER.COOP.TOKENS_SCORE, "w") as outfile:
        #     json.dump(self.img_tokens_rank_dict, outfile, indent=4)
        return list(results.values())[0]

    # def model_inference(self, image, impath=None):
    #     image_features = self.clip_model.encode_image(image, impath)
    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     logit_scale = self.clip_model.logit_scale.exp()
    #     logits = logit_scale * image_features @ self.text_features.t()
    #     return logits


@TRAINER_REGISTRY.register()
class ZeroshotCLIP_predictor(TrainerX):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        backbone_name = cfg.MODEL.BACKBONE.NAME
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        with open(cfg.TRAINER.COOP.TOKENS_SCORE, 'r') as f:
            self.img_tokens_rank_dict = json.load(f)
            # Convert string keys to integer keys for each nested dictionary
            for impath, impath_data in self.img_tokens_rank_dict.items():
                self.img_tokens_rank_dict[impath] = {int(token_index): token_data for token_index, token_data in impath_data.items()}

            
            
        design_details = {"trainer": 'CoOp',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0,
                        "is_vision": False,
                        "use_similarity": False,
                        "pruning_loc": cfg.TRAINER.COOP.PRUNING_LOC,
                        "num_token": cfg.TRAINER.COOP.NUM_TOKEN,
                        "eval_rank": cfg.TRAINER.COOP.EVAL_RANK,
                        "eval_prune": cfg.TRAINER.COOP.EVAL_PRUNE,
                        "eval_mask": cfg.TRAINER.COOP.EVAL_MASK,
                        "train_rank": cfg.TRAINER.COOP.TRAIN_RANK,
                        "tokens_score": self.img_tokens_rank_dict}
        self.design_details = design_details
        clip_model = clip.build_model(state_dict or model.state_dict(), design_details)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        impath = batch["impath"]
    
        input = input.to(self.device)
        label = label.to(self.device)

        return input, label, impath


    def forward_backward(self, batch):
        image, label, impath = self.parse_batch_train(batch)
        pdb.set_trace()
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

        # print(f"Evaluate on the *{split}* set")
        # self.design_details
        # for batch_idx, batch in enumerate(tqdm(data_loader)):
        #     if is_debug:
        #         if batch_idx == 10:
        #             break
        #     input, label, impath = self.parse_batch_test(batch)
        #     output = self.model_inference(input, impath)
        #     self.evaluator.process(output, label)
        # results = self.evaluator.evaluate()
        
        # for k, v in results.items():
        #     tag = f"{split}/{k}"
        #     self.write_scalar(tag, v, self.epoch)
        # # with open(self.cfg.TRAINER.COOP.TOKENS_SCORE, "w") as outfile:
        # #     json.dump(self.img_tokens_rank_dict, outfile, indent=4)
        # return list(results.values())[0]
