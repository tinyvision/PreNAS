import copy
import torch
import math
import logging
from torch import optim
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import suppress
from scipy import stats
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import random
import functools
import torch.distributed as dist
from typing import Iterable, Optional
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from model.supernet_transformer import Vision_TransformerSuper


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def gen_key(embed_dim, choice):
    return f'{embed_dim},{choice}'


class ScoreMaker(object):
    def __init__(self):
        self.grad_dict_before_train = {}
        self.grad_dict_after_train = {}
        self.param_val_dict = {}
        self.item_score_dict = {}
        self.key_items = ['attn.qkv.weight', 'fc1.weight']

    def drop_gradient(self):
        self.grad_dict = None

    def nan_to_zero(self, a):
        return torch.where(torch.isnan(a), torch.full_like(a, 0), a)

    def build_avg_image(self, s):
        # 3 channel in image
        assert s[1] == 3

        img = torch.zeros(s)
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        for i in range(3):
            torch.nn.init.normal_(img[:, i, :, :], mean=mean[i], std=std[i])
        return img

    def get_gradient(self, model, criterion, data_loader, args, choices, device, mixup_fn: Optional[Mixup] = None):
        config = {}
        dimensions = ['mlp_ratio', 'num_heads']
        depth = max(choices['depth'])
        for dimension in dimensions:
            config[dimension] = [max(choices[dimension]) for _ in range(depth)]
        config['embed_dim'] = [max(choices['embed_dim'])] * depth
        config['layer_num'] = depth

        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)

        model.train()
        criterion.train()

        random.seed(0)

        optimizer = create_optimizer(args, model_module)

        batch_num = 0
        grad_dict = {}

        optimizer.zero_grad()

        for samples, targets in data_loader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            if args.data_free:
                input_dim = list(samples[0, :].shape)
                inputs = self.build_avg_image([64] + input_dim).to(device)             # 64 batch image
                output = model.forward(inputs)
                torch.sum(output).backward()
                batch_num += 1
                print('data free!')
                break

            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss.backward()
            batch_num += 1

        for k, param in model_module.named_parameters():
            if param.requires_grad:
                grad_dict[k] = param.grad

        if args.distributed:
            dist.barrier()
            result = torch.tensor([batch_num]).to(args.device, non_blocking=True)
            dist.all_reduce(result)
            batch_num = result[0]
            for k, v in grad_dict.items():
                dist.all_reduce(grad_dict[k])

        for k, v in grad_dict.items():
            grad_dict[k] = v / batch_num
            # grad_dict[k] = grad_dict[k].cpu()

        self.grad_dict = grad_dict

    def get_block_scores(self, model, args, choices):
        model_module = unwrap_model(model)
        param_val_dict = model_module.state_dict()

        head_choices = choices['num_heads']
        mlp_choices = choices['mlp_ratio']
        layers = max(choices['depth'])
        embed_dims = choices['embed_dim']
        max_dim = max(choices['embed_dim'])
        head_dim = model_module.super_embed_dim // model_module.super_num_heads

        head_score_dict = {}
        mlp_score_dict = {}
        for embed_dim in embed_dims:
            for j in range(len(head_choices) - 1):
                head_score_dict[gen_key(embed_dim, head_choices[j + 1])] = []
            for j in range(len(mlp_choices) - 1):
                mlp_score_dict[gen_key(embed_dim, mlp_choices[j + 1])] = []

        def get_item_score(pv, gv, block_score_method):
            pg = pv.mul(gv)

            if 'balance_taylor6_norm' in block_score_method:
                item_score = pg.abs() / pv.abs().sum() / gv.abs().sum() / pg.abs().sum()
            elif 'taylor6_doublenorm' in block_score_method:
                item_score = pg.abs() / pg.abs().sum() / pg.abs().sum()
            elif 'taylor6_norm' in block_score_method:
                item_score = pg.abs() / pg.abs().sum()
            elif 'balance_taylor6' in block_score_method:
                item_score = pg.abs() / pv.abs().sum() / gv.abs().sum()
            elif 'taylor6' in block_score_method:
                item_score = pg.abs()
            elif 'balance_taylor5_norm' in block_score_method:
                item_score = pg / pv.sum().abs() / gv.sum().abs() / pg.sum().abs()
            elif 'taylor5_doublenorm' in block_score_method:
                item_score = pg / pg.sum().abs() / pg.sum().abs()
            elif 'taylor5_norm' in block_score_method:
                item_score = pg / pg.sum().abs()
            elif 'balance_taylor5' in block_score_method:
                item_score = pg / pv.sum().abs() / gv.sum().abs()
            elif 'taylor5' in block_score_method:
                item_score = pg
            elif 'taylor9_norm' in block_score_method:
                item_score = gv.abs() / gv.abs().sum()
            elif 'taylor9_doublenorm' in block_score_method:
                item_score = gv.abs() / gv.abs().sum() / gv.abs().sum()
            elif 'taylor9' in block_score_method:
                item_score = gv.abs()
            elif 'l1norm' in block_score_method:
                item_score = pv.abs()
            else:
                item_score = pv
            return item_score

        for embed_dim in embed_dims:
            for i in range(layers):
                qkv_w = f'blocks.{i}.attn.qkv.weight'
                c_fc_w = f'blocks.{i}.fc1.weight'
                c_proj_w = f'blocks.{i}.fc2.weight'
                qkv_score = get_item_score(param_val_dict[qkv_w][:, :max_dim],
                                           self.grad_dict[qkv_w][:, :max_dim],
                                           args.block_score_method_for_head)[:, :embed_dim]
                c_fc_score = get_item_score(param_val_dict[c_fc_w][:, :max_dim],
                                            self.grad_dict[c_fc_w][:, :max_dim],
                                            args.block_score_method_for_mlp)[:, :embed_dim]
                c_proj_score = get_item_score(param_val_dict[c_proj_w][:max_dim, :],
                                              self.grad_dict[c_proj_w][:max_dim, :],
                                              args.block_score_method_for_mlp)[:embed_dim, :]
                for j in range(len(head_choices) - 1):
                    qkv_embed_base = head_dim * head_choices[j]
                    qkv_embed_dim = head_dim * head_choices[j + 1]
                    left_qkv_score = torch.cat([qkv_score[qkv_embed_base * 3 + k:qkv_embed_dim * 3:3, :] for k in range(3)], dim=0)
                    score = left_qkv_score.sum().abs().cpu()
                    head_score_dict[gen_key(embed_dim, head_choices[j + 1])].append(score)
                for j in range(len(mlp_choices) - 1):
                    mlp_embed_base = int(embed_dim * mlp_choices[j])
                    mlp_embed_dim = int(embed_dim * mlp_choices[j + 1])
                    left_c_fc_score = c_fc_score[mlp_embed_base:mlp_embed_dim, :]
                    left_c_proj_score = c_proj_score[:, mlp_embed_base:mlp_embed_dim]
                    score = left_c_fc_score.sum().abs().cpu() + left_c_proj_score.sum().abs().cpu()
                    mlp_score_dict[gen_key(embed_dim, mlp_choices[j + 1])].append(score)

        return {'head_scores': head_score_dict, 'mlp_scores': mlp_score_dict}


    def get_item_score(self, model, criterion, data_loader, args, choices, device, mixup_fn: Optional[Mixup] = None):

        config = {}
        dimensions = ['mlp_ratio', 'num_heads']
        depth = max(choices['depth'])
        for dimension in dimensions:
            config[dimension] = [max(choices[dimension]) for _ in range(depth)]
        config['embed_dim'] = [max(choices['embed_dim'])] * depth
        config['layer_num'] = depth

        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)

        param_val_dict = model_module.state_dict()
        grad_dict = {}

        if 'taylor' in args.score_method:
            model.train()
            criterion.train()

            random.seed(0)

            optimizer = create_optimizer(args, model_module)

            batch_num = 0

            for samples, targets in data_loader:
                samples = samples.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if mixup_fn is not None:
                    samples, targets = mixup_fn(samples, targets)

                outputs = model(samples)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()

                for k, param in model_module.named_parameters():
                    if param.requires_grad:
                        if batch_num == 0:
                            grad_dict[k] = copy.deepcopy(param.grad)
                        else:
                            grad_dict[k] = grad_dict[k] + param.grad

                batch_num += 1

            if args.distributed:
                dist.barrier()
                result = torch.tensor([batch_num]).to(args.device, non_blocking=True)
                dist.all_reduce(result)
                batch_num = result[0]
                for k, v in grad_dict.items():
                    dist.all_reduce(grad_dict[k])

            for k, v in grad_dict.items():
                grad_dict[k] = v / batch_num
                grad_dict[k] = grad_dict[k]

        for k in param_val_dict.keys():
            for key_item in self.key_items:
                if key_item in k:
                    if 'l1norm' in args.score_method:
                        self.item_score_dict[k] = param_val_dict[k].abs().cpu()
                    elif 'taylor5' in args.score_method:
                        self.item_score_dict[k] = param_val_dict[k].mul(grad_dict[k]).cpu()
                    elif 'taylor6' in args.score_method:
                        self.item_score_dict[k] = param_val_dict[k].mul(grad_dict[k]).abs().cpu()
                    elif 'taylor9' in args.score_method:
                        self.item_score_dict[k] = grad_dict[k].abs().cpu()
                    else:
                        assert False

    def get_head_score(self, head_choices, layers, head_dim, embed_dims, layer_norm=False):
        score_dict = {}
        for embed_dim in embed_dims:
            for j in range(len(head_choices) - 1):
                score_dict[gen_key(embed_dim, head_choices[j + 1])] = []
        for embed_dim in embed_dims:
            for i in range(layers):
                qkv_w = f'blocks.{i}.attn.qkv.weight'
                for j in range(len(head_choices) - 1):
                    qkv_embed_base = head_dim * head_choices[j]
                    qkv_embed_dim = head_dim * head_choices[j + 1]
                    qkv_score = self.item_score_dict[qkv_w][:,:embed_dim]
                    item_score = torch.cat([qkv_score[qkv_embed_base * 3 + k:qkv_embed_dim * 3:3, :] for k in range(3)], dim=0)
                    score = item_score.sum().abs()
                    if layer_norm:
                        score = score / qkv_score.sum().abs()
                    score_dict[gen_key(embed_dim, head_choices[j + 1])].append(score)
        return score_dict

    def get_mlp_score(self, mlp_choices, layers, embed_dims, layer_norm=False):
        score_dict = {}
        for embed_dim in embed_dims:
            for j in range(len(mlp_choices) - 1):
                score_dict[gen_key(embed_dim, mlp_choices[j + 1])] = []
        for embed_dim in embed_dims:
            for i in range(layers):
                c_fc_w = f'blocks.{i}.fc1.weight'
                for j in range(len(mlp_choices) - 1):
                    mlp_embed_base = int(embed_dim * mlp_choices[j])
                    mlp_embed_dim = int(embed_dim * mlp_choices[j+1])
                    mlp_score = self.item_score_dict[c_fc_w][:, :embed_dim]
                    item_score = mlp_score[mlp_embed_base:mlp_embed_dim, :]
                    score = item_score.sum().abs()
                    if layer_norm:
                        score = score / mlp_score.sum().abs()
                    score_dict[gen_key(embed_dim, mlp_choices[j + 1])].append(score)
        return score_dict

    def get_left_part_from_super_model(self, model: Vision_TransformerSuper, para_dict, sample_config):
        layers = model.super_layer_num
        sample_layers = sample_config['layer_num']
        left_dict = {}

        embed_dims = sample_config['embed_dim']
        output_dims = [out_dim for out_dim in sample_config['embed_dim'][1:]] + [sample_config['embed_dim'][-1]]

        left_dict['patch_embed_super.proj.weight'] = para_dict['patch_embed_super.proj.weight'][:embed_dims[0], ...]
        left_dict['patch_embed_super.proj.bias'] = para_dict['patch_embed_super.proj.bias'][:embed_dims[0], ...]
        left_dict['norm.weight'] = para_dict['norm.weight'][:embed_dims[-1]]
        left_dict['norm.bias'] = para_dict['norm.bias'][:embed_dims[-1]]
        left_dict['head.weight'] = para_dict['head.weight'][:, :embed_dims[-1]]
        left_dict['head.bias'] = para_dict['head.bias'][:embed_dims[-1]]

        for i in range(layers):
            qkv_w = f'blocks.{i}.attn.qkv.weight'
            qkv_b = f'blocks.{i}.attn.qkv.bias'
            proj_w = f'blocks.{i}.attn.proj.weight'
            proj_b = f'blocks.{i}.attn.proj.bias'
            ln1_w = f'blocks.{i}.attn_layer_norm.weight'
            ln1_b = f'blocks.{i}.attn_layer_norm.bias'
            c_fc_w = f'blocks.{i}.fc1.weight'
            c_fc_b = f'blocks.{i}.fc1.bias'
            c_proj_w = f'blocks.{i}.fc2.weight'
            c_proj_b = f'blocks.{i}.fc2.bias'
            ln2_w = f'blocks.{i}.ffn_layer_norm.weight'
            ln2_b = f'blocks.{i}.ffn_layer_norm.bias'
            if i < sample_layers:
                num_heads = sample_config['num_heads'][i]
                head_dim = model.super_embed_dim // model.super_num_heads
                qk_embed_dim = head_dim * num_heads
                mlp_ratio = sample_config['mlp_ratio'][i]
                embed_dim = embed_dims[i]
                mlp_width = int(embed_dim * mlp_ratio)
                output_dim = output_dims[i]

                left_dict[qkv_w] = para_dict[qkv_w][:, :embed_dim]
                left_dict[qkv_w] = torch.cat([left_dict[qkv_w][i:qk_embed_dim*3:3, :] for i in range(3)], dim=0)

                # left_dict[qkv_b] = para_dict[qkv_b][:qk_embed_dim*3]
                left_dict[qkv_b] = torch.cat([para_dict[qkv_b][i:qk_embed_dim*3:3] for i in range(3)])

                left_dict[proj_w] = para_dict[proj_w][:, :qk_embed_dim]
                left_dict[proj_w] = left_dict[proj_w][:embed_dim, :]

                left_dict[proj_b] = para_dict[proj_b][:embed_dim]

                left_dict[ln1_w] = para_dict[ln1_w][:embed_dim]

                left_dict[ln1_b] = para_dict[ln1_b][:embed_dim]

                left_dict[c_fc_w] = para_dict[c_fc_w][:, :embed_dim]
                left_dict[c_fc_w] = left_dict[c_fc_w][:mlp_width, :]

                left_dict[c_fc_b] = para_dict[c_fc_b][:mlp_width]

                left_dict[c_proj_w] = para_dict[c_proj_w][:, :mlp_width]
                left_dict[c_proj_w] = left_dict[c_proj_w][:output_dim, :]

                left_dict[c_proj_b] = para_dict[c_proj_b][:output_dim]

                left_dict[ln2_w] = para_dict[ln2_w][:output_dim]

                left_dict[ln2_b] = para_dict[ln2_b][:output_dim]
            else:
                continue

        num_paras = 0
        for k, v in left_dict.items():
            num_paras += v.numel()

        return left_dict, num_paras

    def get_left_attn_mlp_from_super_model(self, model: Vision_TransformerSuper, para_dict, sample_config):

        layers = model.super_layer_num
        sample_layers = sample_config['layer_num']
        left_dict = {}

        embed_dims = sample_config['embed_dim']
        output_dims = [out_dim for out_dim in sample_config['embed_dim'][1:]] + [sample_config['embed_dim'][-1]]

        for i in range(layers):
            qkv_w = f'blocks.{i}.attn.qkv.weight'
            qkv_b = f'blocks.{i}.attn.qkv.bias'
            proj_w = f'blocks.{i}.attn.proj.weight'
            proj_b = f'blocks.{i}.attn.proj.bias'
            c_fc_w = f'blocks.{i}.fc1.weight'
            c_fc_b = f'blocks.{i}.fc1.bias'
            c_proj_w = f'blocks.{i}.fc2.weight'
            c_proj_b = f'blocks.{i}.fc2.bias'
            if i < sample_layers:
                num_heads = sample_config['num_heads'][i]
                head_dim = model.super_embed_dim // model.super_num_heads
                qk_embed_dim = head_dim * num_heads
                mlp_ratio = sample_config['mlp_ratio'][i]
                embed_dim = embed_dims[i]
                mlp_width = int(embed_dim * mlp_ratio)
                output_dim = output_dims[i]

                left_dict[qkv_w] = para_dict[qkv_w][:, :embed_dim]
                left_dict[qkv_w] = torch.cat([left_dict[qkv_w][i:qk_embed_dim * 3:3, :] for i in range(3)], dim=0)

                # left_dict[qkv_b] = para_dict[qkv_b][:qk_embed_dim*3]
                left_dict[qkv_b] = torch.cat([para_dict[qkv_b][i:qk_embed_dim * 3:3] for i in range(3)], dim=0)

                left_dict[proj_w] = para_dict[proj_w][:, :qk_embed_dim]
                left_dict[proj_w] = left_dict[proj_w][:embed_dim, :]

                left_dict[proj_b] = para_dict[proj_b][:embed_dim]

                left_dict[c_fc_w] = para_dict[c_fc_w][:, :embed_dim]
                left_dict[c_fc_w] = left_dict[c_fc_w][:mlp_width, :]

                left_dict[c_fc_b] = para_dict[c_fc_b][:mlp_width]

                left_dict[c_proj_w] = para_dict[c_proj_w][:, :mlp_width]
                left_dict[c_proj_w] = left_dict[c_proj_w][:output_dim, :]

                left_dict[c_proj_b] = para_dict[c_proj_b][:output_dim]
            else:
                continue

        num_paras = 0
        for k, v in left_dict.items():
            num_paras += v.numel()

        return left_dict, num_paras

    def get_scores(self, model, score_methods, config):
        score_methods = score_methods.strip().split('+')
        scores = []
        for score_method in score_methods:
            scores.append(self.get_score(model, score_method, config))
        return scores

    def get_score(self, model, score_method, config):

        if score_method == 'entropy':
            depth, mlp_ratio, num_heads, embed_dim = config['layer_num'], config['mlp_ratio'], config['num_heads'], config['embed_dim']
            entropy_score = 0.
            for i in range(depth):
                d = embed_dim[i]
                n = 14 * 14  # input_size = 224, patch_size = 16
                d_f = mlp_ratio[i] * d
                d_h = 64
                n_h = num_heads[i]
                entropy_score += math.log(d_f) + math.log(d_h * n_h) + math.log(n) + 4 * math.log(d)
            return entropy_score

        super_paras = unwrap_model(model).state_dict()

        if 'left_attn_mlp' in score_method:
            para, num_paras = self.get_left_attn_mlp_from_super_model(unwrap_model(model), super_paras, config)
        else:
            para, num_paras = self.get_left_part_from_super_model(unwrap_model(model), super_paras, config)
        grad = None
        if 'taylor' in score_method:
            if 'left_attn_mlp' in score_method:
                grad, _ = self.get_left_attn_mlp_from_super_model(unwrap_model(model), self.grad_dict, config)
            else:
                grad, _ = self.get_left_part_from_super_model(unwrap_model(model), self.grad_dict, config)

        if 'avg' not in score_method:
            num_paras = None

        if 'l1norm' in score_method:
            res = self.criterion_l_l1norm(para, num_paras)
        elif 'l1norm_norm' in score_method:
            res = self.criterion_l_l1norm(paras, super_paras=super_paras)
        elif 'taylor5' in score_method:
            res = self.criterion_l_taylor5(para, grad, num_paras)
        elif 'taylor5_norm' in score_method:
            res = self.criterion_l_taylor5(paras, grads, super_paras=super_paras, super_grads=super_grads)
        elif 'taylor6' in score_method:
            res = self.criterion_l_taylor6(para, grad, num_paras)
        elif 'taylor6_norm' in score_method:
            res = self.criterion_l_taylor6(paras, grads, super_paras=super_paras, super_grads=super_grads)
        elif 'taylor9' in score_method:
            res = self.criterion_l_taylor9(para, grad, num_paras)
        elif 'taylor9_norm' in score_method:
            res = self.criterion_l_taylor9(paras, grads, super_paras=super_paras, super_grads=super_grads)
        else:
            assert False

        if 'pruned' in score_method:
            res = - res
        if type(res) == float:
            return res
        else:
            return res.cpu()

    def criterion_l_l1norm(self, paras, num_paras=None, super_paras=None):
        score = 0.
        for k, v in paras.items():
            if super_paras is not None:
                score += v.abs().sum() / super_paras[k].abs().sum()
            else:
                score += v.abs().sum()
        if num_paras:
            score /= num_paras
        return score.cpu()

    def criterion_l_l2norm(self, paras, num_paras=None, super_paras=None):
        score = 0.
        for k, v in paras.items():
            if super_paras is not None:
                score += v.norm() / super_paras[k].norm()
            else:
                score += v.norm()
        if num_paras:
            score /= num_paras
        return score.cpu()

    def criterion_l_taylor1(self, paras, grads, num_paras=None, super_paras=None, super_grads=None):
        score = 0.
        for k, v in paras.items():
            g = grads[k]
            if super_paras is not None and super_grads is not None:
                score += v.mul(g).sum() / super_paras[k].mul(super_grads[k]).sum()
            else:
                score += v.mul(g).sum()
        if num_paras:
            score /= num_paras
        score = score ** 2
        return score.cpu()

    def criterion_l_taylor2(self, paras, grads, num_paras=None, super_paras=None, super_grads=None):        # fisher
        score = 0.
        for k, v in paras.items():
            g = grads[k]
            if super_paras is not None and super_grads is not None:
                score += (v.mul(g) ** 2).sum() / (super_paras[k].mul(super_grads[k]) ** 2).sum()
            else:
                score += (v.mul(g) ** 2).sum()
        if num_paras:
            score /= num_paras
        return score.cpu()

    def criterion_l_taylor3(self, paras, grads, num_paras=None, super_paras=None, super_grads=None):
        score = 0.
        for k, v in paras.items():
            g = grads[k]
            if super_paras is not None and super_grads is not None:
                score += g.sum() / super_grads[k].sum()
            else:
                score += g.sum()
        if num_paras:
            score /= num_paras
        score = score ** 2
        return score.cpu()

    def criterion_l_taylor4(self, paras, grads, num_paras=None, super_paras=None, super_grads=None):
        score = 0.
        for k, v in paras.items():
            g = grads[k]
            if super_paras is not None and super_grads is not None:
                score += (g ** 2).sum() / (super_grads[k] ** 2).sum()
            else:
                score += (g ** 2).sum()
        if num_paras:
            score /= num_paras
        return score.cpu()

    def criterion_l_taylor5(self, paras, grads, num_paras=None, super_paras=None, super_grads=None):        # synflow
        score = 0.
        for k, v in paras.items():
            g = grads[k]
            if super_paras is not None and super_grads is not None:
                score += v.mul(g).sum() / super_paras[k].mul(super_grads[k]).sum()
            else:
                score += v.mul(g).sum()
        score = score.abs()
        if num_paras:
            score /= num_paras
        return score.cpu()

    def criterion_l_taylor6(self, paras, grads, num_paras=None, super_paras=None, super_grads=None):
        score = 0.
        for k, v in paras.items():
            g = grads[k]
            if super_paras is not None and super_grads is not None:
                score += v.mul(g).abs().sum() / super_paras[k].mul(super_grads[k]).abs().sum()
            else:
                score += v.mul(g).abs().sum()
        if num_paras:
            score /= num_paras
        return score.cpu()

    def criterion_l_taylor7(self, paras, grads, num_paras=None, super_paras=None, super_grads=None):        # Euclidean norm of the gradients v1
        score = 0.
        for k, v in paras.items():
            g = grads[k]
            if super_paras is not None and super_grads is not None:
                score += (g ** 2).sum() / (super_grads[k] ** 2).sum()
            else:
                score += (g ** 2).sum()
        if num_paras:
            score /= num_paras
        score = math.sqrt(score)
        return score

    def criterion_l_taylor8(self, paras, grads, num_paras=None, super_paras=None, super_grads=None):        # Euclidean norm of the gradients v2
        score = 0.
        for k, v in paras.items():
            g = grads[k]
            if super_paras is not None and super_grads is not None:
                score += g.norm() / super_grads[k].norm()
            else:
                score += g.norm()
        if num_paras:
            score /= num_paras
        return score.cpu()

    def criterion_l_taylor9(self, paras, grads, num_paras=None, super_paras=None, super_grads=None):        # snip
        score = 0.
        for k, v in paras.items():
            g = grads[k]
            if super_paras is not None and super_grads is not None:
                score += g.abs().sum() / super_grads[k].abs().sum()
            else:
                score += g.abs().sum()
        if num_paras:
            score /= num_paras
        score = math.sqrt(score)
        return score

    def sort_eval(self, vis_dict, top_num=50):
        acc_list = []
        score_lists = {}
        for cand, info in vis_dict.items():
            # only consider the model under limits
            if 'acc' not in info.keys():
                continue
            acc_list.append(info['acc'])
            score_stats = info['score_stats']
            for k, score in score_stats.items():
                if k not in score_lists.keys():
                    score_lists[k] = []
                if type(score) == float:
                    score_lists[k].append(score)
                else:
                    score_lists[k].append(score.cpu())
        acc_list = np.array(acc_list)
        for k in score_lists.keys():
            score_lists[k] = np.array(score_lists[k])

        p_vals = self.get_p_value(acc_list, score_lists)
        p_dict = {}
        for (p_val, k) in p_vals:
            p_dict[k] = p_val

        idx = acc_list.argsort()[-top_num:][::-1]
        sorted_acc = [(acc_list[idx], 'acc_list')]
        for k, scores in score_lists.items():
            if p_dict[k] > 0:
                idx = scores.argsort()[-top_num:][::-1]
            else:
                idx = scores.argsort()[:top_num]
            acc = acc_list[idx]
            acc.sort()
            sorted_acc.append((acc[::-1], k))

        def compare(a, b):
            for i in range(a[0].size):
                if a[0][i] > b[0][i]:
                    return 1
                elif a[0][i] < b[0][i]:
                    return -1
            return 0

        sorted_acc.sort(key=functools.cmp_to_key(compare))
        for acc in sorted_acc:
            print(acc)

