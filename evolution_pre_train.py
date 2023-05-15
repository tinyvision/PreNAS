import random

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from lib.datasets import build_dataset
from lib import utils
from supernet_engine import evaluate
from model.supernet_transformer import Vision_TransformerSuper
import argparse
import os
import yaml
from lib.config import cfg, update_config_from_file
from lib.score_maker import ScoreMaker
import math
from itertools import combinations
import json


def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    return depth, list(cand_tuple[1:depth+1]), list(cand_tuple[depth + 1: 2 * depth + 1]), cand_tuple[-1]


def get_max_min_model(choices):
    max_depth = max(choices['depth'])
    max_emb = max(choices['embed_dim'])
    max_num_head = max(choices['num_heads'])
    max_mlp_ratio = max(choices['mlp_ratio'])
    min_depth = min(choices['depth'])
    min_emb = min(choices['embed_dim'])
    min_num_head = min(choices['num_heads'])
    min_mlp_ratio = min(choices['mlp_ratio'])
    max_model = tuple([max_depth] + [max_mlp_ratio] * max_depth + [max_num_head] * max_depth + [max_emb])
    min_model = tuple([min_depth] + [min_mlp_ratio] * min_depth + [min_num_head] * min_depth + [min_emb])
    return max_model, min_model


class Searcher(object):

    def __init__(self, args, device, model, model_without_ddp, choices, output_dir, score_maker):
        self.device = device
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.output_dir = output_dir
        self.s_prob =args.s_prob
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {}
        self.epoch = 0
        self.checkpoint_path = args.resume
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.choices = choices
        self.choices['num_heads'].sort()
        self.choices['mlp_ratio'].sort()

        self.score_maker = score_maker
        self.eval_cnt = 0
        self.update_num = 0
        self.un_update_cnt = 0

        self.all_cands = {}
        min_param = self.min_parameters_limits
        max_param = min_param + self.args.param_interval
        while max_param < self.parameters_limits + 1e-6:
            params = (max_param + min_param) / 2
            self.all_cands[self.param_to_index(params)] = []
            min_param = max_param
            max_param = min_param + self.args.param_interval

        self.cur_min_param = args.min_param_limits
        self.cur_max_param = args.param_limits
        self.interval_cands = {}
        self.max_model, self.min_model = get_max_min_model(choices)
        self.search_mode = args.search_mode
        self.head_mlp_scores = {}

    def get_params_range(self):
        depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(self.max_model)
        sampled_config = {}
        sampled_config['layer_num'] = depth
        sampled_config['mlp_ratio'] = mlp_ratio
        sampled_config['num_heads'] = num_heads
        sampled_config['embed_dim'] = [embed_dim] * depth

        n_parameters = self.model_without_ddp.get_sampled_params_numel(sampled_config)
        max_params = n_parameters / 10. ** 6

        depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(self.min_model)
        sampled_config = {}
        sampled_config['layer_num'] = depth
        sampled_config['mlp_ratio'] = mlp_ratio
        sampled_config['num_heads'] = num_heads
        sampled_config['embed_dim'] = [embed_dim] * depth

        n_parameters = self.model_without_ddp.get_sampled_params_numel(sampled_config)
        min_params = n_parameters / 10. ** 6
        return min_params, max_params

    def select_cands(self, *, key, reverse=True):
        for k in self.all_cands.keys():
            t = self.all_cands[k]
            t.sort(key=key, reverse=reverse)
            self.all_cands[k] = t[:self.args.cand_per_interval]

    def param_to_index(self, param):
        if param < self.min_parameters_limits:
            return -1
        if param >= self.parameters_limits:
            return -1
        return math.floor((param - self.min_parameters_limits) / self.args.param_interval)

    def index_to_param_interval(self, index):
        if index == -1:
            return (0, self.min_parameters_limits)
        if index == -2:
            return (self.parameters_limits, 2*self.parameters_limits)
        down = self.min_parameters_limits + index * self.args.param_interval
        up = down + self.args.param_interval
        return (down, up)

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand


    def get_random_cand_without_reallocate(self):

        cand_tuple = list()
        dimensions = ['mlp_ratio', 'num_heads']
        depth = random.choice(self.choices['depth'])
        cand_tuple.append(depth)
        for dimension in dimensions:
            idx = list(range(len(self.choices[dimension])))
            random.shuffle(idx)
            choice_cnt = {}
            left_layers = depth
            for i in idx[:-1]:
                choice = self.choices[dimension][i]
                cnt = random.choice(range(left_layers + 1))
                left_layers = left_layers - cnt
                choice_cnt[choice] = cnt
            choice = self.choices[dimension][idx[-1]]
            choice_cnt[choice] = left_layers
            conf = [0] * depth

            for choice in self.choices[dimension][1:][::-1]:
                scores = np.random.rand(depth)
                mask = np.where(np.array(conf) > 0, -1, 1)
                mask_scores = scores * mask
                for i in mask_scores.argsort()[::-1][:choice_cnt[choice]]:
                    conf[i] = choice
            for i in range(len(conf)):
                if conf[i] == 0:
                    conf[i] = self.choices[dimension][0]

            cand_tuple.extend(conf)

        cand_tuple.append(random.choice(self.choices['embed_dim']))
        return tuple(cand_tuple)

    def get_random_cand(self):

        cand_tuple = list()
        dimensions = ['mlp_ratio', 'num_heads']
        score_names = ['mlp_scores', 'head_scores']
        depth = random.choice(self.choices['depth'])
        cand_tuple.append(depth)
        emb_dim = random.choice(self.choices['embed_dim'])
        max_dim = max(self.choices['embed_dim'])
        for (dimension, score_name) in zip(dimensions, score_names):
            idx = list(range(len(self.choices[dimension])))
            random.shuffle(idx)
            choice_cnt = {}
            left_layers = depth
            for i in idx[:-1]:
                choice = self.choices[dimension][i]
                cnt = random.choice(range(left_layers + 1))
                left_layers = left_layers - cnt
                choice_cnt[choice] = cnt
            choice = self.choices[dimension][idx[-1]]
            choice_cnt[choice] = left_layers
            choice_cnt_list = [choice_cnt[choice] for choice in self.choices[dimension]]
            method = None
            if dimension == 'mlp_ratio':
                method = self.args.block_score_method_for_mlp
            else:
                method = self.args.block_score_method_for_head
            cand_tuple.extend(self.reallocate(depth,
                                              emb_dim,
                                              dimension,
                                              self.head_mlp_scores[score_name],
                                              choice_cnt_list,
                                              method))

        cand_tuple.append(emb_dim)
        return tuple(cand_tuple)

    def get_random(self, num):
        print('random select ........')
        if self.args.search_mode == 'iteration' or self.args.reallocate:
            cand_iter = self.stack_random_cand(self.get_random_cand)
        else:
            cand_iter = self.stack_random_cand(self.get_random_cand_without_reallocate)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def is_legal(self, cand):
        assert isinstance(cand, tuple)

        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
        sampled_config = {}
        sampled_config['layer_num'] = depth
        sampled_config['mlp_ratio'] = mlp_ratio
        sampled_config['num_heads'] = num_heads
        sampled_config['embed_dim'] = [embed_dim]*depth

        n_parameters = self.model_without_ddp.get_sampled_params_numel(sampled_config)
        info['params'] = n_parameters / 10.**6

        if info['params'] > self.cur_max_param:
            print('parameters limit exceed {}'.format(self.cur_max_param))
            return False

        if info['params'] < self.cur_min_param:
            print('under minimum parameters limit {}'.format(self.cur_min_param))
            return False

        info['visited'] = True

        return True

    def conf_to_cnt_list(self, conf, part):
        cnt_list = [0]*len(self.choices[part])
        for choice in conf:
            cnt_list[self.choices[part].index(choice)] += 1
        return cnt_list

    def reallocate(self, depth, embed_dim, part, scores, choice_cnt, method):

        if method == 'deeper_is_better':
            conf = []
            for choice, cnt in zip(self.choices[part], choice_cnt):
                conf = conf + ([choice] * cnt)
            return conf

        if 'max_dim' in method:
            embed_dim = max(self.choices['embed_dim'])

        conf = [0] * depth
        for choice, cnt in zip(self.choices[part][1:][::-1], choice_cnt[1:][::-1]):
            cur_scores = np.array(scores[(f"{embed_dim},{choice}")][:depth])
            mask = np.where(np.array(conf) > 0, -1, 1)
            mask_scores = cur_scores * mask
            for i in mask_scores.argsort()[::-1][:cnt]:
                conf[i] = choice
        for i in range(len(conf)):
            if conf[i] == 0:
                conf[i] = self.choices[part][0]
        return conf

    def get_score(self):
        for cand in self.candidates:
            info = self.vis_dict[cand]
            if self.args.score_method == 'params':
                info['score'] = info['params']
            else:
                depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
                sampled_config = {}
                sampled_config['layer_num'] = depth
                sampled_config['mlp_ratio'] = mlp_ratio
                sampled_config['num_heads'] = num_heads
                sampled_config['embed_dim'] = [embed_dim] * depth
                score = self.score_maker.get_score(self.model, self.args.score_method, config=sampled_config)
                info['score'] = score

    def update_top_k(self, candidates, *, k, key, reverse=True, get_update_num=False):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]
        if get_update_num:
            self.update_num = 0
            for cand in self.keep_top_k[k]:
                if cand in candidates:
                    self.update_num += 1
            print('update {} models in top {}.'.format(self.update_num, k))
            if self.update_num == 0:
                self.un_update_cnt += 1

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
            random_s = random.random()

            # depth
            if random_s < s_prob:
                new_depth = random.choice(self.choices['depth'])

                if new_depth > depth:
                    mlp_ratio = mlp_ratio + [random.choice(self.choices['mlp_ratio']) for _ in range(new_depth - depth)]
                    num_heads = num_heads + [random.choice(self.choices['num_heads']) for _ in range(new_depth - depth)]
                else:
                    mlp_ratio = mlp_ratio[:new_depth]
                    num_heads = num_heads[:new_depth]

                depth = new_depth
            # mlp_ratio
            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    mlp_ratio[i] = random.choice(self.choices['mlp_ratio'])

            # num_heads

            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    num_heads[i] = random.choice(self.choices['num_heads'])

            # embed_dim
            random_s = random.random()
            if random_s < s_prob:
                embed_dim = random.choice(self.choices['embed_dim'])

            mlp_cnt = self.conf_to_cnt_list(mlp_ratio, 'mlp_ratio')
            head_cnt = self.conf_to_cnt_list(num_heads, 'num_heads')
            mlp_ratio = self.reallocate(depth,
                                        embed_dim,
                                        'mlp_ratio',
                                        self.head_mlp_scores['mlp_scores'],
                                        mlp_cnt,
                                        self.args.block_score_method_for_mlp)
            num_heads = self.reallocate(depth,
                                        embed_dim,
                                        'num_heads',
                                        self.head_mlp_scores['head_scores'],
                                        head_cnt,
                                        self.args.block_score_method_for_head)

            result_cand = [depth] + mlp_ratio + num_heads + [embed_dim]

            return tuple(result_cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():

            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 50
            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            cand = tuple(random.choice([i, j]) for i, j in zip(p1, p2))
            depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
            mlp_cnt = self.conf_to_cnt_list(mlp_ratio, 'mlp_ratio')
            head_cnt = self.conf_to_cnt_list(num_heads, 'num_heads')
            mlp_ratio = self.reallocate(depth,
                                        embed_dim,
                                        'mlp_ratio',
                                        self.head_mlp_scores['mlp_scores'],
                                        mlp_cnt,
                                        self.args.block_score_method_for_mlp)
            num_heads = self.reallocate(depth,
                                        embed_dim,
                                        'num_heads',
                                        self.head_mlp_scores['head_scores'],
                                        head_cnt,
                                        self.args.block_score_method_for_head)
            result_cand = [depth] + mlp_ratio + num_heads + [embed_dim]
            return tuple(result_cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self, out_file_name=None):

        print('searching...')
        if not self.args.block_score_method_for_mlp == 'deeper_is_better' or not self.args.block_score_method_for_head == 'deeper_is_better':
            self.head_mlp_scores = self.score_maker.get_block_scores(self.model, self.args, self.choices)

        # random search
        if self.args.search_mode == 'random':
            self.cur_min_param = self.min_parameters_limits
            self.cur_max_param = self.cur_min_param + self.args.param_interval

            while self.cur_max_param < self.parameters_limits + 1e-6:
                self.candidates = []
                self.keep_top_k = {100: []}
                self.get_random(self.population_num)
                self.get_score()
                self.update_top_k(
                    self.candidates, k=100, key=lambda x: self.vis_dict[x]['score'])
                for i, cand in enumerate(self.keep_top_k[100]):
                    print('No.{} {} score = {}, params = {}'.format(
                        i + 1, cand, self.vis_dict[cand]['score'], self.vis_dict[cand]['params']))
                self.interval_cands[(self.cur_min_param, self.cur_max_param)] = self.keep_top_k[100][:self.args.cand_per_interval]
                self.cur_min_param = self.cur_max_param
                self.cur_max_param = self.cur_min_param + self.args.param_interval
        # evolution search
        elif self.args.search_mode == 'evolution':
            self.cur_min_param = self.min_parameters_limits
            self.cur_max_param = self.cur_min_param + self.args.param_interval

            while self.cur_max_param < self.parameters_limits + 1e-6:
                self.update_num = 0
                self.un_update_cnt = 0
                self.epoch = 0
                self.candidates = []
                self.keep_top_k = {self.select_num: [], 100: []}
                self.get_random(self.population_num)
                while self.epoch < self.max_epochs:
                    print('epoch = {} for param {} to param {}'.format(self.epoch, self.cur_min_param, self.cur_max_param))

                    if self.un_update_cnt == 2:
                        self.epoch += 1
                        continue

                    self.get_score()
                    self.update_top_k(
                        self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['score'], get_update_num=True)
                    self.update_top_k(
                        self.candidates, k=100, key=lambda x: self.vis_dict[x]['score'])

                    print('epoch = {} for param {} to param {} : top {} result'.format(
                        self.epoch, self.cur_min_param, self.cur_max_param, len(self.keep_top_k[100])))
                    for i, cand in enumerate(self.keep_top_k[100]):
                        print('No.{} {} score = {}, params = {}'.format(
                            i + 1, cand, self.vis_dict[cand]['score'], self.vis_dict[cand]['params']))

                    self.epoch += 1
                    if self.epoch >= self.max_epochs:
                        break

                    # check
                    mutation = self.get_mutation(
                        self.select_num, self.mutation_num, self.m_prob, self.s_prob)
                    crossover = self.get_crossover(self.select_num, self.crossover_num)

                    self.candidates = mutation + crossover

                    self.get_random(self.population_num)

                self.interval_cands[(self.cur_min_param, self.cur_max_param)] = self.keep_top_k[100][:self.args.cand_per_interval]
                self.cur_min_param = self.cur_max_param
                self.cur_max_param = self.cur_min_param + self.args.param_interval
        # force search
        else:
            max_dim = max(self.choices['embed_dim'])
            iter_cnt = 0
            for embed_dim in self.choices['embed_dim']:
                for depth in self.choices['depth']:
                    depth_ids = list(range(depth+1))
                    num_head_choice = len(self.choices['num_heads'])
                    num_mlp_choice = len(self.choices['mlp_ratio'])
                    mlp_confs = []
                    head_confs = []

                    for mlp_dist in combinations(depth_ids, num_mlp_choice - 1):
                        mlp_dist = [0] + list(mlp_dist) + [depth]
                        mlp_cnt = [mlp_dist[i+1] - mlp_dist[i] for i in range(len(mlp_dist)-1)]
                        mlp_confs.append(self.reallocate(depth,
                                                         embed_dim,
                                                         'mlp_ratio',
                                                         self.head_mlp_scores['mlp_scores'],
                                                         mlp_cnt,
                                                         self.args.block_score_method_for_mlp))

                    for head_dist in combinations(depth_ids, num_head_choice - 1):
                        head_dist = [0] + list(head_dist) + [depth]
                        head_cnt = [head_dist[i+1] - head_dist[i] for i in range(len(head_dist)-1)]
                        head_confs.append(self.reallocate(depth,
                                                          embed_dim,
                                                          'num_heads',
                                                          self.head_mlp_scores['head_scores'],
                                                          head_cnt,
                                                          self.args.block_score_method_for_head))

                    for mlp_conf in mlp_confs:
                        iter_cnt += 1
                        for head_conf in head_confs:
                            cand = tuple([depth] + mlp_conf + head_conf + [embed_dim])
                            depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
                            sampled_config = {}
                            sampled_config['layer_num'] = depth
                            sampled_config['mlp_ratio'] = mlp_ratio
                            sampled_config['num_heads'] = num_heads
                            sampled_config['embed_dim'] = [embed_dim] * depth
                            n_parameters = self.model_without_ddp.get_sampled_params_numel(sampled_config)
                            params = n_parameters / 10. ** 6
                            index = self.param_to_index(params)

                            if self.args.score_method == 'params':
                                score = params
                            else:
                                score = self.score_maker.get_score(self.model, self.args.score_method, config=sampled_config)

                            info = {'cand': cand, 'score': score, 'params': params}
                            self.vis_dict[cand] = info
                            if index in self.all_cands.keys():
                                self.all_cands[index].append(info)

            self.select_cands(key=lambda x: x['score'])

            for index in self.all_cands.keys():
                k = self.index_to_param_interval(index)
                self.interval_cands[k] = [item['cand'] for item in self.all_cands[index]]

        if out_file_name is None:
            out_file_name = f'out/interval_cands_{self.args.super_model_size}_{self.args.score_method}_{self.args.block_score_method_for_mlp}_for_mlp_{self.args.block_score_method_for_head}_for_head'
            out_file_name += f'_i{self.args.param_interval}_top_{self.args.cand_per_interval}.pt'
            torch.save(self.interval_cands, out_file_name)
        else:
            json_dict = {}
            for interval in self.interval_cands.keys():
                cand_list = []
                for cand in self.interval_cands[interval]:
                    depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
                    info = {
                        'layer_num': depth,
                        'mlp_ratio': mlp_ratio,
                        'num_heads': num_heads,
                        'embed_dim': [embed_dim]*depth,
                        'num_params': float(self.vis_dict[cand]['params']),
                        'score': float(self.vis_dict[cand]['score'])
                    }
                    cand_list.append(info)
                if len(cand_list) > 0:
                    json_dict[str(interval[1])] = cand_list
            print("selected candidates:")
            print(json_dict)
            with open(out_file_name, "w") as fp:
                json.dump(json_dict, fp, indent=2)
            fp.close()



        return self.interval_cands

