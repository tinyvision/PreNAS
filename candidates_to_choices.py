#!/usr/bin/env python
import json
from collections import defaultdict

def candidate_to_choices(candidate_path, topN=float('inf')):
    interval_cands = json.load(open(candidate_path))

    # init
    new_embed_dim = []
    new_mlp_ratio = defaultdict(lambda : defaultdict(list))
    new_num_heads = defaultdict(lambda : defaultdict(list))
    new_depth = defaultdict(list)

    for cand_list in interval_cands.values():
        for i in range(min(topN, len(cand_list))):
            cur_cand = cand_list[i]
            # embed dim
            embed_dim = cur_cand['embed_dim'][0]
            new_embed_dim.append(embed_dim) if embed_dim not in new_embed_dim else None
            # depth
            depth = cur_cand['layer_num']
            new_depth[embed_dim].append(depth) if depth not in new_depth[embed_dim] else None
            # mlp & heads
            for layer_id, (mlp_ratio, num_heads) in enumerate(zip(cur_cand['mlp_ratio'], cur_cand['num_heads'])):
                pt_mlp_ratio = new_mlp_ratio[embed_dim][layer_id]
                pt_mlp_ratio.append(mlp_ratio) if mlp_ratio not in pt_mlp_ratio else None
                pt_num_heads = new_num_heads[embed_dim][layer_id]
                pt_num_heads.append(num_heads) if num_heads not in pt_num_heads else None

    return {'embed_dim': sorted(new_embed_dim),
            'mlp_ratio': {dim: [sorted(ratios[layer]) for layer in sorted(ratios)] for dim, ratios in new_mlp_ratio.items()},
            'num_heads': {dim: [sorted(heads[layer]) for layer in sorted(heads)] for dim, heads in new_num_heads.items()},
            'depth': {dim: sorted(deps) for dim, deps in new_depth.items()},
           }


if __name__ == '__main__':
    import os, sys, yaml

    cand_file = os.path.normpath(sys.argv[1])
    conf_file = os.path.normpath(sys.argv[2])
    if os.path.exists(conf_file):
        print(f'Target file already exists: {conf_file}')
        exit()

    new_choices = candidate_to_choices(cand_file)
    #print(new_choices)
    cfg = dict()
    cfg['SEARCH_SPACE'] = {k.upper(): v for k, v in new_choices.items()}
    max_depth = max({dep for deps in new_choices['depth'].values() for dep in deps})
    max_ratio = max(max(ratio_list) for ratios in new_choices['mlp_ratio'].values() for ratio_list in ratios)
    max_heads = max(max(heads_list) for heads in new_choices['num_heads'].values() for heads_list in heads)
    max_dim = max_heads * 64
    assert max_dim >= max(new_choices['embed_dim'])
    cfg['SUPERNET'] = {'DEPTH': max_depth, 'MLP_RATIO': max_ratio, 'NUM_HEADS': max_heads, 'EMBED_DIM': max_dim}

    yaml.safe_dump(cfg, open(conf_file, 'w'))
    print(f'Saved to: {conf_file}')
