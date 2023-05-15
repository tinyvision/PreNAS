### for tiny search space
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
two_step_search.py \
--gp \
--change_qk \
--relative_position \
--dist-eval \
--batch-size 64 \
--data-free \
--score-method left_super_taylor6 \
--block-score-method-for-head balance_taylor6_max_dim \
--block-score-method-for-mlp balance_taylor6_max_dim \
--cand-per-interval 1 \
--param-interval 1.0 \
--min_param_limits 5 \
--param_limits 12 \
--data-path ../datas/imagenet \
--cfg ./experiments/supernet/supernet-T.yaml \
--interval-cands-output ./interval_cands/tiny.json

python candidates_to_choices.py ./interval_cands/tiny.json ./experiments/supernet/tiny.yaml

### for small search space
#python -m torch.distributed.launch \
#--nproc_per_node=8 \
#--use_env \
#two_step_search.py \
#--gp \
#--change_qk \
#--relative_position \
#--dist-eval \
#--batch-size 64 \
#--data-free \
#--score-method left_super_taylor6 \
#--block-score-method-for-head balance_taylor6_max_dim \
#--block-score-method-for-mlp balance_taylor6_max_dim \
#--cand-per-interval 1 \
#--param-interval 5.0 \
#--min_param_limits 13 \
#--param_limits 33 \
#--data-path ../datas/imagenet \
#--cfg ./experiments/supernet/supernet-S.yaml \
#--interval-cands-output ./interval_cands/small.json
#
#python candidates_to_choices.py ./interval_cands/small.json ./experiments/supernet/small.yaml

### for base search space
#python -m torch.distributed.launch \
#--nproc_per_node=8 \
#--use_env two_step_search.py \
#--gp \
#--change_qk \
#--relative_position \
#--dist-eval \
#--batch-size 64 \
#--data-free \
#--score-method left_super_taylor6 \
#--block-score-method-for-head balance_taylor6_max_dim \
#--block-score-method-for-mlp balance_taylor6_max_dim \
#--cand-per-interval 1 \
#--param-interval 12.0 \
#--min_param_limits 30 \
#--param_limits 70 \
#--data-path ../datas/imagenet \
#--cfg ./experiments/supernet/supernet-B.yaml \
#--interval-cands-output ./interval_cands/base.json
#
#python candidates_to_choices.py ./interval_cands/base.json ./experiments/supernet/base.yaml