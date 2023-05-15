### eval PreNAS_tiny
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env  \
supernet_train.py \
--gp \
--change_qk \
--relative_position \
--mode retrain \
--dist-eval \
--batch-size 128 \
--eval \
--data-path ../datas/imagenet \
--cfg ./experiments/supernet/tiny.yaml \
--candfile ./interval_cands/tiny.json \
--resume ./output/tiny/checkpoint.pth

### eval PreNAS_small
#python -m torch.distributed.launch \
#--nproc_per_node=8 \
#--use_env  \
#supernet_train.py \
#--gp \
#--change_qk \
#--relative_position \
#--mode retrain \
#--dist-eval \
#--batch-size 128 \
#--eval \
#--data-path ../datas/imagenet \
#--cfg ./experiments/supernet/small.yaml \
#--candfile ./interval_cands/small.json \
#--resume ./output/small/checkpoint.pth

### eval PreNAS_base
#python -m torch.distributed.launch \
#--nproc_per_node=8 \
#--use_env  \
#supernet_train.py \
#--gp \
#--change_qk \
#--relative_position \
#--mode retrain \
#--dist-eval \
#--batch-size 128 \
#--eval \
#--data-path ../datas/imagenet \
#--cfg ./experiments/supernet/base.yaml \
#--candfile ./interval_cands/base.json \
#--resume ./output/base/checkpoint.pth
