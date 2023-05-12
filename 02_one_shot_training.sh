### train PreNAS_tiny
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
supernet_train.py \
--gp \
--change_qk \
--relative_position \
--mode super \
--dist-eval \
--epochs 500 \
--warmup-epochs 20 \
--batch-size 128 \
--min-lr 1e-7 \
--group-by-dim \
--group-by-depth \
--mixup-mode elem \
--aa rand-n3-m10-mstd0.5-inc1 \
--recount 2 \
--data-path ../datas/imagenet \
--cfg ./experiments/supernet/base.yaml \
--candfile ./interval_cands/base.json \
--output_dir ./output/tiny

### train PreNAS_small
#python -m torch.distributed.launch \
#--nproc_per_node=8 \
#--use_env \
#supernet_train.py \
#--gp \
#--change_qk \
#--relative_position \
#--mode super \
#--dist-eval \
#--epochs 500 \
#--warmup-epochs 20 \
#--batch-size 128 \
#--group-by-dim \
#--group-by-depth \
#--mixup-mode elem \
#--aa v0r-mstd0.5 \
#--data-path ../datas/imagenet \
#--cfg ./experiments/supernet/small.yaml \
#--candfile ./interval_cands/small.json \
#--output_dir ./output/small

### train PreNAS_base
#python -m torch.distributed.launch \
#--nproc_per_node=8 \
#--use_env \
#supernet_train.py \
#--gp \
#--change_qk \
#--relative_position \
#--mode super \
#--dist-eval \
#--epochs 500 \
#--warmup-epochs 20 \
#--batch-size 128 \
#--min-lr 1e-7 \
#--group-by-dim \
#--group-by-depth \
#--mixup-mode elem \
#--aa rand-n3-m10-mstd0.5-inc1 \
#--recount 2 \
#--data-path ../datas/imagenet \
#--cfg  ./experiments/supernet/base.yaml \
#--candfile ./interval_cands/base.json \
#--output_dir ./output/base
