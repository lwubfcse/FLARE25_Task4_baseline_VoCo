now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_VoCo
json_file=./jsons/abdomen_organ.json
data_dir=./data/abdomen_organ_seg/
cache_dir=./data/cache

a_min=-175.0
a_max=250.0
roi_x=96
roi_y=96
roi_z=64
out_channels=14

mkdir -p $logdir

torchrun --master_port=24113 main.py \
    --data_dir $data_dir \
    --json_file $json_file\
    --cache_dir $cache_dir \
    --a_min $a_min \
    --a_max $a_max \
    --roi_x $roi_x \
    --roi_y $roi_y \
    --roi_z $roi_z \
    --out_channels $out_channels \
    --logdir $logdir | tee $logdir/$now.txt