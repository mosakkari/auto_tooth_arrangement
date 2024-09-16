python get_result.py \
	--checkpoint "" \
	--dataroot "" \
	--before_path "" \
	--after_path "" \
    --before_mesh_path "" \
    --after_mesh_path "" \
    --outputroot "" \
	--batch_size 4 --n_classes 40 \
	--n_epoch 700 \
	--name "axis" \
	--weight_decay 0.05 \
	--mask_ratio 0.5 \
	--channels 10 --patch_size 64 \
	--weight 0.5 \
	--depth 12 \
	--heads 12 \
	--lr_milestones "none" --optim "adamw" \
	--encoder_depth 12 \
	--decoder_depth 6 \
	--decoder_dim 512 \
	--decoder_num_heads 16 \
	--num_warmup_steps "2" \
    --point_num 128 \
	--segmented \
	--dim 768 \

# checkpoint: The path to model's checkpoint
# dataroot: The root to data after remesh
# before_path: The root to pointcloud data
# after_path: The root to pointcloud data after 
# before_mesh_path：The root to mesh before treatment
# after_mesh_path: The root to mesh after treatment
# outputroot: root for results
