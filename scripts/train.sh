python train.py train \
	--checkpoint "" \
	--encoder_checkpoint "" \
	--dataroot "" \
	--before_path "" \
	--after_path "" \
	--paramroot "" \
	--batch_size 16 --n_classes 40 \
	--n_epoch 700 \
	--name "tmp" \
	--weight_decay 0.05 \
	--mask_ratio 0.5 \
	--channels 10 --patch_size 64 \
	--lr 1e-4 \
	--weight 0.5 \
	--depth 12 \
	--heads 12 \
	--lr_milestones "none" --optim "adamw" \
	--encoder_depth 12 \
	--decoder_depth 6 \
	--decoder_dim 512 \
	--decoder_num_heads 16 \
	--num_warmup_steps "2" \
	--dim 768 \
	--point_num 512 \
	--saveroot "" \

# checkpoint(optional): The path to model's checkpoint
# encoder_checkpoint: The path to Meshmae's checkpoint
# dataroot: The root to data after remesh
# before_path: The root to pointcloud data
# after_path: The root to pointcloud data after
# paramroot: The root to gt rotation matrix
# saveroot: output root for model checkpoint and log file