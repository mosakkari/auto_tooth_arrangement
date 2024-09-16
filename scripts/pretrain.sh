python pretrain.py train \
	--dataroot "" \
	--checkpoint "" \
	--batch_size 512 \
	--augment_scale --augment_orient --n_epoch 700 \
	--num_warmup_steps 2 \
	--name "" \
	--weight_decay 0.05 \
	--mask_ratio 0.5 \
	--channels 10 --patch_size 64 \
	--lr 1e-4 \
	--weight 0.5 \
	--depth 12 \
	--n_dropout 2 \
	--heads 12 \
	--lr_milestones "none" --optim "adamw" \
	--encoder_depth 12 \
	--decoder_depth 6 \
	--decoder_dim 512 \
	--decoder_num_heads 16 \
	--saveroot "" \
	--dim 768

# dataroot: The root to data after remesh
# saveroot: output root for model checkpoint and log file