python tbot_pretrain.py --batch_size 32 --n_epochs_pretrain 20
python tbot_finetune.py --is_finetune 1 --pretrained_model ./saved_models/etth1/masked_tbot/based_model/tbot_pretrained_cw512_patch12_stride12_epochs-pretrain100_mask0.2_model1.pth
