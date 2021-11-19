python train.py --arch vit_b \
                --batch-size 256 \
                --max_epoch 300 \
                --optimizer adamw \
                --lr 0.001 \
                --lr_schedule cos \
                --data_root /mnt/share/ssd2/dataset/imagenet/
