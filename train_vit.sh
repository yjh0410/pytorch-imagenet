python train.py --arch vit_256_6 \
                --batch-size 256 \
                --max_epoch 200 \
                --optimizer adamw \
                --lr 0.001 \
                --lr_schedule cos \
                --data_root /mnt/share/ssd2/dataset/imagenet/
