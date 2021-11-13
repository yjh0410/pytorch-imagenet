# ConvMixer_384_32
python train.py --arch convmixer_384_32 \
                --batch-size 256 \
                --max_epoch 90 \
                --lr_epoch 30 60 \
                --data_root /mnt/share/ssd2/dataset/imagenet/

# # ConvMixer_768_32
# python train.py --arch convmixer_768_32 \
#                 --batch-size 256 \
#                 --max_epoch 90 \
#                 --lr_epoch 30 60 \
#                 --data_root /mnt/share/ssd2/dataset/imagenet/

# # ConvMixer_1024_20
# python train.py --arch convmixer_1024_20 \
#                 --batch-size 256 \
#                 --max_epoch 90 \
#                 --lr_epoch 30 60 \
#                 --data_root /mnt/share/ssd2/dataset/imagenet/

# # ConvMixer_1536_20
# python train.py --arch convmixer_1536_20 \
#                 --batch-size 256 \
#                 --max_epoch 90 \
#                 --lr_epoch 30 60 \
#                 --data_root /mnt/share/ssd2/dataset/imagenet/
