STEPS=2000
LR=0.0002
BETA1=0.5
BETA2=0.99
WEIGHT_INIT=0.02
RELU_SLOPE=0.2
Z_SIZE=10
BATCH_SIZE=128
SUMMARY_DIR="logs"
IMAGE_DIR="flickr_resize"

python scrape_images.py
python convert_images.py
python model.py \
    --num_steps=$STEPS \
    --learning_rate=$LR \
    --beta1=$BETA1 \
    --beta2=$BETA2 \
    --weight_init=$WEIGHT_INIT \
    --relu_slope=$RELU_SLOPE \
    --z_size=$Z_SIZE \
    --batch_size=$BATCH_SIZE \
    --summary_dir=$SUMMARY_DIR \
    --image_dir=$IMAGE_DIR
