# raw
# deepspeed --master_port=24999 train_ds.py \
#   --version="PATH_TO_LLaVA" \
#   --dataset_dir='./dataset' \
#   --vision_pretrained="PATH_TO_SAM" \
#   --dataset="sem_seg||refer_seg||vqa||reason_seg" \
#   --sample_rates="9,3,3,1" \
#   --exp_name="lisa-7b"

CUDA_VISIBLE_DEVICES=1 deepspeed --master_port=24999 train_ds.py \
  --version="xinlai/LISA-7B-v1" \
  --dataset_dir='./dataset' \
  --dataset="reason_seg" \
  --sample_rates="1" \
  --exp_name="lisa-7b-5epoch" \
  --epochs 5 \
  --steps_per_epoch 500 \
  --vision_pretrained "./Lisa_tuned_SAM.bin" \
  --explanatory -1


CUDA_VISIBLE_DEVICES=0,1  deepspeed --master_port=24999 train_ds.py \
  --version="xinlai/LISA-7B-v1" \
  --dataset_dir='./dataset' \
  --dataset="reason_seg" \
  --sample_rates="1" \
  --exp_name="lisa-7b-10epoch" \
  --epochs 100 \
  --steps_per_epoch 50 \
  --vision_pretrained "./Lisa_tuned_SAM.bin" \
  --lr 0.0001 \
  --auto_resume \
  --grad_accumulation_steps 10 \
  --batch_size 1 \
  --explanatory -1

# To huggingface format
cd ./runs/lisa-7b-10epoch/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
cd ../../../
CUDA_VISIBLE_DEVICES="1" python merge_lora_weights_and_save_hf_model.py \
  --version="xinlai/LISA-7B-v1" \
  --weight="runs/lisa-7b-10epoch/pytorch_model.bin" \
  --save_path="./LISA_mini7b_1epoch"

    # --weight="runs/lisa-7b-test/pytorch_model.bin" \
