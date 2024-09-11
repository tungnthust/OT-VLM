export LD_LIBRARY_PATH=/home/minhnh/python_venv/cv/lib/python3.9/site-packages/nvidia/cublas/lib:/home/minhnh/python_venv/cv/lib/python3.9/site-packages/nvidia/cudnn/lib

/home/minhnh/python_venv/cv/bin/python /raid/kientdt/shared_drive_cv/VLM/Kien_code/Zeroshot_REC/eval_refcoco/main.py --input_file /raid/kientdt/shared_drive_cv/VLM/data/RefCOCO/my_refcoco+_val.jsonl --image_root /raid/kientdt/shared_drive_cv/VLM/data/RefCOCO/train2014 --method matching --clip_model ViT-B/32 --triplets_file /raid/kientdt/shared_drive_cv/VLM/data/RefCOCO/RefCOCO_json/gpt_refcoco+_val.jsonl --rule_filter --lora_path /raid/kientdt/shared_drive_cv/VLM/Kien_code/Zeroshot_REC/Outputs/CLIP_finetune/checkpoints/epoch_20.pt --detector_file /raid/kientdt/shared_drive_cv/VLM/data/RefCOCO/refcoco+/instances.json --part 10,2 --enable_lora


/home/minhnh/python_venv/cv/bin/python /raid/kientdt/shared_drive_cv/VLM/Kien_code/Zeroshot_REC/eval_refcoco/distributed_evaluate.py clip_ot_v1 refcoco+ val 5

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node 2 --master_port 23451 -m VLA_finetune.training.main --name CLIP_finetune --lora 4 --pretrained openai --epochs 20 --warmup 150 --workers 12 --lr 0.000005 --save-frequency 5 --batch-size 128 --model ViT-B/32 --resume /raid/kientdt/shared_drive_cv/VLM/Kien_code/Zeroshot_REC/Outputs/CLIP_finetune/checkpoints/epoch_15.pt