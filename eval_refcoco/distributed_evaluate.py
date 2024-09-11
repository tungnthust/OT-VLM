import subprocess
import sys

if __name__ == "__main__":
    model = sys.argv[1]
    dataset = sys.argv[2]
    split = sys.argv[3]
    num_parts = int(sys.argv[4])
    
    input_file = f'/raid/kientdt/shared_drive_cv/VLM/data/RefCOCO/my_{dataset}_{split}.jsonl'
    detection_file = f'/raid/kientdt/shared_drive_cv/VLM/data/RefCOCO/{dataset}/instances.json'
    triplets_file = f'/raid/kientdt/shared_drive_cv/VLM/data/RefCOCO/RefCOCO_json/gpt_{dataset}_{split}.jsonl'

    part_args = [f"{num_parts},{i}" for i in range(num_parts)]
    processes = []

    # Loop through the arguments
    for part in part_args:
        part_id = int(part.split(",")[1])
        # if part_id not in [6, 7]:
        #     continue
        log_file_path = f'/raid/kientdt/shared_drive_cv/VLM/logs/eval_logs_{model}_{dataset}_{split}_part{part_id + 1}of{num_parts}.txt'

        command = [
            "/home/minhnh/python_venv/cv/bin/python",
            "/raid/kientdt/shared_drive_cv/VLM/Kien_code/Zeroshot_REC/eval_refcoco/main.py",
            "--input_file", input_file,
            "--image_root", "/raid/kientdt/shared_drive_cv/VLM/data/RefCOCO/train2014",
            "--method", "matching",
            "--clip_model", "ViT-B/32",
            "--triplets_file", triplets_file,
            "--rule_filter",
            "--lora_path", "/raid/kientdt/shared_drive_cv/VLM/Kien_code/Zeroshot_REC/Outputs/CLIP_finetune/checkpoints/epoch_latest.pt",
            "--detector_file", detection_file,
            "--part", part,
            '--enable_lora'
        ]
        with open(log_file_path, 'w') as outfile:
            process = subprocess.Popen(command, stdout=outfile, stderr=subprocess.STDOUT)
            processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.wait()
