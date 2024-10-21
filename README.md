# Teach Multimodal LLMs to Comprehend Electrocardiographic Images
The code, data, and models for "Teach Multimodal LLMs to Comprehend Electrocardiographic Images".

## Dataset and Model
#### Training data: [ECGInstruct](https://figshare.com/s/c95aea10d8364fd9170c)
#### Model: [PULSE-7B](https://figshare.com/s/881c755905d9587ce1ce)
#### Evaluation data: [ECGBench](https://figshare.com/s/54aa9d8e638843518306) 

## Installation

Clone the repository and create the environment:

```shell
cd LLaVA

conda create -n pulse-llava python=3.10 -y

conda activate pulse-llava

pip install -e ".[train]"

pip install flash-attn --no-build-isolation
```

## Training

PULSE is trained based on `llava-v1.6-vicuna-7b`, and we have modified the LLaVA code to support the training of `llava-v1.6`.

Before training, ensure that the ECG image storage path matches the path in your training set.

```python
from datasets import load_dataset
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set HF_DATASETS_CACHE environment variable
# os.environ["HF_DATASETS_CACHE"] = "set your hf cache"

# Set the image root
IMAGE_ROOT = "image root"

# Load the dataset
repo_id = ""
dataset = load_dataset(repo_id, split="training", streaming=False)
print(dataset)

# Function to process and save an image
def process_and_save(item):
    image_name = item["image_path"]
    image = item["image"]
    save_path = f"{IMAGE_ROOT}/{image_name}"
    image.save(save_path)

# Use ThreadPoolExecutor for multithreading
num_threads = os.cpu_count()  # Or specify the number of threads you want to use
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_and_save, item) for item in dataset]
    # Use tqdm to show progress
    for future in tqdm(as_completed(futures), total=len(futures)):
        future.result()
```

After preparing the image files, pass the `IMAGE_ROOT` to `image_folder` in `LLaVA/scripts/PULSE_training/finetune_pulse.sh`, and set `data_path` (path to the dataset) and `output_dir` (checkpoint save directory). Then you can start the training process.

The training parameters for PULSE are as follows:

| Global Batch Size | Epoch | Learning Rate | Max Length | LR Scheduler | Warmup Ratio | Zero Stage |
|-------------------|-------|---------------|------------|--------------|--------------|------------|
| 128               | 3     | 2e-5          | 4096       | cosine       | 0.03         | 2          |

Training PULSE for 3 epochs on 32 H100 GPUs took around 10 hours. Since learning to comprehend ECG images is challenging, we recommend training for more epochs to help the model gradually learn how to interpret ECG images.

## Evaluation

After training PULSE, we evaluated the model on 9 datasets from ECGBench. All text data is provided in the `/data` folder.

### 1. Preprocess the images

Before evaluation, download and unzip all test images into `data/ECGBench/images`.

The final directory structure should be:

```
├── ECGBench
    └── images
          └── ptb-xl
          └── cpsc
          └── csn
          └── g12
          └── code15
          └── mmmu-ecg
          └── ecg-arena
```

### 2. Configure inference scripts

Set `SAVE_DIR` and `CKPT_DIR` in `evaluation/pulse/bench_ecgbench.sh` and `evaluation/pulse/bench_ecgarena.sh` to the locations for saving the model's inference results and model weights.

### 3. Run inference

```shell
cd evaluation/pulse/
# ptb-xl test
bash bench_ecgbench.sh -m pulse -d ptb-test

# ptb report generation
bash bench_ecgbench.sh -m pulse -d ptb-test-report

# code15 test
bash bench_ecgbench.sh -m pulse -d code15-test

# mmmu ecg
bash bench_ecgbench.sh -m pulse -d mmmu-ecg

# cpsc test
bash bench_ecgbench.sh -m pulse -d cpsc-test

# g12 test
bash bench_ecgbench.sh -m pulse -d g12-test-no-cot

# csn test
bash bench_ecgbench.sh -m pulse -d csn-test-no-cot

# ecgqa test
bash bench_ecgbench.sh -m pulse -d ecgqa-test

# ecg arena multi-turn
bash bench_ecgarena.sh -m pulse -d arena
```

- `-m`: Model name
- `-d`: Evaluation task name

### 4. Calculate scores

To automatically compute the scores for tasks such as `ptb-test`, `code15-test`, `mmmu-ecg`, `cpsc-test`, `g12-test-no-cot`, `csn-test-no-cot`, and `ecgqa-test`, run the following command:

```python
python evaluate_ecgbench.py --input_dir "/path/to/eval_outputs/"
```

For LLM-as-Judge tasks, including `arena` and `ptb-test-report`, configure `eva_arena.py` and `eval_report.py` with OpenAI settings and the model's inference result paths, then run the evaluation:

```python
python eval_report.py

python eval_arena.py
```
