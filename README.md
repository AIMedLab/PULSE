# Teach Multimodal LLMs to Comprehend Electrocardiographic Images
The code, data, and models for "Teach Multimodal LLMs to Comprehend Electrocardiographic Images".

## Dataset and Model
#### 🌐 Project Page: [Page](https://aimedlab.github.io/PULSE/)
#### 📄 Paper: [Arxiv](https://arxiv.org/abs/2410.19008)
#### 🤗 Model: [PULSE-7B](https://huggingface.co/PULSE-ECG/PULSE-7B)
#### 👩‍⚕️ Training data: [ECGInstruct](https://huggingface.co/datasets/PULSE-ECG/ECGInstruct)
#### ⚖️ Evaluation data: [ECGBench](https://huggingface.co/datasets/PULSE-ECG/ECGBench) 
#### 🩺 Demo: [Demo](https://huggingface.co/spaces/aidhlab/PULSE-7B)




## Installation

Clone the repository and create the environment:

```shell
git clone git@github.com:AIMedLab/PULSE.git

cd PULSE/LLaVA

conda create -n pulse-llava python=3.10 -y

conda activate pulse-llava

pip install -e ".[train]"

pip install flash-attn --no-build-isolation
```

## Inference Example

```shell
cd LLaVA
python llava/eval/run_llava.py --model-path "PULSE-ECG/PULSE-7B" --image-file "images/ecg_example.png" --query "What are the main features in this ECG image?" --conv-mode "llava_v1"
```

## Training

PULSE is trained based on `llava-v1.6-vicuna-7b`, and we have modified the LLaVA code to support the training of `llava-v1.6`.

Before training, please download the ECG images and the training set from [link](https://huggingface.co/datasets/PULSE-ECG/ECGInstruct), and ensure that the storage path of the ECG images matches the path specified in the training set.


The full ECG image dataset occupies a large amount of space, so please ensure you have a stable network connection and sufficient storage space. You can use the following script to download ECGInstruct and extract images.

```
huggingface-cli download --resume-download PULSE-ECG/ECGInstruct --local-dir /path/to/local/directory
```

```
source_dir="/path/to/local/directory"  # directory to store shard_*.tar.gz
target_dir="/path/to/target"  # target directory

mkdir -p "$target_dir"

ls "$source_dir"/shard_*.tar.gz | parallel -j 4 tar -xzf {} -C "$target_dir"
```


After preparing the training files, pass `/path/to/local/directory` to `image_folder` in `LLaVA/scripts/PULSE_training/finetune_pulse.sh`, and set `data_path` (path to the dataset) and `output_dir` (checkpoint save directory). Then you can start the training process.

The training parameters for PULSE are as follows:

| Global Batch Size | Epoch | Learning Rate | Max Length | LR Scheduler | Warmup Ratio | Zero Stage |
|-------------------|-------|---------------|------------|--------------|--------------|------------|
| 128               | 3     | 2e-5          | 4096       | cosine       | 0.03         | 2          |

Training PULSE for 3 epochs on 32 H100 GPUs took around 10 hours. Since learning to comprehend ECG images is challenging, we recommend training for more epochs to help the model gradually learn how to interpret ECG images.

## Evaluation

After training PULSE, we evaluated the model on 9 datasets from ECGBench. All text data is provided in the `/data` folder.

### 1. Preprepare the ECGBench data

Before evaluation, download and process all ECGBench data into the required format, assuming we have created a `data/ECGBench` directory in the main project folder to store the processed ECGBench data. 
Use the following code to download and process the data from [Hugging Face](https://huggingface.co/datasets/PULSE-ECG/ECGBench).

<details>
  <summary>Click to expand the code</summary>

  ```python
  from datasets import load_dataset
  import os
  import json
  from tqdm import tqdm
  from concurrent.futures import ThreadPoolExecutor, as_completed

  # Define the root path where images will be saved
  IMAGE_SAVE_DIR = "data/ECGBench/images"
  JSON_SAVE_DIR = "data/ECGBench"

  # Create a list of dataset subsets to process
  subset_names = ['arena', 'code15-test', 'cpsc-test', 'csn-test-no-cot', 'ecgqa-test', 'g12-test-no-cot', 'mmmu-ecg', 'ptb-test', 'ptb-test-report']

  for name in subset_names:
      dataset = load_dataset("PULSE-ECG/ECGBench", name=name, streaming=False)
      
      dataset_items = []

      def process_and_save(idx):
          item = dataset['test'][idx]
          
          image_path = item["image_path"]
          image = item["image"]
          conversations = item["conversations"]

          dataset_items.append({
              "id": item["id"],
              "image": image_path,
              "conversations": conversations
          })

          save_path = os.path.join(IMAGE_SAVE_DIR, image_path)
          os.makedirs(os.path.dirname(save_path), exist_ok=True)
          image.save(save_path)

      with ThreadPoolExecutor(max_workers=8) as executor:
          futures = [executor.submit(process_and_save, idx) for idx in range(len(dataset['test']))]

          for future in tqdm(as_completed(futures), total=len(futures)):
              future.result()

      # After processing all dataset items, save them to a JSON file for evaluation
      json_filename = os.path.join(JSON_SAVE_DIR, f"{name}.json")
      with open(json_filename, "w", encoding='utf-8') as json_file:
          json.dump(dataset_items, json_file, indent=4, ensure_ascii=False)

      print(f"Dataset '{name}' has been processed and saved to {json_filename}.")
 ```
</details>



The final directory structure should be:

```
├── ECGBench
    ├── images
    │     └── ptb-xl
    │     └── cpsc
    │     └── csn
    │     └── g12
    │     └── code15
    │     └── mmmu-ecg
    │     └── ecg-arena
    ├── arena.json
    ├── code15-test.json
    └── ...
    └── ...
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

## Citation
If you find this work helpful, please cite our paper:
```
@article{liu2024teach,
  title={Teach Multimodal LLMs to Comprehend Electrocardiographic Images},
  author={Ruoqi Liu, Yuelin Bai, Xiang Yue, Ping Zhang},
  journal={arXiv preprint arXiv:2410.19008},
  year={2024}
}
```

