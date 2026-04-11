import os
import yaml
import random

rate_list = [0, 20, 100, 1000]
base_dir = "/home/adamga/fictional_entity_data/gemini_seeds"
output_dir = "/home/adamga/lm-evaluation-harness/custom_evals/gemini_seeds_fictive_entity_eval_suite"

# Create the directory for the lm-eval tasks
os.makedirs(output_dir, exist_ok=True)

all_task_names = []

file_order_shuffler = random.Random(43)
# file_order = []
# for start in range(0, 1280, 32):
#     chunk = list(range(start, start + 32))
#     file_order_shuffler.shuffle(chunk)
#     file_order.extend(chunk)
file_order = list(range(2080))  # range(2080)
file_order_shuffler.shuffle(file_order)

for group_idx in range(len(rate_list)**2):
    # Calculate k (English) and t (Arabic) based on your math
    k = rate_list[group_idx % len(rate_list)]
    t = rate_list[group_idx // len(rate_list)]
    
    # Base name for this specific injection combo
    combo_base_name = f"en_{k}_ar_{t}"
    
    # Calculate interleaved file indices: e.g., for group 0 -> 0, 16, 32, 48, 64
    file_indices = [file_order[group_idx + ((len(rate_list)**2) * i)] for i in range(2080 // (len(rate_list)**2))] # range(52) range(52,104) range(104,156)
    # file_indices = [group_idx % 8 + (8 * i) for i in range(10)]  # Ensure we don't go beyond 159
    
    # Generate a task for both English and Arabic
    for lang in ["en", "ar"]:
        # e.g., gemini_fictive_en_5_ar_5_en
        task_name = f"gemini_fictive_{combo_base_name}_{lang}"
        all_task_names.append(task_name)
        
        # Point to the correct language JSONL
        # data_files = [f"{base_dir}/{idx}/mcq_{lang}.jsonl" for idx in file_indices]
        data_files = [f"{base_dir}/{idx}/mcq_en.jsonl" for idx in file_indices]
        
        config = {
            "task": task_name,
            "dataset_path": "json",
            "dataset_kwargs": {
                "data_files": {
                    "test": data_files
                }
            },
            "output_type": "multiple_choice",
            "training_split": None,
            "validation_split": None,
            "test_split": "test",
            # "doc_to_text": "Question: {{question}}\nAnswer:",
            "doc_to_text": "{{question}} ",
            "doc_to_target": "answer_index",
            "doc_to_choice": "{{choices}}",
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
                {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True}
            ]
        }
        
        # Save individual task YAML
        with open(os.path.join(output_dir, f"{task_name}.yaml"), "w") as f:
            yaml.dump(config, f, sort_keys=False)

# Create the Group YAML to run all 32 tasks at once
group_config = {
    "group": "fictive_entity_all_rates",
    "task": all_task_names
}

with open(os.path.join(output_dir, "_group.yaml"), "w") as f:
    yaml.dump(group_config, f, sort_keys=False)

print(f"Successfully generated 32 tasks and 1 group config in ./{output_dir}/")