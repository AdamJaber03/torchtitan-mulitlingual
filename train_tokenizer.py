import argparse
import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDec
from tokenizers.processors import ByteLevel as ByteLevelPost
from transformers import PreTrainedTokenizerFast

def train_tokenizer(args):
    print("Initializing Byte-Level BPE Tokenizer...")
    
    # 1. Initialize a tokenizer with the BPE model
    tokenizer = Tokenizer(BPE())

    # 2. Setup ByteLevel pre-tokenizer, decoder, and post-processor 
    # (Standard practice for modern LLMs like Llama/GPT architectures)
    tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDec()
    tokenizer.post_processor = ByteLevelPost(trim_offsets=False)

    # 3. Define standard special tokens
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]

    # 4. Initialize the BPE Trainer
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevelPre.alphabet(),
        show_progress=True
    )

    # 5. Create a memory-efficient iterator to stream datasets
    def batch_iterator():
        for ds_path in args.datasets:
            print(f"Streaming dataset: {ds_path}...")
            # Streaming prevents Out-of-Memory (OOM) errors on massive datasets
            dataset = load_dataset("json", data_dir=ds_path, split="train", streaming=True)
            
            for item in dataset:
                text = item.get(args.text_column)
                if text and isinstance(text, str):
                    yield text

    # 6. Train the tokenizer
    print(f"Training tokenizer with vocab size {args.vocab_size}...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # 7. Wrap it in Hugging Face's PreTrainedTokenizerFast
    # This step is crucial for generating the `tokenizer_config.json` file!
    print("Wrapping tokenizer for Hugging Face compatibility...")
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        clean_up_tokenization_spaces=False
    )

    # 8. Save to the specified directory
    os.makedirs(args.save_path, exist_ok=True)
    fast_tokenizer.save_pretrained(args.save_path)
    
    print(f"\n✅ Training Complete! Files saved to: '{args.save_path}'")
    print(f" ├── tokenizer.json (The core BPE merges and vocabulary)")
    print(f" ├── tokenizer_config.json (The HF configuration and special tokens)")
    print(f" └── special_tokens_map.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE Tokenizer for torchtitan")
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        required=True, 
        help="List of Hugging Face dataset paths (e.g., wikitext wikitext-103-v1)"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="train", 
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--text_column", 
        type=str, 
        default="text", 
        help="Name of the text column in the dataset (default: text)"
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=32000, 
        help="Target vocabulary size (default: 32000)"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=True, 
        help="Directory to save the trained tokenizer"
    )
    
    args = parser.parse_args()
    train_tokenizer(args)