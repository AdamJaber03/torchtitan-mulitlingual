import json
import random
import re
import os
from torchtitan.hf_datasets.value_schedualers import SCHEDUALER_REGISTRY
from torchtitan.tools.logging import logger

class WordwiseCodeSwitching:
    """
    On-the-fly n-gram code-switching augmentation for language model training.
    """
    def __init__(self, config: dict):
        self.name = config.get("name", "wordwise_codeswitching")
        self.replace_prob = config.get("prob", 0.3)
        self.dict_paths = config.get("dict_paths", {})
        
        self.dictionaries = {}
        self._load_dictionaries()
        
        # Pre-compile regex patterns for speed
        self.en_pattern = re.compile(r'([a-zA-Z]+)')
        self.ar_pattern = re.compile(r'([\u0600-\u06FF]+)')

    def _load_dictionaries(self):
        print(f"Initializing {self.name} augmentation...")
        for dataset_name, path in self.dict_paths.items():
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self.dictionaries[dataset_name] = json.load(f)
                print(f"✅ Loaded dictionary for {dataset_name} ({len(self.dictionaries[dataset_name])} entries)")
            else:
                print(f"⚠️ Warning: Dictionary path not found for {dataset_name}: {path}")
                # Fallback empty dict so it doesn't crash during training
                self.dictionaries[dataset_name] = {}

    def __call__(self, text: dict, dataset_name: str) -> dict:
        """
        Applies the code-switching augmentation to a given string.
        
        Args:
            text: The original document string.
            dataset_name: The identifier (e.g., "fineweb-edu-ar-en") to select the right dict.
        """
        # print(f"Applying {self.name} to dataset {dataset_name} with replace_prob={self.replace_prob}")
        # print(f"Original text: {text[:100]}...")  # Print the first 100 chars for context
        if dataset_name not in self.dictionaries or self.replace_prob <= 0.0:
            return text
            
        translation_dict = self.dictionaries[dataset_name]
        
        # Infer language logic based on the dataset key suffix
        if dataset_name.endswith("-en"):
            lang = 'en'
            pattern = self.en_pattern
        elif dataset_name.endswith("-ar"):
            lang = 'ar'
            pattern = self.ar_pattern
        else:
            # If the language can't be inferred, return original text safely
            return text

        tokens = pattern.split(text["text"])
        result = []
        
        if len(tokens) > 0:
            result.append(tokens[0])
            
        i = 1
        while i < len(tokens):
            replaced = False
            
            # Try matching 3-grams, 2-grams, 1-grams
            for n in [3, 2, 1]:
                if i + (n - 1) * 2 < len(tokens):
                    words = [tokens[i + j*2] for j in range(n)]
                    key = " ".join(words)
                    
                    if lang == 'en':
                        key = key.lower()
                        
                    if key in translation_dict:
                        if random.random() < self.replace_prob:
                            result.append(translation_dict[key])
                            i += (n - 1) * 2 
                            replaced = True
                            break
                            
            if not replaced:
                result.append(tokens[i])
                
            if i + 1 < len(tokens):
                result.append(tokens[i + 1])
                
            i += 2
        # print(f"Augmented text: {''.join(result)[:100]}...")  # Print the first 100 chars of augmented text for context
        text["text"] = "".join(result)
        return text

def decapitalize(text):
    """
    Decapitalizes the input text by converting it to lowercase and removing any leading or trailing whitespace.

    Args:
        text (str): The input text to be decapitalized.

    Returns:
        str: The decapitalized version of the input text.
    """
    return text.lower().strip()

class DocumentTranslation:

    def __init__(self, config: dict):
        self.name = config.get("name", "document_translation")

    def __call__(self, text: list|tuple|dict, dataset_name: str) -> str:
        if type(text) == dict:
            return text
        shuffled_list = random.sample(text, len(text))
        text["text"] = "\n".join([l["text"] for l in shuffled_list])
        return text

class TextDuplication:
    """
    Augmentation that receives a text string and returns a list 
    with n copies of that string.
    """
    def __init__(self, config: dict):
        self.name = config.get("name", "text_duplication")
        self.n = config.get("n", 2) # Defaults to 2 copies if not specified in config

    def __call__(self, text: dict, dataset_name: str = "") -> list:
        """
        Applies the duplication to a given string.
        
        Args:
            text: The original document string.
            dataset_name: The identifier (kept for signature compatibility).
            
        Returns:
            A list containing n copies of the input text.
        """
        if not isinstance(text, dict) or "text" not in text:
            raise TypeError(f"Expected a dictionary with a 'text' key, but received {type(text).__name__}")
            
        return [{k: v for k, v in text.items()} for _ in range(self.n)]


# class contrastiveMask:
#     """
#     Augmentation that receives a text string and returns a list 
#     with n copies of that string.
#     """
#     def __init__(self, config: dict):
#         self.name = config.get("name", "contrastive_mask")
#         self.n = config.get("n", 2) # Defaults to 2 copies if not specified in config

#     def __call__(self, text: dict, dataset_name: str = "") -> list:
#         """
#         Applies creates a contrastive mask to a given string.
        
#         Args:
#             text: The original document string.
#             dataset_name: The identifier (kept for signature compatibility).
            
#         Returns:
#             A list containing n copies of the input text.
#         """
#         if not isinstance(text, dict) or "text" not in text:
#             raise TypeError(f"Expected a dictionary with a 'text' key, but received {type(text).__name__}")
            
#         return [{k: v for k, v in text.items()}] * self.n

import os
import json
import re
import random
from multiprocessing import Value
from unidecode import unidecode

class WordwiseUnigramCodeSwitching:
    """
    On-the-fly uni-gram code-switching augmentation for language model training.
    """
    def __init__(self, config: dict):
        self.name = config.get("name", "wordwise_codeswitching")
        self.replace_prob = config.get("prob", None)
        self.prob_schedualer = config.get("prob_schedualer", None)
        if self.prob_schedualer is not None:
            assert self.replace_prob is None, "Cannot specify both a fixed replace_prob and a prob_schedualer. Please choose one."
            if self.prob_schedualer["name"] not in SCHEDUALER_REGISTRY:
                raise ValueError(f"Prob schedualer '{self.prob_schedualer['name']}' is not registered. Available: {list(SCHEDUALER_REGISTRY.keys())}")
            self.prob_schedualer = SCHEDUALER_REGISTRY[self.prob_schedualer["name"]](**{k:v for k,v in config.get("prob_schedualer", {}).items() if k != "name"})
            self.replace_prob = self.prob_schedualer(0)  # Initialize with the starting probability
            print(f"Initialized {self.name} with dynamic prob schedualer '{self.prob_schedualer.__class__.__name__}', starting at replace_prob={self.replace_prob}")
        self.replace_prob = Value('d', self.replace_prob)  # For shared memory access if needed
        self.dict_paths = config.get("dict_paths", {})
        self.idx = config.get("idx", None)
        self.pattern = config.get("pattern", None)
        if self.pattern is not None:
            assert self.pattern in ["en", "ar", "ru"]
        self.tokenizer = config.get("tokenizer")
        self.dictionaries = {}
        self._load_dictionaries()
        self.fallback_to_transliteration = config.get("fallback_to_transliteration", False)
        # Pre-compile regex patterns for speed
        self.en_pattern = re.compile(r'([a-zA-Z]+)')
        # self.ar_pattern = re.compile(r'([\u0600-\u06FF]+)')
        self.ar_pattern = re.compile(r'([\u0620-\u065F\u0670-\u06EF\u06FA-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF\u200C\u200D]+)')
        self.ru_pattern = re.compile(r'([\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\u1C80-\u1C8F\u0300-\u036F\u200C\u200D\u00AD]+)')

    def step(self, global_step):
        # logger.info(f"************Stepping {self.name} augmentation at global step {global_step}...*********************")
        if self.prob_schedualer is not None:
            self.replace_prob.value = self.prob_schedualer(global_step)
            # logger.info(f"Updated replace_prob to {self.replace_prob.value} based on schedualer at step {global_step}")

    def _load_dictionaries(self):
        print(f"Initializing {self.name} augmentation...")
        for dataset_name, path in self.dict_paths.items():
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self.dictionaries[dataset_name] = json.load(f)
                print(f"✅ Loaded dictionary for {dataset_name} ({len(self.dictionaries[dataset_name])} entries)")
            else:
                print(f"⚠️ Warning: Dictionary path not found for {dataset_name}: {path}")
                # Fallback empty dict so it doesn't crash during training
                self.dictionaries[dataset_name] = {}

    def __call__(self, text: dict, dataset_name: str) -> dict:
        raw_text = text[self.idx]["text"] if self.idx is not None else text["text"]
        
        do_augment = dataset_name in self.dictionaries and self.replace_prob.value > 0.0
        translation_dict = {}
        pattern = None
        # print(f"***************************wordwise prob for augmentation: {self.replace_prob.value}***********************************")
        
        if do_augment:
            translation_dict = self.dictionaries[dataset_name]
            if self.pattern is not None:
                pattern = {"en": self.en_pattern, "ar": self.ar_pattern, "ru": self.ru_pattern}[self.pattern]
            elif dataset_name.endswith("-en"):
                pattern = self.en_pattern
            elif dataset_name.endswith("-ar"):
                pattern = self.ar_pattern
            elif dataset_name.endswith("-ru"):
                pattern = self.ru_pattern
            else:
                raise ValueError(f"Dataset name '{dataset_name}' does not match expected language suffixes for augmentation. Expected suffixes: '-en', '-ar', '-ru'.")
            if self.fallback_to_transliteration:
                assert pattern == self.ar_pattern or pattern == self.ru_pattern, "Fallback to transliteration is only supported for Arabic or Russian text."

        tokens = re.split(r'(\s+)', raw_text)
        reconstructed_parts = []
        end_idxs = []
        end_idx = 0
        for token in tokens:
            # Keep whitespace structurally intact, but skip it for logic and mask-counting
            if not token.strip():
                reconstructed_parts.append(token)
                end_idx += len(token)
                continue
            
            new_token = token
            if do_augment and pattern:
                def replace_word(match):
                    word = match.group(1)
                    lookup_word = word if word in translation_dict else word.lower()
                    # assert lookup_word in translation_dict, f"Lookup word '{lookup_word}' should either be in the dictionary or not, but got an unexpected case. Original word: '{word}'"
                    if random.random() < self.replace_prob.value:
                        if lookup_word in translation_dict:
                            return translation_dict[lookup_word]
                        if self.fallback_to_transliteration:
                            return unidecode(lookup_word)
                    return word
                
                # Search and replace words within the current token
                new_token = pattern.sub(replace_word, token)
            
            reconstructed_parts.append(new_token)
            
            end_idx += len(new_token)
            end_idxs.append(end_idx-1)
        if end_idxs[-1] != end_idx-1:
            end_idxs.append(end_idx-1)               
        reconstructed_text = "".join(reconstructed_parts)
        # Save the outputs
        if self.idx is not None:
            text[self.idx]["text"] = reconstructed_text
            text[self.idx]["word_sep_idx"] = end_idxs
        else:
            text["text"] = reconstructed_text
            text["word_sep_idx"] = end_idxs

        return text

    # def __call__(self, text: dict, dataset_name: str) -> dict:
    #     """
    #     Applies the code-switching augmentation to a given string.
        
    #     Args:
    #         text: The original document string.
    #         dataset_name: The identifier (e.g., "fineweb-edu-ar-en") to select the right dict.
    #     """
    #     if dataset_name not in self.dictionaries or self.replace_prob <= 0.0:
    #         text["word_mask"] = list(range(len(text["text"].split())))  # If no augmentation, mask is just the original word indices
    #         return text
            
    #     translation_dict = self.dictionaries[dataset_name]
        
    #     # Infer language logic based on the dataset key suffix
    #     if dataset_name.endswith("-en"):
    #         lang = 'en'
    #         pattern = self.en_pattern
    #     elif dataset_name.endswith("-ar"):
    #         lang = 'ar'
    #         pattern = self.ar_pattern
    #     else:
    #         # If the language can't be inferred, return original text safely
    #         text["word_mask"] = list(range(len(text["text"].split())))  # Mask is just the original word indices
    #         return text

    #     if self.idx is not None:
    #         tokens = text[self.idx]["text"]
    #     else:
    #         tokens = text["text"]
    #     # Tokenize by space to align perfectly with the assert
    #     tokens = tokens.split()
    #     final_tokens = []
    #     mask = []
    #     j = 0

    #     for token in tokens:
    #         sub_tokens = pattern.split(token)
            
    #         for i in range(1, len(sub_tokens), 2):
    #             word = sub_tokens[i]
    #             lookup_word = word if word in translation_dict else word.lower()
                
    #             if lookup_word in translation_dict and random.random() < self.replace_prob:
    #                 sub_tokens[i] = translation_dict[lookup_word]
                    
    #         new_token = "".join(sub_tokens)
    #         final_tokens.append(new_token)
    #         num_spaces_in_new_token = len(new_token.split())
    #         mask.extend([j] * num_spaces_in_new_token)
    #         j += 1

    #     if self.idx is not None:
    #         text[self.idx]["text"] = " ".join(final_tokens)
    #         if self.output_word_mask:
    #             text[self.idx]["word_mask"] = mask
    #         assert len(text[self.idx]["text"].split()) == len(mask), f"Mask length does not match the number of words in the text, expected {len(text['text'].split())}, got {len(mask)}"
    #         return text
    #     text["text"] = " ".join(final_tokens)
    #     if self.output_word_mask:
    #         text["word_mask"] = mask
    #     assert len(text["text"].split()) == len(mask), f"Mask length does not match the number of words in the text, expected {len(text['text'].split())}, got {len(mask)}"
    #     return text
class AddPrefix:
    """
    On-the-fly uni-gram code-switching augmentation for language model training.
    """
    def __init__(self, config: dict):
        self.name = config.get("name", "add_prefix")
        self.prefix = config.get("prefix", None)
        self.idx = config.get("idx", None)
        if self.prefix is None:
            raise ValueError(f"[{self.name}] config must include 'prefix'")
        print(f"Initialized {self.name} augmentation with prefix='{self.prefix}'...")

    def __call__(self, text: dict, dataset_name: str) -> dict:
        raw_text = text[self.idx]["text"] if self.idx is not None else text["text"]
        reconstructed_text = self.prefix + raw_text
        # Save the outputs
        if self.idx is not None:
            text[self.idx]["text"] = reconstructed_text
        else:
            text["text"] = reconstructed_text
        return text


class mergeSeperators:
    """
    Augmentation that merges word separators to create longer "words" for the model to process, which can be useful for certain types of code-switching or to encourage the model to learn more holistic representations.
    """
    def __init__(self, config: dict):
        self.name = config.get("name", "merge_seperators")
        self.idx = config.get("idx", None)
        self.n_merge = config.get("n_merge", 2)
        print(f"Initialized {self.name} augmentation with n_merge={self.n_merge}...")

    def __call__(self, text: dict, dataset_name: str) -> dict:
        org_sep = text[self.idx]["word_sep_idx"] if self.idx is not None else text["word_sep_idx"]
        sep = []
        for i in range(self.n_merge-1, len(org_sep), self.n_merge):
            sep.append(org_sep[i])
        if len(sep) == 0 or sep[-1] != org_sep[-1]:
            sep.append(org_sep[-1])  # Ensure the last index is always included as a separator
        # Save the outputs
        if self.idx is not None:
            text[self.idx]["word_sep_idx"] = sep
        else:
            text["word_sep_idx"] = sep
        return text

class uniformMatchSeperators:
    """
    Augmentation that merges word separators to create longer "words" for the model to process, which can be useful for certain types of code-switching or to encourage the model to learn more holistic representations.
    """
    def __init__(self, config: dict):
        self.name = config.get("name", "uniform_match_seperators")
        self.idx = config.get("idx", None)
        assert self.idx is not None and self.idx in [0,1], "to match seperators idx must be 0 or 1"
        print(f"Initialized {self.name} augmentation with for idx: {self.idx}")

    def __call__(self, text: dict, dataset_name: str) -> dict:
        org_sep = text[self.idx]["word_sep_idx"] if self.idx is not None else text["word_sep_idx"]
        num_parts = len(text[1-self.idx]["word_sep_idx"])
        sep = []
        if num_parts == 0:
            text[self.idx]["word_sep_idx"] = sep
            return text
        if num_parts > len(org_sep):
            dup_sep = text[1-self.idx]["word_sep_idx"] if self.idx is not None else text["word_sep_idx"]
            for i in range(len(dup_sep) // len(org_sep) - 1, len(dup_sep)-1, len(dup_sep) // len(org_sep)):
                sep.append(dup_sep[i])
                if len(sep) == len(org_sep) -1:
                    break
            sep.append(dup_sep[-1])
            assert len(sep) == len(org_sep), f"output num seperators should be same as other entry. num_parts: {num_parts}, len(sep): {len(sep)}"
            text[1-self.idx]["word_sep_idx"] = sep
            return text
        if num_parts == len(org_sep):
            return text
        
        for i in range(len(org_sep) // num_parts - 1, len(org_sep)-1, len(org_sep) // num_parts):
            sep.append(org_sep[i])
            if len(sep) == num_parts -1:
                break
        sep.append(org_sep[-1])
        assert len(sep) == num_parts, f"output num seperators should be same as other entry. num_parts: {num_parts}, len(sep): {len(sep)}"
        text[self.idx]["word_sep_idx"] = sep
        return text

AUGMENTATIONS_REGISTRY = {
    "wordwise_codeswitching": WordwiseCodeSwitching,
    "decapitalization": decapitalize,
    "document_translation": DocumentTranslation,
    "text_duplication": TextDuplication,
    "wordwise_unigram_codeswitching": WordwiseUnigramCodeSwitching,
    "add_prefix": AddPrefix,
    "merge_seperators": mergeSeperators,
    "uniform_match_seperators": uniformMatchSeperators,
}
