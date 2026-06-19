import json
import random
import re
import os
from torchtitan.hf_datasets.value_schedualers import SCHEDUALER_REGISTRY

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
            
        return [{k: v for k, v in text.items()}] * self.n


import os
import json
import re
import random
from multiprocessing import Value

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
        self.output_word_mask = config.get("output_word_mask", False)
        self.idx = config.get("idx", None)
        self.tokenizer = config.get("tokenizer")
        self.tokenizer_aware = config.get("tokenizer_aware", False)
        self.dictionaries = {}
        self._load_dictionaries()

        # Pre-compile regex patterns for speed
        self.en_pattern = re.compile(r'([a-zA-Z]+)')
        # self.ar_pattern = re.compile(r'([\u0600-\u06FF]+)')
        self.ar_pattern = re.compile(r'([\u0620-\u065F\u0670-\u06EF\u06FA-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF\u200C\u200D]+)')

    def step(self, global_step):
        # print(f"************Stepping {self.name} augmentation at global step {global_step}...*********************")
        if self.prob_schedualer is not None:
            self.replace_prob.value = self.prob_schedualer(global_step)
            # print(f"Updated replace_prob to {self.replace_prob.value} based on schedualer at step {global_step}")

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
            if dataset_name.endswith("-en"):
                pattern = self.en_pattern
            elif dataset_name.endswith("-ar"):
                pattern = self.ar_pattern
            else:
                do_augment = False

        if not self.tokenizer_aware:
            # --- TOKENIZER UNAWARE PATH ---
            # Split text while capturing exact whitespace spaces/newlines in the array
            tokens = re.split(r'(\s+)', raw_text)
            reconstructed_parts = []
            mask = []
            
            orig_idx = 0
            for token in tokens:
                # Keep whitespace structurally intact, but skip it for logic and mask-counting
                if not token.strip():
                    reconstructed_parts.append(token)
                    continue
                
                new_token = token
                if do_augment and pattern:
                    def replace_word(match):
                        word = match.group(1)
                        lookup_word = word if word in translation_dict else word.lower()
                        # assert lookup_word in translation_dict, f"Lookup word '{lookup_word}' should either be in the dictionary or not, but got an unexpected case. Original word: '{word}'"
                        if lookup_word in translation_dict and random.random() < self.replace_prob.value:
                            return translation_dict[lookup_word]
                        return word
                    
                    # Search and replace words within the current token
                    new_token = pattern.sub(replace_word, token)
                
                reconstructed_parts.append(new_token)
                
                # The mask corresponds exactly to the number of space-separated words the translation yielded
                num_output_words = len(new_token.split())
                if num_output_words == 0:
                    num_output_words = 1  # Fallback for punctuation-only tokens
                    
                mask.extend([orig_idx] * num_output_words)
                orig_idx += 1
            
            reconstructed_text = "".join(reconstructed_parts)
            # print(f"Original text: {raw_text[:100]}...")  # Print the first 100 chars for context
            # print(f"Augmented text: {reconstructed_text[:100]}...")  #

        else:
            # --- TOKENIZER AWARE PATH ---
            pre_tokenized = self.tokenizer.tokenizer.pre_tokenizer.pre_tokenize_str(raw_text)
            
            reconstructed_text = ""
            chunk_intervals = []
            j = 0

            for chunk, (start, end) in pre_tokenized:
                # THE TRICK: Slicing the original string completely bypasses byte-level mojibake 
                # and perfectly preserves native spaces. No need to touch 'Ġ' at all!
                real_chunk = raw_text[start:end]
                
                if do_augment and pattern:
                    sub_tokens = pattern.split(real_chunk)
                    for i in range(1, len(sub_tokens), 2):
                        word = sub_tokens[i]
                        lookup_word = word if word in translation_dict else word.lower()
                        
                        if lookup_word in translation_dict and random.random() < self.replace_prob.value:
                            sub_tokens[i] = translation_dict[lookup_word]
                    new_chunk = "".join(sub_tokens)
                else:
                    new_chunk = real_chunk
                    
                # Record exactly where this chunk lives in the final string based on character index
                start_char = len(reconstructed_text)
                reconstructed_text += new_chunk
                end_char = len(reconstructed_text)
                
                chunk_intervals.append((start_char, end_char, j))
                j += 1

            # --- THE ALIGNMENT FIX ---
            # Encode the FULL reconstructed string once, exactly as it will be down the pipeline.
            encoding = self.tokenizer.tokenizer.encode(reconstructed_text, add_special_tokens=False)
            
            mask = []
            # Use the Hugging Face character offsets to map every subword back to its original chunk
            for sub_start, sub_end in encoding.offsets:
                found_j = None
                for start_char, end_char, original_j in chunk_intervals:
                    # If the subword starts inside this chunk's boundaries, it belongs to it!
                    if start_char <= sub_start < end_char:
                        found_j = original_j
                        break
                
                # Edge case safety fallback
                if found_j is None:
                    found_j = chunk_intervals[-1][2] if chunk_intervals else 0
                    
                mask.append(found_j)

            # This assert is now mathematically guaranteed to pass 100% of the time.
            assert len(mask) == len(encoding.ids), f"Fatal Alignment: Mask length {len(mask)} != Subwords {len(encoding.ids)}"

        # Save the outputs
        if self.idx is not None:
            text[self.idx]["text"] = reconstructed_text
            if self.output_word_mask:
                text[self.idx]["word_mask"] = mask
        else:
            text["text"] = reconstructed_text
            if self.output_word_mask:
                text["word_mask"] = mask

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

class SyntheticSentenceLanguageMixing:
    """
    Builds artificial en1/en2 documents before tokenization and records the
    character spans that should be shifted into the second vocabulary copy.

    Supported modes:
    - sentence_wise_code_switching: each sentence independently chooses en1/en2.
    - sentence_parallel_doc_order: each sentence appears twice, with the
      en1/en2 order sampled once per document.
    - sentence_parallel_sentence_order: each sentence appears twice, with the
      en1/en2 order sampled independently per sentence.
    """

    def __init__(self, config: dict):
        self.name = config.get("name", "synthetic_sentence_language_mixing")
        self.mode = config.get("mode", "sentence_wise_code_switching")
        self.lang2_prob = config.get("lang2_prob", 0.5)
        self.document_prob = config.get("document_prob", 1.0)
        self.separator = config.get("separator", " ")
        self._punkt_tokenizer = self._load_punkt_tokenizer()

        valid_modes = {
            "sentence_wise_code_switching",
            "sentence_parallel_doc_order",
            "sentence_parallel_sentence_order",
        }
        if self.mode not in valid_modes:
            raise ValueError(
                f"[{self.name}] unsupported mode '{self.mode}'. "
                f"Available modes: {sorted(valid_modes)}"
            )

    def _load_punkt_tokenizer(self):
        try:
            import nltk.data

            return nltk.data.load("tokenizers/punkt/english.pickle")
        except (ImportError, LookupError) as exc:
            raise RuntimeError(
                f"[{self.name}] requires the nltk package and English Punkt "
                "data. Install nltk in the training environment and run: "
                "python -m nltk.downloader punkt punkt_tab."
            ) from exc

    def __call__(self, text: dict, dataset_name: str) -> dict:
        if not isinstance(text, dict) or "text" not in text:
            raise TypeError(
                f"[{self.name}] expected a dictionary with a 'text' key, "
                f"got {type(text).__name__}"
            )

        if self.document_prob < 1.0 and random.random() >= self.document_prob:
            return text

        raw_text = text["text"]
        if not raw_text:
            text["language_spans"] = []
            return text

        if self.mode == "sentence_wise_code_switching":
            text["text"], text["language_spans"] = self._codeswitch(raw_text)
        else:
            text["text"], text["language_spans"] = self._parallel(raw_text)
        return text

    def _sentence_chunks(self, raw_text: str) -> list[str]:
        spans = list(self._punkt_tokenizer.span_tokenize(raw_text))
        if not spans:
            return [raw_text]

        chunks = []
        cursor = 0
        for start, end in spans:
            if start > cursor:
                chunks.append(raw_text[cursor:start])
            chunks.append(raw_text[start:end])
            cursor = end
        if cursor < len(raw_text):
            chunks.append(raw_text[cursor:])
        return chunks

    def _append_with_lang(
        self,
        parts: list[str],
        spans: list[tuple[int, int, int]],
        segment: str,
        lang: int,
    ) -> None:
        if not segment:
            return

        start = sum(len(part) for part in parts)
        parts.append(segment)
        end = start + len(segment)
        spans.append((start, end, lang))

    def _split_outer_space(self, chunk: str) -> tuple[str, str, str]:
        left_len = len(chunk) - len(chunk.lstrip())
        right_len = len(chunk) - len(chunk.rstrip())
        if left_len + right_len >= len(chunk):
            return chunk, "", ""

        leading = chunk[:left_len]
        trailing = chunk[len(chunk) - right_len :] if right_len else ""
        core = chunk[left_len : len(chunk) - right_len if right_len else len(chunk)]
        return leading, core, trailing

    def _codeswitch(self, raw_text: str) -> tuple[str, list[tuple[int, int, int]]]:
        parts = []
        spans = []
        for chunk in self._sentence_chunks(raw_text):
            leading, core, trailing = self._split_outer_space(chunk)
            if not core:
                parts.append(chunk)
                continue

            lang = 2 if random.random() < self.lang2_prob else 1
            parts.append(leading)
            self._append_with_lang(parts, spans, core, lang)
            parts.append(trailing)
        return "".join(parts), spans

    def _parallel(self, raw_text: str) -> tuple[str, list[tuple[int, int, int]]]:
        parts = []
        spans = []
        doc_order = (1, 2) if random.random() < 0.5 else (2, 1)

        for chunk in self._sentence_chunks(raw_text):
            leading, core, trailing = self._split_outer_space(chunk)
            if not core:
                parts.append(chunk)
                continue

            order = doc_order
            if self.mode == "sentence_parallel_sentence_order":
                order = (1, 2) if random.random() < 0.5 else (2, 1)

            parts.append(leading)
            self._append_with_lang(parts, spans, core, order[0])
            parts.append(self.separator)
            self._append_with_lang(parts, spans, core, order[1])
            parts.append(trailing)

        return "".join(parts), spans

AUGMENTATIONS_REGISTRY = {
    "wordwise_codeswitching": WordwiseCodeSwitching,
    "decapitalization": decapitalize,
    "document_translation": DocumentTranslation,
    "text_duplication": TextDuplication,
    "wordwise_unigram_codeswitching": WordwiseUnigramCodeSwitching,
    "synthetic_sentence_language_mixing": SyntheticSentenceLanguageMixing,
}
