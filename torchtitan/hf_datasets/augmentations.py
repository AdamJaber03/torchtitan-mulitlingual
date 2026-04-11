import json
import random
import re
import os

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

    def __call__(self, text: str, dataset_name: str) -> str:
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

        tokens = pattern.split(text)
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
        return "".join(result)

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

    def __call__(self, text: list|tuple|str, dataset_name: str) -> str:
        if type(text) == str:
            return text
        shuffled_list = random.sample(text, len(text))
        return "\n".join(shuffled_list)

AUGMENTATIONS_REGISTRY = {
    "wordwise_codeswitching": WordwiseCodeSwitching,
    "decapitalization": decapitalize,
    "document_translation": DocumentTranslation,
}
