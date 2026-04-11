import sys
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer

# Import your custom HF wrapper
from torchtitan.models.multilingual_bottleneck.hf import MultilingualBottleneckForCausalLM

@register_model("bilingual_bottleneck")
class BilingualBottleneckLM(HFLM):
    def __init__(self, pretrained, tokenizer_path=None, lang_id=0, **kwargs):
        print(f"Loading custom model for lang_id: {lang_id}")
        
        # 1. Load the custom model directly from the HF directory
        model = MultilingualBottleneckForCausalLM.from_pretrained(pretrained)
        
        # 2. Load the tokenizer 
        tok_path = tokenizer_path if tokenizer_path else pretrained
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        
        # 3. Initialize the parent HFLM class
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            backend="causal",
            **kwargs
        )
        
        # 4. Setup the lang_id tensor for the forward passes
        self.lang_id_val = int(lang_id)
        self.lang_id_tensor = torch.tensor([self.lang_id_val], device=self.device)

    def _model_call(self, inps, attn_mask=None, labels=None):
        """Overrides the loglikelihood forward pass to inject lang_id"""
        with torch.no_grad():
            return self.model(inps, attention_mask=attn_mask, lang_id=self.lang_id_tensor).logits

    def _model_generate(self, context, max_length, **kwargs):
        """Overrides the generation pass to inject lang_id"""
        return self.model.generate(
            context, 
            max_length=max_length, 
            lang_id=self.lang_id_tensor, 
            **kwargs
        )

if __name__ == "__main__":
    # Hand execution over to the standard lm_eval CLI engine!
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()