from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GemmaModel():
    def __init__(self) -> None:
        llm_model = "google/gemma-3-1b-it"
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model, device_map="auto")

    def get_model_size(self):
        try:

            llm_size = self.llm.config.text_config.hidden_size

            if llm_size > 0:
                return llm_size
            raise Exception("NO LLM FOUND")
        except:
            llm_size = self.llm.config.hidden_size
            if llm_size > 0:
                return llm_size
            raise Exception("NO LLM FOUND")

    
    def get_tokens(self, text: str):
        in_token = self.tokenizer(text, return_tensors="pt").to(self.llm.device)
        in_token_ids = in_token['input_ids'].to(self.llm.device)
        in_embed = self.llm.get_input_embeddings()
        with torch.no_grad():
            Hq = in_embed(in_token_ids)
        return Hq
    
    def generate(self, target : torch.Tensor):
        outputs = self.llm.generate(inputs_embeds=target, max_new_tokens=100, do_sample=True, temperature=0.7)
        response = self.tokenizer.decode(outputs[0], skip_spacial_tokens=True)
        print("answer: ", response)
