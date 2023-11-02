from prompt import Prompter
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from peft import (
    LoraConfig,
    get_peft_model,
)

class Inference:
    def __init__(
        self,
        model_name: str,
        checkpoint: str,
        use_peft: bool = False,
        use_4bit: bool = False
        ):
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.prompter = Prompter()
        self.use_peft = use_peft
        self.use_4bit = use_4bit
        self.device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")
        self.model = self.load_model(self.model_name)
        self.tokenizer = self.load_tokenizer(self.model_name)
        
    def load_tokenizer(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        architecture = config.architectures[0]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if "Llama" in architecture:
            tokenizer.add_special_tokens(
                {
                    "eos_token": "</s>",
                    "bos_token": "</s>",
                    "unk_token": "</s>",
                }
            )
            tokenizer.pad_token_id = 0
        return tokenizer 
    
    def load_model(self, model_name):
        if self.use_4bit == True:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_compute_dtype = torch.float16,
                )
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = bnb_config, device_map = "auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(self.device)
        
        if self.use_peft == True:
            lora_config = LoraConfig(
                r = 8,
                lora_alpha = 32,
                lora_dropout = 0.05,
                bias = "none",
                task_type = "CAUSAL_LM",
                )
            model = get_peft_model(model, lora_config)
        
        checkpoint = torch.load(self.checkpoint)
        model.load_state_dict(checkpoint)
        return model
    
    def get_answer(self, instruction: str, input :str = None):
        prompt = self.prompter.generate_prompt(
            instruction = instruction,
            input = input,
            )
        inputs = self.tokenizer(prompt, return_tensors = "pt")
        inputs = {k:v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens = 1024,
            no_repeat_ngram_size = 3,
            num_beams = 3,
            top_k = 40,
            top_p = 128,
            bos_token_id = self.tokenizer.bos_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            pad_token_id = self.tokenizer.pad_token_id,
            do_sample = True,
            )
        text = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
        response = self.prompter.get_response(text)
        return response
        
