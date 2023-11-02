import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from dataset import Dataset
from tqdm.auto import tqdm
import math


class Trainer:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        num_epochs: int,
        batch_size: int = 1,
        logging_step: int = 50,
        use_peft: bool = False,
        use_4bit: bool = False,
        ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_epochs = num_epochs
        self.logging_step = logging_step
        self.batch_size = batch_size
        self.use_peft = use_peft
        self.use_4bit = use_4bit
        self.device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
        
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
            
        return model
    
    def train(self):
        tokenizer = self.load_tokenizer(self.model_name)
        model = self.load_model(self.model_name)
        
        train_dataloader, valid_dataloader = Dataset(
            tokenizer = tokenizer,
            dataset_name = self.dataset_name,
            batch_size = self.batch_size,
            ).dataloader()
        

        if self.use_peft == True:
            lr = 3e-4
        else:
            lr = 5e-5
        num_training_steps = self.num_epochs * len(train_dataloader)
        gradient_accumulation_steps = 4
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer = optimizer,
            num_warmup_steps = 100,
            num_training_steps = num_training_steps,
            )
        progress_bar = tqdm(range(num_training_steps))
        
        def eval(dataset):
            model.eval()
            total_loss = 0.0
            for batch in dataset:
                batch = {k:v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
            
            return total_loss / len(dataset)
        
        
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch = {k:v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                
                train_loss += loss.item()
                
                loss /= gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                lr_scheduler.step()
                progress_bar.update(1)
                
                if (step + 1) % self.logging_step == 0:
                    avg_train_loss = train_loss / (step + 1)
                    print(f'Epoch: {epoch + 1} -- step: {step + 1} -- avg_train_loss: {avg_train_loss} -- avg_train_ppl: {math.exp(avg_train_loss)}')
                
            print("Evaluating..............................")  
            avg_train_loss = train_loss / len(train_dataloader)
            avg_eval_loss = eval(valid_dataloader)
            print(f'Epoch: {epoch + 1} -- avg_train_loss: {avg_train_loss} -- avg_val_loss: {avg_eval_loss} -- avg_train_ppl: {math.exp(avg_train_loss)} -- avg_val_ppl: {math.exp(avg_eval_loss)}')
            print("================================================ End of epoch {} ================================================".format(epoch + 1))
                
            
            print("Saving..........")
            torch.save(model.state_dict(), "{}.checkpoint".format(self.model_name.split("/")[1]))
            print("****************** Save successfully ******************")
        
