from prompt import Prompter
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

class Dataset:
    def __init__(
        self,
        tokenizer,
        dataset_name: str,
        batch_size: int,
        ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.prompter = Prompter()
    
    def load_data(self, dataset_name):
        dataset = load_dataset(self.dataset_name, "vi", split = "train")
        return dataset
    
    def tokenize(self, prompt, max_length = 512, add_eos_token = True):
        result = self.tokenizer(prompt,
                                truncation = True,
                                max_length = max_length,
                                padding = False,
                                return_tensors = None)
        if (   
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < max_length
            and add_eos_token
            ):
            
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["labels"] = result["input_ids"].copy()
        return result 
    
    def generate_and_tokenize_prompt(self, dataset):
        full_prompt = self.prompter.generate_prompt(dataset["instruction"],
                                                    dataset["input"],
                                                    dataset["output"])
        
        tokenized_full_prompt = self.tokenize(full_prompt)
        return tokenized_full_prompt
    
    def dataloader(self):
        dataset = self.load_data(self.dataset_name)
        dataset = dataset.shuffle().select(range(3000))
        dataset = dataset.train_test_split(test_size = 0.05, seed = 42)
        
        train_data = dataset["train"].map(self.generate_and_tokenize_prompt, num_proc = 13)
        valid_data = dataset["test"].map(self.generate_and_tokenize_prompt, num_proc = 13)
        
        train_data = train_data.remove_columns(["instruction", "input", "id", "output"])
        valid_data = valid_data.remove_columns(["instruction", "input", "id", "output"])

        train_data.set_format("torch")
        valid_data.set_format("torch")
        
        train_dataloader = DataLoader(
            train_data,
            batch_size = self.batch_size,
            collate_fn = DataCollatorForSeq2Seq(
                tokenizer = self.tokenizer,
                padding = True,
                return_tensors = "pt",
                ),
            )
        
        valid_dataloader = DataLoader(
            valid_data,
            batch_size = self.batch_size,
            collate_fn = DataCollatorForSeq2Seq(
                tokenizer = self.tokenizer,
                padding = True,
                return_tensors = "pt",
                ),
            )
        return train_dataloader, valid_dataloader
        