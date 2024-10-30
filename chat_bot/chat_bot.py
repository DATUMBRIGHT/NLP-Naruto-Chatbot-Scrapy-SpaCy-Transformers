import pandas as pd
import huggingface_hub
import re
import gc
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from transformers import Trainer,TrainingArguments,Pipeline
import torch
from trl import SFTConfig,SFTTrainer
from peft import get_peft_config, get_peft_model, LoraConfig,TaskType
from dotenv import load_dotenv

load_dotenv()

class ChatBotTrainer:

    def __init__(self,
                 data_path,
                 model_path,
                 hf_token = None):
        
        self.data_path = data_path
        self.model_path = model_path
        self.model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
        self.hf_token = hf_token
        self.device = self.load_device()
        
        #load tokenizer
        self.tokenizer = self.load_tokenizer()
        self.tokenizer.pad_token= 'eos'
        self.dataset = self.make_dataset()

      
        if self.hf_token:
            huggingface_hub.login(self.hf_token,new_session=True)
            
            if huggingface_hub.repo_exists(self.model_path):
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        else:
            print('module not found')
            
            self.train()
            self.model = self.load_model()
            
    
    def preprocess_df(self):
        df  = pd.read_csv(self.data_path)

        #remove actions
        pattern = re.compile(r'\(.*?\)')
        df['line'].apply(lambda x: re.sub(pattern," ",x).strip())
        
        #clean text
        df['line'] = (df['line']
        .apply(lambda x: x.replace('\n', " "))  # Replace newline characters with spaces
        .apply(lambda x: re.sub(r'[!@#$%^&*(){}|\\[\]]', '', x))  # Remove special characters
        .apply(lambda x: x.strip())  # Remove leading and trailing whitespace
        .apply(lambda x: x.lower()) ) # Convert to lowercase

        #create words column
        df['words'] = [len(line.split()) for line in df['line']]

        return df 
    
    
    def make_dataset(self):
        df = self.preprocess_df()
        naruto_speeches = df[(df['name'] == 'Naruto') & (df['words']>2)]
        naruto_speech_index = list(naruto_speeches.index)
        naruto_speech_index.remove(0)
        naruto_pre_speech_index = [i-1 for i in naruto_speech_index if i>1] 

        
        #generate prompt for any other person + naturo response
        prompts = []
        system_prompt = """"You are Naruto Chatbot.
        Every question answered should be Naruto tone.
        Use question answer conversation provided """
        for speech,question in zip(naruto_speech_index,naruto_pre_speech_index):
            speech = df.iloc[speech]['line']
            question = df.iloc[question]['line']
            prompt = system_prompt
            prompt += '\n'
            prompt += f'User: {question}'
            prompts.append(prompt)
            
            prompt += '\n'
            prompt += f'Naruto: {speech}' 
            prompts.append(prompt)

        df = pd.DataFrame(prompts,columns= ['Convo'])
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self.tokenize_function,batched=True)

        return dataset 
    
    def tokenize_function(self,examples):
        return self.tokenizer(examples['Convo'],truncation =True,padding= True)

    

    def train(self):

        bnbconfig = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_quant_type = "fp4" ,
                                      bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_quant_storage=torch.bfloat16,)

        base_model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                     quantization_config = bnbconfig,
                                                     torch_dtype = torch.bfloat16,
                                                     device_map = self.device)
        
        tokenizer =self.tokenizer
       
        peft_config = LoraConfig(lora_alpha=8,
                                 lora_dropout=0,
                                 task_type=TaskType.SEQ_2_SEQ_LM,
                                 r=64)
        
        perft_model = get_peft_model(model = base_model,
                              peft_config=peft_config,
                                   )
        
        
        
        training_args = SFTConfig(
                output_dir='./results',
                push_to_hub=True,
                do_predict=True,
                do_train=True,
                learning_rate=2e-5,
                logging_dir='/logs',  
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                weight_decay=0.1,
                num_train_epochs=3,
                lr_scheduler_type='constant',
                max_grad_norm=300,
                save_steps=100,
                optim='adam',
                eval_strategy='epoch',
                fp16=True,
                warmup_ratio=0.3,
                group_by_length=True,
                max_seq_length=512

            )
        
        
        
        
        
        trainer = SFTTrainer(model = perft_model,
                          args=  training_args,
                          tokenizer = tokenizer,
                          peft_config= peft_config,
                          train_dataset = self.dataset,
                          dataset_text_field='Convo')
        

        
        
        trainer.train()
        trainer.save_model('final_ckpt')
        trainer.save_metrics('final_metrics')
        tokenizer.save_pretrained('final_tokenizer')
        perft_model.push_to_hub(self.model_path,token =self.hf_token)
        tokenizer.push_to_hub(self.model_path,token =self.hf_token)

        #flush memory
        del perft_model,tokenizer,base_model
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
              


    def load_tokenizer(self):
        # Initialize tokenizer variable
        tokenizer = None
        
        try:
            if self.hf_token:
                # Log in to Hugging Face Hub with the token
                huggingface_hub.login(self.hf_token)
                
                # Check if the repository exists
                if huggingface_hub.repo_exists(self.model_path):
                    
                    tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    print(f"Tokenizer loaded from the Hugging Face Hub at {self.model_path}")
                else:
                    
                    print(f"Tokenizer not found in the Hugging Face Hub. Loading from local model name: {self.model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            else:
                
                print('No token provided. Loading tokenizer from local model name.')
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        except Exception as e:
            # Print the exception and re-raise it
            print(f"An error occurred while loading the tokenizer: {e}")
            raise

        return tokenizer

    def load_device(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def load_model(self):

        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        pipeline = Pipeline(model = model,tokenizer=tokenizer,
                            device = self.device,
                            torch_dtype=torch.float16,
                            task='text generation')

        return pipeline
    
    def chat(self,message,history):
        
        #prompt + oldchat + new message
        messages = []
        messages.append("Your an expert Naruto Chat. Answer every question with Naruto's tone")
        
        for chat in history:
            messages.append({"role": "user", "content": chat[0]}) #user content
            messages.append({"role": "system", "content": chat[1]}) #system content
              
        messages.append({"role": "user", "content": message})
        
        

        output_message = self.model(messages = messages,max_new_tokens = 256)

        output_message = output_message[0]["generated_text"][-1]

        return output_message



    


        
    


