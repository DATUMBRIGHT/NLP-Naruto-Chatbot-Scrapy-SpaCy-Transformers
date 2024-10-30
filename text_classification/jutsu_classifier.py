import numpy as np
import gc
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import huggingface_hub
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from text_classification.training_utils import compute_metrics
from text_classification.custom_trainer import CustomTrainer
from text_classification.training_utils import get_weights
from text_classification.cleaner import Cleaner


class JutsuPredictor(object):

    def __init__(self,
                  model_path,
                  data_path=None,
                  hugging_face_token=None):
        
        self.model_name = "distilbert-base-uncased"
        self.model_path = model_path
        self.data_path = data_path
        self.text_size = 0.2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hugging_face_token = hugging_face_token
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.df = self.load_df(self.data_path)
        self.classes = pd.unique(self.df['jutsu_type'])
        self.model = None
        self.decodes = {k: v for k, v in enumerate(self.classes)}
       
        if self.hugging_face_token is not None:
            huggingface_hub.login(self.hugging_face_token)
            if huggingface_hub.repo_exists(self.model_path):

                #load model
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path,num_labels = len(self.classes))
            else:
                print('model not found')
                # Load and process data
                self.tokenized_train, self.tokenized_test = self.preprocess_df(self.data_path)
                
                self.train()

                self.model = self.load_model()
            

                
                
                
                # Save the model using the class method
                #JutsuPredictor.save_model(self.model, self.tokenizer, self.model_path)

                #Load tokenizer
                #self.tokenizer = self.load_tokenizer()

                #load trained model
                self.model = self.load_model()
        else:
            
            self.model =  AutoModelForSequenceClassification.from_pretrained(self.model_name,num_labels = len(self.classes))
                
        


    def load_df(self,file_path):
       
        df = pd.read_json(file_path)
        df['jutsu_description'] = df['jutsu_title'] + ". " + df['jutsu_description']
       

        df = df[['jutsu_description', 'jutsu_type']]
        df['jutsu_type'] = [item[0] for item in df['jutsu_type']]
        
        return df



        
          
        
    def simplify_jutsu(self, type):
        return type[0]

   

    def preprocess_df(self, file_path):
        df = pd.read_json(file_path)
        df['jutsu_description'] = df['jutsu_title'] + ". " + df['jutsu_description']
        cleaner = Cleaner()
        
        df['jutsu_description'] = df['jutsu_description'].apply(cleaner.clean)

        df = df[['jutsu_description', 'jutsu_type']]
        #df['jutsu_type'] = df['jutsu_type'].apply(self.simplify_jutsu)
        df['jutsu_type'] = [item[0] for item in df['jutsu_type']]
        
        unique_classes = set(df['jutsu_type'])
        
        codes = {v: k for k, v in enumerate(unique_classes)}
        
        
        
        df['jutsu_type'] = [codes[type] for type in df['jutsu_type']]
        
        tokenized_train, tokenized_test = self._split_tokenize(df)

        return tokenized_train, tokenized_test

    

    def load_tokenizer(self):
        if huggingface_hub.repo_exists(self.model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def tokenize_function(self, examples):
        return self.tokenizer(examples['jutsu_description'], truncation=True, padding=True)

    def _split_tokenize(self, df):
        test_size = 0.2
        random_state = 1234
        df_train, df_test = train_test_split(df,
                                             test_size=test_size,
                                             random_state=random_state,
                                             shuffle=True,
                                             stratify=df['jutsu_type'])
        
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_test = test_dataset.map(self.tokenize_function, batched=True)
        tokenized_test = tokenized_test.rename_column('jutsu_type', 'labels')
        tokenized_train = tokenized_train.rename_column('jutsu_type', 'labels')
        
        return tokenized_train, tokenized_test

    
    def train(self):

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name,num_labels = len(self.classes))
         
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.model_path,
            evaluation_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            logging_dir=f'/Users/brighttenkorangofori/Desktop/naruto/text_classification/logs',
            logging_steps=10,
            weight_decay=0.01,
            learning_rate=2e-4,
            push_to_hub=True,
           
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        # Initialize Trainer
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_test,
            data_collator=data_collator,
            
            compute_metrics = compute_metrics
        )
        
        trainer.set_device(self.device)
        trainer.train()
       


        

        

        # Flush memory
        del self.model, trainer
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
      

    def load_model(self,):
        if huggingface_hub.repo_exists(self.model_path):
            print('repo found')
        
            try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path,num_labels = len(self.classes))
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                
                    return pipeline('text-classification', model=self.model, tokenizer=self.tokenizer, return_all_scores=True)
            except Exception as e:
                    print('model not found in repo')
                    return None
        else:

            return None
        

    
    def post_process(self, model_output):
        scores = []
        label_predictions = []
        for prediction in model_output:
            for pred in prediction:
                scores.append(pred['score'])
                max_score = np.max(scores)
                if pred['score'] == max_score:
                    label = pred['label']
                    label = int(label.split('_')[-1])
                    label = self.decodes.get(label)
            label_predictions.append(label)
        return label_predictions.pop()

    def classify_justsu(self, text):
        output = self.model(text)
        output = self.post_process(output)
        return output
