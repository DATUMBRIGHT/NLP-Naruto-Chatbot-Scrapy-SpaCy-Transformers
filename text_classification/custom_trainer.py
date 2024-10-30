from transformers import Trainer
import torch 
from torch.nn import CrossEntropyLoss
import pandas as pd
from transformers import AutoModelForSequenceClassification
from text_classification.training_utils import get_weights


class CustomTrainer(Trainer):
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels from inputs
        labels = inputs.get('labels')
        
        # Move inputs and labels to the appropriate device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float()
        
        # Compute the loss
        loss_fcn = CrossEntropyLoss()
        loss = loss_fcn(logits.view(-1,model.config.num_labels),labels.view(-1))
        
        if return_outputs:
            return loss, outputs
        else:
            return loss
        
    def set_device(self,device):
        
         self.device = device
         return self.device
        

    
    

        

        
        

    