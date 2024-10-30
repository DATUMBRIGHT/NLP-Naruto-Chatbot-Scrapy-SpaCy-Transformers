import evaluate
from sklearn.utils.class_weight import compute_class_weight
import numpy as np 
import pandas as pd 
from itertools import chain



def get_weights(df):
    # Flatten the list of jutsu types
    all_jutsu_types = list(chain.from_iterable(df['jutsu_type']))
    
    # Get unique classes
    unique_classes = pd.unique(all_jutsu_types)
    
    # Compute weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=all_jutsu_types
    )
    
    # Create a dictionary mapping each class to its weight
    weight_dict = dict(zip(unique_classes, weights))
    
    return weight_dict



metric = evaluate.load('accuracy')

def compute_metrics(predictions):
        logits,labels = predictions
        preds = np.argmax(logits,axis = -1)
        return metric.compute(predictions=preds,references=labels)



         