from transformers import pipeline
import torch
import os
from nltk import sent_tokenize
import pandas as pd
import seaborn as sns
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from utils.data_loader import load_subtitles

import nltk
nltk.download('punkt')



current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up one directory and then into folder2
subs_path = os.path.join(current_dir, '..', 'data', 'subtitles')
file_path = glob('/Users/brighttenkorangofori/Desktop/naruto/data/subtitles/*')


class themeClassifier:

    def __init__(self,batch_size):

        self.model = "facebook/bart-large-mnli"
        self.device = ['cuda' if torch.cuda.is_available() else 'cpu']
        self.batch_size = batch_size

    def load_model(self):

        self.classifier = pipeline("zero-shot-classification",
                      model=self.model)

        return self.classifier
    

    

    def get_theme(self,subs):
        """"
        
        gets classes
        
        """
        labels = ['self-development','battle','hope','friendship','love','betrayal','sacrifice']
        scripts = ' '.join(subs)
        
        #individual sentences
        sentence_tokens = sent_tokenize(scripts)
        
        #clean text
        sentence_tokens = [token.replace('\\N',' ') for token in sentence_tokens]
    

        #divide into batchs 
        batches = []
        for i in range(0,len(sentence_tokens)-1,self.batch_size):
                batch = sentence_tokens[i : (i+self.batch_size)]
                batches.append(batch)

        #merge each 20 sentences in big strings
        batches = [' '.join(batch) for batch in batches]

        self.classes = self.classifier(batches[:3],labels)

        return self.classes
    
    def plot_themes(self):
        """  plot themes
             
        """
        c = {'sequence' : [],
            'labels' : [],
            'scores' : []}
        for clas in self.classes:
            for k,v in clas.items():
                c[k].append(v)

        df = pd.DataFrame(c)

        max_scores = []
        classification = []



        for score, label in zip(df['scores'], df['labels']):
            max_score = np.max(score)
            max_scores.append(max_score)
            index = score.index(max_score)
            name = label[index]
            classification.append(name)
        df['max_scores'] = max_scores
        df['classification'] = classification
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plot = sns.barplot(data=df, x='classification', y='max_scores', hue='classification')
        plt.title('Theme Classification Scores')
        fig = plot.get_figure()
        # Return the plot figure object
        return fig









