import gradio as gr
import glob
import pandas as pd
import spacy
import networkx as nx
from pyvis.network import Network
from nltk import sent_tokenize
from text_classification.cleaner import Cleaner
from itertools import combinations

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize cleaner
cleaner = Cleaner()

# Define the class
class generateEntRelationship:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def get_scripts(self):
        subs = []
        self.file_paths = glob.glob(f'{self.file_paths}/*')
        self.file_paths.sort()
        for path in self.file_paths:
            with open(path, 'r') as f:
                texts = f.readlines()[27:]
                texts = [text.split(',,')[-1] for text in texts]
                texts = [text.replace('\\N', ' ') for text in texts]
                texts = [text.replace('\n', '') for text in texts]
                subs.append(texts)
                scripts = [''.join(sub) for sub in subs]
        
        return scripts

    def generate_subs_df(self,scripts):
        seasons = []
        episodes = []
        for path in self.file_paths:
            season = int(path.split('Season')[-1].split('-')[0].strip())
            seasons.append(season)
            episode = int(path.split('-')[-1].split('.')[0])
            episodes.append(episode)

        df = pd.DataFrame({'scripts': scripts,
                           'seasons': seasons,
                           'episodes': episodes})
        
        return df

    def get_named_entities(self, text):
        names = []
        episode_sentences = sent_tokenize(str(text))
        for sentence in episode_sentences:
            sentence = cleaner.clean(sentence)
            doc = nlp(sentence)
            sets = set()
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    full_name = ent.text.split(' ')
                    first = full_name[0]
                    sets.add(first)
            names.append(list(sets))
        
        return names

    def get_relationship_pairs(self,df):
        window=10
        entity_relationship = []

        for row in df['names']:
            previous_entities_in_window = []
            
            for sentence in row:
                previous_entities_in_window.append(sentence)
                previous_entities_in_window = previous_entities_in_window[-window:]
                
                previous_entities_flattened= sum(previous_entities_in_window, [])
                
                for entity in sentence:            
                    for entity_in_window in previous_entities_flattened:
                        if entity!=entity_in_window:
                            entity_rel = sorted([entity,entity_in_window])
                            entity_relationship.append(entity_rel)
        return entity_relationship
      
    
    def plot_entities(self, entity_relationship):
        # Create DataFrame from entity relationships
        relationship_df = pd.DataFrame({'value' : entity_relationship})
        
        (relationship_df.head())
        
        relationship_df['source'] = relationship_df['value'].apply(lambda x: x[0])
        relationship_df['target'] = relationship_df['value'].apply(lambda x: x[1])
        
        # Group by 'source' and 'target' to count occurrences
        relationship_df = relationship_df.groupby(['source', 'target']).count().reset_index()
        
        # Sort by count and select the top 200 relationships
        relationship_df = relationship_df.sort_values('value', ascending=False).head(200)
        
        # Create NetworkX graph
        G = nx.from_pandas_edgelist(relationship_df,
                                    source="source",
                                    target="target",
                                    edge_attr="value",
                                    create_using=nx.Graph())

        # Create Pyvis network
       
        
        net = Network(notebook = True, width="1000px", height="700px", bgcolor='#222222', font_color='white')

        node_degree = dict(G.degree)

        #Setting up node size attribute
        nx.set_node_attributes(G, node_degree, 'size')

        net.from_nx(G)
        
        
        net.show("naruto.html")
        with open("naruto.html", "r") as f:
            html_content = f.read()
            
            html = f"""<iframe style="width: 100%; height: 600px; margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; display-capture; encrypted-media;" 
            sandbox="allow-modals allow-forms allow-scripts allow-same-origin allow-popups allow-top-navigation-by-user-activation allow-downloads" 
            allowfullscreen="" allowpaymentrequest="" frameborder="0" srcdoc="{html_content}"></iframe>"""


        return html
    
      
        
        
    
        