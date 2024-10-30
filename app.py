from glob import glob
import gradio as gr
import torch
import pandas as pd
from cer.entity_relationship import generateEntRelationship
from text_classification.jutsu_classifier import JutsuPredictor
from utils.data_loader import load_subtitles
from theme_classifier.classifer import themeClassifier
from dotenv import load_dotenv
from chat_bot.chat_bot import ChatBotTrainer
import os

load_dotenv()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_response(message,history):
    trainer = ChatBotTrainer(model_path= 'datumbright/narutocb',
                             hf_token='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',
                             data_path= '/Users/brighttenkorangofori/Desktop/naruto/data/transcript/naruto.csv')
    
    output = trainer.chat(message = message,
                 history = history)
    
    return output



def process_file(subs_path, save_path):
    df = load_subtitles(subs_path)
    print(df.head())
    return df

def get_theme(subs_df):
    classifier = themeClassifier(batch_size=20)
    classifier.load_model()
    classes = classifier.get_theme(subs_df['subtitles'])
    plot = classifier.plot_themes()
    return plot

def plot_relationships(subs_path):
    generator = generateEntRelationship(subs_path)
    scripts = generator.get_scripts()
    subs_df = generator.generate_subs_df(scripts)
    subs_df['names'] = subs_df['scripts'].apply(generator.get_named_entities)
    
    relationships = generator.get_relationship_pairs(subs_df)
    
    html_plot = generator.plot_entities(relationships)
    return html_plot
  


def classify_text(text, model_path, data_path):
    classifier = JutsuPredictor(model_path=model_path,
                                data_path=data_path,
                                hugging_face_token=  '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')
    
    
    

    # Classify the input text
    outputs = classifier.classify_justsu(text)
    return outputs

def main():
    with gr.Blocks() as iface:
        gr.HTML('<h1>Welcome to Naruto Text Classifier</h1>')
        
        with gr.Tab("Subtitle Processing"):
            subs_path = gr.Textbox(label="Subtitles Path")
            save_path = gr.Textbox(label="Save Path")
            process_btn = gr.Button("Process Subtitles")
            subs_df = gr.DataFrame(label="Processed Subtitles")
            process_btn.click(fn=process_file, inputs=[subs_path, save_path], outputs=subs_df)

        with gr.Tab("Theme Classification"):
            theme_btn = gr.Button("Classify Themes")
            theme_plot = gr.Plot(label="Theme Classification Plot")
            theme_btn.click(fn=get_theme, inputs=subs_df, outputs=theme_plot)

        with gr.Tab("Entity Relationships"):
            rel_btn = gr.Button("Plot Relationships")
            rel_plot = gr.HTML()
            rel_btn.click(fn=plot_relationships, inputs=subs_path, outputs=rel_plot)

        with gr.Tab("Text Classification"):
            text_input = gr.Textbox(label="Enter Text to Classify")
            model_path = gr.Textbox(label="Model Path")
            data_path = gr.Textbox(label="Data Path")

            classify_btn = gr.Button("Classify")
            classification_result = gr.Textbox(label="Classification Result")
            classify_btn.click(fn=classify_text, 
                               inputs=[text_input, model_path, data_path], 
                               outputs=classification_result)
            
        with gr.Tab("Naruto Chatbot"):

            gr.HTML('<h1> Welcome To Naruto Chatbot</h1>')
           
            chatbot = gr.Chatbot()
            text_input = gr.Textbox('hi im Naruto. Ask me anything')
            submit_btn = gr.Button('submit')

            submit_btn.click(fn=generate_response,
                             inputs=[text_input,chatbot],
                             outputs=chatbot)
            
            clear_btn = gr.Button('Clear')
            clear_btn.click()
    
    iface.launch()

if __name__ == '__main__':
    main()