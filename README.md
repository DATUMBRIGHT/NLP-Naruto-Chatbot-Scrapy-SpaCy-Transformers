**NARUTO_TVSERIES_NLP_ANALYSIS_AND_CHATBOT**

Overview

NARUTO_TVSERIES_NLP_ANALYSIS_AND_CHATBOT is a project that integrates Natural Language Processing (NLP), 

data visualization, and chatbot functionality to analyze textual data from the Naruto TV series. 

The project involves scraping subtitles and Naruto-related data, classifying jutsu types, and providing interactive visualizations and chatbots.

Features

**Web Scraping**: Extracted subtitles and Naruto TV series data using Scrapy.

**Named Entity Recognition (NER)**: Detected and categorized entities from the text using SpaCy.

**Data Visualization:** Rendered entities and their relationships in interactive graphs.

**Text Classification:** Developed a classifier to categorize different jutsu types.

**Interactive Chatbot**: Created a chatbot using Gradio to interact with users based on the trained model.
Technologies

Scrapy: For web scraping and data extraction.

SpaCy: For Named Entity Recognition (NER) and entity extraction.

Hugging Face Transformers: For building and training text classification models.

Gradio: For creating interactive web interfaces for chatbots.

Pandas: For data manipulation and preprocessing.

PyTorch: For model training and evaluation with Hugging Face Transformers.

Scikit-Learn: For additional machine learning utilities and data preprocessing.



**Data Scraping**
Scraped Data: Used Scrapy to scrape subtitles and other Naruto-related data. Ensure Scrapy is correctly set up and configured for this purpose.

**Data Preparation**
Data Loading and Preprocessing: Loaded and processed data files, which included combining and cleaning text fields and encoding categorical labels.

**Model Training**
Training the Model: Trained a text classification model using the preprocessed data. The training involved setting up training arguments and evaluating the model.

**Data Visualization**

**Entity Graph Generation:** Visualized named entities and their relationships using integrated graphing functions and libraries.

**Gradio Interface**

**Interactive Chatbot**: Ran the Gradio web interface to interact with the chatbot. This interface allowed users to classify text and visualize data interactively.
