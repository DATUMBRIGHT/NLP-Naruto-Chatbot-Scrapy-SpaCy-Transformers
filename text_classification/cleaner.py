from bs4 import BeautifulSoup
import re


class Cleaner:

    def __init__(self):

        pass
    
    
    
    
    def clean(self,text):
        text = text.lower()  # Convert to lower case
        text = text.strip()  # Remove leading and trailing spaces
        
        # Remove special characters, URLs, and extra spaces
        text = re.sub(r'https?://\S+', ' ', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters (keep letters, numbers, and spaces)
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text =re.sub(r'[!@#$%^&*(_+)]','',text)
        return text