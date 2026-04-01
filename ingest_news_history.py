#ingest_news_history.py
import json
from datasets import load_dataset

# --- CONFIGURATION ---
DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"
DATASET_SPLIT = "train" 
OUTPUT_FILE = 'news_history_data.json'
MAX_ARTICLES = 5000 # Let's take 30,000 news articles
# --------------------

def process_news_dataset():
    """
    Downloads the CNN/DailyMail dataset, processes it, and saves it to a JSON file.
    """
    print(f"--- Starting News History Ingestion ---")
    print(f"Downloading dataset '{DATASET_NAME}' ({DATASET_CONFIG})...")
    
    try:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
        print("Dataset downloaded successfully.")
        
        all_articles_data = []
        
        print(f"Processing the first {MAX_ARTICLES} articles...")
        for i, item in enumerate(dataset):
            if i >= MAX_ARTICLES:
                print(f"Reached article limit of {MAX_ARTICLES}. Stopping.")
                break
                
            # This dataset uses 'article' for the main text and 'highlights' for a summary
            text_content = item.get('article', '')
            
            if len(text_content) > 300:
                # We'll use the first sentence as a proxy for the title
                first_sentence = text_content.split('.')[0]
                
                all_articles_data.append({
                    "title": first_sentence,
                    "text": text_content
                })
        
        return all_articles_data

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return []

if __name__ == "__main__":
    articles_data = process_news_dataset()
    
    if articles_data:
        print(f"\nSuccessfully processed {len(articles_data)} articles.")
        print(f"Saving data to {OUTPUT_FILE}...")
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(articles_data, f, ensure_ascii=False, indent=4)
            
        print("--- Ingestion process complete! ---")