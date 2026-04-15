#ingest_simple_wiki.py

import json
from datasets import load_dataset

# --- CONFIGURATION ---
DATASET_NAME = "pszemraj/simple_wikipedia"
DATASET_SPLIT = "train" 
OUTPUT_FILE = 'simple_wikipedia_data.json'
MAX_ARTICLES = 20000 
# --------------------

# --- NEW: Intelligent Text Cleaning Function ---
def clean_wikipedia_text(raw_text: str) -> str:
    """
    Cleans the raw text from the dataset by removing common footer sections
    like 'References', 'See also', and categories.
    """
    # These are the headers that typically mark the end of the main content
    stop_headers = [
        'References',
        'Other websites',
        'See also',
        'External links'
    ]
    
    lines = raw_text.split('\n')
    clean_lines = []
    
    for line in lines:
        # Check if the line is one of our stop headers
        if line.strip() in stop_headers:
            # If we find a stop header, we stop processing completely
            break 
        
        # Add the line to our clean list if it's not a stop header
        clean_lines.append(line)
        
    # Join the clean lines back together and remove any trailing whitespace
    return "\n".join(clean_lines).strip()
# ---------------------------------------------


def process_simple_wikipedia():
    """
    Downloads the Simple Wikipedia dataset, cleans each article, and saves it to a JSON file.
    """
    print(f"--- Starting Simple Wikipedia Ingestion ---")
    print(f"Downloading dataset '{DATASET_NAME}'...")
    
    try:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, trust_remote_code=True)
        print("Dataset downloaded successfully.")
        
        all_articles_data = []
        
        print(f"Processing and cleaning up to {MAX_ARTICLES} articles...")
        for i, item in enumerate(dataset):
            if len(all_articles_data) >= MAX_ARTICLES:
                print(f"Reached article limit of {MAX_ARTICLES}. Stopping.")
                break
                
            title = item.get('title', '')
            raw_text_content = item.get('text', '')
            
            # --- THIS IS THE NEW STEP: Clean the text ---
            clean_text = clean_wikipedia_text(raw_text_content)
            # ---------------------------------------------
            
            # We now check the length of the CLEAN text
            if title and len(clean_text) > 300: 
                all_articles_data.append({
                    "title": title,
                    "text": clean_text, # Save the clean text
                    "url": item.get('url', '')
                })
        
        return all_articles_data

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return []

if __name__ == "__main__":
    articles_data = process_simple_wikipedia()
    
    if articles_data:
        print(f"\nSuccessfully processed and filtered {len(articles_data)} articles.")
        print(f"Saving data to {OUTPUT_FILE}...")
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(articles_data, f, ensure_ascii=False, indent=4)
            
        print("--- Ingestion process complete! ---")