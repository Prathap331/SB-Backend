#ingest_huggingface.py

import json
from datasets import load_dataset
import re

# --- CONFIGURATION ---
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "train" 
OUTPUT_FILE = 'huggingface_wikitext.json'
# We'll set a high limit, but the script will naturally find the true number of articles
MAX_ARTICLES = 25000 
# --------------------

def process_huggingface_dataset():
    """
    Downloads and correctly processes the wikitext dataset by grouping paragraphs into full articles.
    """
    print(f"--- Starting Hugging Face Dataset Ingestion ---")
    print(f"Downloading dataset '{DATASET_NAME}' ({DATASET_CONFIG})...")
    
    try:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
        print("Dataset downloaded successfully.")
        
        all_articles_data = []
        current_title = None
        current_text_lines = []
        
        print("Processing and grouping articles...")
        for item in dataset:
            text_content = item.get('text', '').strip()
            
            if not text_content:
                continue

            # Check if the line is a main article title (e.g., " = Article Name = ")
            # We look for single '=' on each side, not '==' which is a sub-header.
            match = re.match(r'^\s*=\s([^=]+)\s=\s*$', text_content)

            if match:
                # If we found a new title, it means the previous article is complete.
                # Save the completed previous article before starting the new one.
                if current_title and current_text_lines:
                    full_text = "\n".join(current_text_lines)
                    if len(full_text) > 200: # Only save if it has substantial content
                        all_articles_data.append({
                            "title": current_title,
                            "text": full_text
                        })
                        
                        # Print progress
                        if len(all_articles_data) % 100 == 0:
                            print(f"  ...saved {len(all_articles_data)} complete articles...")
                
                # Start the new article
                current_title = match.group(1).strip()
                current_text_lines = []
                
                if len(all_articles_data) >= MAX_ARTICLES:
                    print(f"Reached article limit of {MAX_ARTICLES}. Stopping.")
                    break
            
            # If it's not a title, it's a content paragraph. Add it to the current article.
            elif current_title and not text_content.startswith(' = '):
                current_text_lines.append(text_content)

        # After the loop, save the very last article that was being built
        if current_title and current_text_lines:
            full_text = "\n".join(current_text_lines)
            if len(full_text) > 200:
                all_articles_data.append({
                    "title": current_title,
                    "text": full_text
                })

        return all_articles_data

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return []


if __name__ == "__main__":
    articles_data = process_huggingface_dataset()
    
    if articles_data:
        print(f"\nSuccessfully processed and grouped {len(articles_data)} articles.")
        print(f"Saving data to {OUTPUT_FILE}...")
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(articles_data, f, ensure_ascii=False, indent=4)
            
        print("--- Ingestion process complete! ---")





