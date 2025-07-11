from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForTokenClassification, AutoTokenizer
from bs4 import BeautifulSoup
import requests
import torch
import re
import json

app = Flask(__name__)

MODEL_PATH = "./ner_model_simple"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
LABEL_LIST = ["O", "B-PRODUCT", "I-PRODUCT"]

def clean_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'[^\w\s.,$£€-]', ' ', text)
    text = ' '.join(text.split())
    return text

def predict_entities(text):
    """Predict named entities in text with character-level alignment"""
    cleaned_text = clean_text(text)
    
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [LABEL_LIST[pred] for pred in predictions[0].numpy()]
    
    reconstructed_text = ""
    char_positions = []
    for token in tokens:
        if token == "[CLS]" or token == "[SEP]":
            continue
        if token.startswith("##"):
            token = token[2:]
            reconstructed_text += token
        else:
            if reconstructed_text:
                reconstructed_text += " " + token
            else:
                reconstructed_text = token
        char_positions.append((len(reconstructed_text) - len(token), len(reconstructed_text)))
    
    entities = []
    current_entity = ""
    current_label = ""
    start_pos = -1
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            if current_entity:
                entities.append((current_entity.strip(), current_label[2:], start_pos, char_positions[i-1][1]))
            current_entity = token.replace("##", "")
            current_label = label
            start_pos = char_positions[i][0]
        elif label.startswith("I-") and current_label[2:] == label[2:]:
            current_entity += " " + token.replace("##", "")
        else:
            if current_entity:
                entities.append((current_entity.strip(), current_label[2:], start_pos, char_positions[i-1][1]))
            current_entity = ""
            current_label = ""
            start_pos = -1
    
    if current_entity:
        entities.append((current_entity.strip(), current_label[2:], start_pos, char_positions[-1][1]))
    
    return entities, reconstructed_text

def extract_content(url):
    """Extract content from a webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse
        soup = BeautifulSoup(response.text, 'html.parser')

        for element in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
            element.decompose()
        
        text = soup.get_text(separator='\n', strip=True) 
        with open("D:\proga\\test3\data\checking.json", "w", encoding="utf-8") as f:
            json.dump(text, f, ensure_ascii=False)       
        return text, None
    
    except Exception as e:
        return None, f"Error processing URL: {str(e)}"

def find_full_lines_with_products(text, products):
    """Find full lines containing product entities"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    product_lines = []
    
    for line in lines:
        for product in products:
            if product['name'].lower() in line.lower():
                product_lines.append(line)
                break
    
    return product_lines

def organize_products(entities, original_text):
    """Organize extracted entities into product structures with full lines"""
    products = []
    
    lines = [line.strip() for line in original_text.split('\n') if line.strip()]
    
    for entity, label, start_pos, end_pos in entities:
        if label != "PRODUCT":
            continue
            
        for line in lines:
            if entity in line:
                products.append({
                    "entity": entity,
                    "full_line": line
                })
                break
    
    seen = set()
    unique_products = []
    for p in products:
        if p['entity'] not in seen:
            seen.add(p['entity'])
            unique_products.append(p)
    
    return unique_products

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for analyzing furniture store URLs"""
    
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    if not url.startswith(('http://', 'https://')):
        return jsonify({"error": "Invalid URL format"}), 400
    
    original_text, error = extract_content(url)
    if error:
        return jsonify({"error": error}), 400
    
    entities, cleaned_text = predict_entities(original_text)
    
    products = organize_products(entities, original_text)

    print(f"Extracted entities: {entities}")
    print("Products before sending:", products)
    
    return jsonify({
        "url": url,
        "products": products,
        "count": len(products)
    })

if __name__ == '__main__':
    app.run(debug=True)