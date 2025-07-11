# Task Completion Report  
**Objective**: Develop a solution capable of extracting product information from furniture store websites.  
**Functionality**:  
1. Loading and parsing web pages.  
2. Extracting textual content.  
3. Identifying products (entities of type "PRODUCT") using a pre-trained model.  
4. Displaying results both on the website and in the console.  

## Implemented Components  
**Frontend**: Interactive HTML interface for URL input and result display; processing user input, sending requests to the backend, and dynamically updating data.  

**Backend**: API for URL analysis; parsing web pages using BeautifulSoup; utilizing a Hugging Face model (BERT) for entity recognition (products); post-processing results (removing duplicates, aligning with source text).  

**ML Components**: NER model for token classification (BIO tagging); functions for text processing and entity alignment.  

## Key Implementation Features  
**Text Processing**:  
1. Cleaning text of unnecessary symbols and HTML tags.  
2. Splitting into lines for accurate product-context matching.  

**NER Model**:  
1. Using the pre-trained `bert-base-cased` model.  
2. Fine-tuning on a dataset with labeled products.  
3. Processing tokens while considering their positions in the text.  

**Error Handling**:  
1. URL validation.  
2. Catching exceptions during page loading and content analysis.  

**Optimizations**:  
1. Removing duplicate products.  
2. Logging results for debugging.  

## Setup Instructions  
**Install dependencies**:  

`pip install flask transformers torch beautifulsoup4 requests pandas`  

**Run the server**:  

`python app.py`  

**Open in browser**: http://localhost:5000.  

## Website Operation  
### Test 1  
**Input**: Valid URL https://www.factorybuys.com.au/products/dining-table-4-seater-wooden-kitchen-tables-white-120cm-cafe-restaurant  

**Output**:  
<img width="1098" height="810" alt="image" src="https://github.com/user-attachments/assets/5d9cb1eb-8ba8-4135-9029-ad6381f868c0" />  

### Test 2  
**Input**: Non-URL input `some_trash`  

**Output**:  
<img width="1098" height="433" alt="image" src="https://github.com/user-attachments/assets/ae7517f7-bf30-4f53-ab78-35ad7750cd31" />  

### Test 3  
**Input**: Invalid URL https://brooklyncityfurniture.com/collections/living-roo  

**Output**:  
<img width="1081" height="431" alt="image" src="https://github.com/user-attachments/assets/72c48df6-b0fa-4787-8a3e-6e957ca19cee" />  

## Encountered Issues  
**Insufficient provided data**: Out of 700 parsed links, only 100 were valid.  

Solution: Added synthetic data to the training and validation sets. The total dataset size reached 350 entries (100 natural + 250 synthetic).  

Result: Model performance significantly improved.  

**Noise in output**: Some tokens unrelated to store products were incorrectly classified as products.  

Potential solution: Enhance post-processing or further train the model (e.g., adding an `OTHER` entity).  

## Potential Improvements  
1. Expanding the training dataset.  
2. Switching to a different model.  
3. Extracting additional entities like price, material, and color.  

## Conclusion  
The project demonstrates skills in web development, natural language processing, data parsing, and cleaning.  

The project is ready for further refinement and integration into real-world processes.
