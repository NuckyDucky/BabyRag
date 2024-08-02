import os
import re
import json
import gradio as gr
import PyPDF2
import chardet
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModel
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)
app = Flask(__name__)

UPLOAD_FOLDER = 'extensions/baby_rag/uploads'
EMBEDDINGS_FILE = 'extensions/baby_rag/embeddings.json'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

params = {
    'model_name': 'bert-base-uncased',
    'chunk_length': 700,
    'truncation': True,
    'padding': True,
    'max_length': 512,
    'batch_size': 8,
    'return_tensors': 'pt',
    'add_special_tokens': True,
    'stride': 0,
    'is_split_into_words': False,
    'return_attention_mask': True,
    'return_token_type_ids': True,
    'return_length': False,
    'verbose': False,
    'use_fast': True,
    'add_prefix_space': False,
    'do_lower_case': True,
    'strip_accents': None,
    'do_basic_tokenize': True,
    'never_split': None,
    'pad_token': '[PAD]',
}

model_choices = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased', 'gpt2']
tensor_choices = ['pt', 'tf', 'np']

logging.basicConfig(level=logging.DEBUG)

class PreprocessData:
    def __init__(self, text):
        self.text = text
        
    def preprocess(self):
        text = self.text
        text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces/newlines
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        text = re.sub(r'•', ' ', text)  # Remove bullet points
        text = re.sub(r'[\*\-–—]', ' ', text)  # Remove common special characters
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove all other special characters
        text = text.strip()  # Remove leading/trailing whitespace
        return text

class TextEmbedding:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=params['use_fast'],
            add_prefix_space=params['add_prefix_space'],
            do_lower_case=params['do_lower_case'],
            strip_accents=params['strip_accents'],
            do_basic_tokenize=params['do_basic_tokenize'],
            never_split=params['never_split'],
            pad_token=params['pad_token']
        )
        self.model = AutoModel.from_pretrained(model_name)

    def get_embeddings(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors=params['return_tensors'],
            truncation=params['truncation'],
            padding=params['padding'],
            max_length=params['max_length'],
            add_special_tokens=params['add_special_tokens'],
            stride=params['stride'],
            is_split_into_words=params['is_split_into_words'],
            return_attention_mask=params['return_attention_mask'],
            return_token_type_ids=params['return_token_type_ids'],
            return_length=params['return_length'],
            verbose=params['verbose']
        )
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()
        return embeddings

def save_embeddings_locally(embeddings, text, file_path=EMBEDDINGS_FILE):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = {"embeddings": []}

    data['embeddings'].append({"text": text, "embedding": embeddings[0]})

    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_embeddings(file_path=EMBEDDINGS_FILE):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        return {"embeddings": []}

def delete_embeddings(file_path=EMBEDDINGS_FILE):
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": "Embeddings file deleted."}
    return {"error": "Embeddings file not found."}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file_upload', methods=['POST'])
def handle_file_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Detect encoding if text file
        if filename.endswith('.txt'):
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        elif filename.endswith('.pdf'):
            content = extract_text_from_pdf(file_path)

        processor = PreprocessData(content)
        preprocessed_text = processor.preprocess()

        # Generate embeddings
        embedding_model = TextEmbedding(params['model_name'])
        embeddings = embedding_model.get_embeddings(preprocessed_text)

        # Save embeddings locally
        save_embeddings_locally(embeddings, preprocessed_text)
        return jsonify({"message": "Embeddings saved locally"})
    return jsonify({"error": "Invalid file format"})

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfFileReader(f)
        text = ""
        for page_num in range(reader.getNumPages()):
            text += reader.getPage(page_num).extract_text()
    return text

def feed_file_into_collector(file, chunk_len, chunk_sep):
    yield 'Reading the input dataset...\n\n'
    text = file.decode('utf-8', errors='ignore')
    processor = PreprocessData(text)
    preprocessed_text = processor.preprocess()

    # Generate embeddings
    embedding_model = TextEmbedding(params['model_name'])
    embeddings = embedding_model.get_embeddings(preprocessed_text)
    save_embeddings_locally(embeddings, preprocessed_text)
    yield "Embeddings have been saved locally.\n\n"

def fetch_embeddings():
    yield "Fetching embeddings from local storage...\n\n"
    embeddings_data = load_embeddings()
    
    # Formatting embeddings data as text to be saved in context.txt
    embeddings_text = json.dumps(embeddings_data, indent=4)
    
    # Saving to a .txt file
    with open('context.txt', 'w') as f:
        f.write(embeddings_text)

    yield f"Fetched embeddings: {embeddings_data}\n\n"

def delete_embeddings_and_respond():
    result = delete_embeddings()
    return json.dumps(result)

def apply_settings(model_name, chunk_length, truncation, padding, max_length, batch_size, return_tensors, add_special_tokens, stride, is_split_into_words, return_attention_mask, return_token_type_ids, return_length, verbose, use_fast, add_prefix_space, do_lower_case, strip_accents, do_basic_tokenize, never_split, pad_token):
    global params
    params['model_name'] = model_name
    params['chunk_length'] = int(chunk_length)
    params['truncation'] = truncation
    params['padding'] = padding
    params['max_length'] = int(max_length)
    params['batch_size'] = int(batch_size)
    params['return_tensors'] = return_tensors
    params['add_special_tokens'] = add_special_tokens
    params['stride'] = int(stride)
    params['is_split_into_words'] = is_split_into_words
    params['return_attention_mask'] = return_attention_mask
    params['return_token_type_ids'] = return_token_type_ids
    params['return_length'] = return_length
    params['verbose'] = verbose
    params['use_fast'] = use_fast
    params['add_prefix_space'] = add_prefix_space
    params['do_lower_case'] = do_lower_case
    params['strip_accents'] = strip_accents
    params['do_basic_tokenize'] = do_basic_tokenize
    params['never_split'] = never_split
    params['pad_token'] = pad_token
    yield f"The following settings are now active: {params}\n\n"

def get_most_relevant_text(input_embedding, embeddings_data, top_n=1):
    try:
        # Reshape each embedding array to 2D array if it's not already
        embeddings = [np.array(item['embedding']).reshape(1, -1) for item in embeddings_data['embeddings']]
        texts = [item['text'] for item in embeddings_data['embeddings']]

        # Ensure the input embedding is also reshaped to a 2D array
        input_embedding = np.array(input_embedding).reshape(1, -1)
        
        # Calculate cosine similarity between input embedding and all stored embeddings
        similarities = np.vstack([cosine_similarity(input_embedding, emb) for emb in embeddings])

        # Get the indices of the top n most similar embeddings
        most_relevant_indices = np.argsort(similarities.flatten())[-top_n:]

        return [texts[i] for i in most_relevant_indices]
    except Exception as e:
        print(f"Error processing embeddings: {e}")
        return []

def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """
    # Load embeddings
    embeddings_data = load_embeddings()
    embedding_model = TextEmbedding(params['model_name'])
    user_input_embedding = embedding_model.get_embeddings(user_input)

    # Get the most relevant past text based on embeddings
    relevant_texts = get_most_relevant_text(user_input_embedding, embeddings_data)

    # Incorporate the relevant texts into the chat prompt
    context = "\n".join(relevant_texts)
    context = f"Context from past interactions:\n{context}\n\n"

    # Original chat prompt generation logic
    original_prompt = chat.generate_chat_prompt(user_input, state, **kwargs)
    
    # Modify the result to include context
    custom_prompt = context + original_prompt
    
    return custom_prompt

def ui():
    with gr.Accordion("Click for more information...", open=False):
        gr.Markdown("""
        ## About
        This extension processes text or PDF files to generate text embeddings and stores them in a local file.
        You can load data and fetch embeddings using the interface below.
        """)

    with gr.Row():
        with gr.Column(min_width=600):
            with gr.Tab("File input"):
                file_input = gr.File(label='Input file', type='binary')
                chunk_sep = gr.Textbox(value='', label='Chunk separator', info='Used to manually split chunks. Manually split chunks longer than chunk length are split again. This value is used when you click on "Load data".')
                update_file = gr.Button('Load data')

            with gr.Tab("Fetch embeddings"):
                fetch_embeddings_btn = gr.Button('Fetch embeddings')

            with gr.Tab("Settings"):
                model_name = gr.Dropdown(choices=model_choices, value=params['model_name'], label='Model name')
                chunk_len = gr.Number(value=params['chunk_length'], label='Chunk length')
                truncation = gr.Checkbox(value=params['truncation'], label='Truncation')
                padding = gr.Checkbox(value=params['padding'], label='Padding')
                max_length = gr.Number(value=params['max_length'], label='Max length')
                batch_size = gr.Number(value=params['batch_size'], label='Batch size')
                return_tensors = gr.Dropdown(choices=tensor_choices, value=params['return_tensors'], label='Return tensors')
                add_special_tokens = gr.Checkbox(value=params['add_special_tokens'], label='Add special tokens')
                stride = gr.Number(value=params['stride'], label='Stride')
                is_split_into_words = gr.Checkbox(value=params['is_split_into_words'], label='Is split into words')
                return_attention_mask = gr.Checkbox(value=params['return_attention_mask'], label='Return attention mask')
                return_token_type_ids = gr.Checkbox(value=params['return_token_type_ids'], label='Return token type ids')
                return_length = gr.Checkbox(value=params['return_length'], label='Return length')
                verbose = gr.Checkbox(value=params['verbose'], label='Verbose')
                use_fast = gr.Checkbox(value=params['use_fast'], label='Use fast tokenizer')
                add_prefix_space = gr.Checkbox(value=params['add_prefix_space'], label='Add prefix space')
                do_lower_case = gr.Checkbox(value=params['do_lower_case'], label='Do lower case')
                strip_accents = gr.Checkbox(value=params['strip_accents'], label='Strip accents')
                do_basic_tokenize = gr.Checkbox(value=params['do_basic_tokenize'], label='Do basic tokenize')
                never_split = gr.Textbox(value='', label='Never split', info='List of tokens to never split, separated by spaces.')
                pad_token = gr.Textbox(value=params['pad_token'], label='Pad token', info='Token to be used for padding.')
                update_settings = gr.Button('Apply changes')

            with gr.Tab("Manage Embeddings"):
                delete_embeddings_btn = gr.Button('Delete embeddings')

            last_updated = gr.Markdown()

    update_file.click(feed_file_into_collector, [file_input, chunk_len, chunk_sep], last_updated, show_progress=False)
    fetch_embeddings_btn.click(fetch_embeddings, [], last_updated, show_progress=False)
    update_settings.click(apply_settings, [model_name, chunk_len, truncation, padding, max_length, batch_size, return_tensors, add_special_tokens, stride, is_split_into_words, return_attention_mask, return_token_type_ids, return_length, verbose, use_fast, add_prefix_space, do_lower_case, strip_accents, do_basic_tokenize, never_split, pad_token], last_updated, show_progress=False)
    delete_embeddings_btn.click(delete_embeddings_and_respond, [], last_updated, show_progress=False)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
