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
from modules.text_generation import decode, encode, generate_reply
import nltk
from bs4 import BeautifulSoup
import unicodedata
nltk.download('punkt')

app = Flask(__name__)

UPLOAD_FOLDER = 'extensions/baby_rag/uploads'
EMBEDDINGS_FILE = 'extensions/baby_rag/embeddings.json'
PRESET_FILE = 'extensions/baby_rag/preset.json'
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
    'normalize_unicode': False,
    'remove_stop_words': False,
    'lemmatize': False,
    'dynamic_padding': False,
    'data_augmentation': {
        'synonym_replacement': False,
        'back_translation': False
    },
    'cross_encoder_model': None,
    'pooling_strategy': 'mean',
    'fine_tune': False,
    'context_window_size': 256,
    'context_window_stride': 128,
    'correct_spelling_grammar': False,
    'custom_pretrained_embeddings': None
}

model_choices = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased', 'gpt2', 't5-small', 'albert-base-v2']
tensor_choices = ['pt', 'tf', 'np']
pooling_choices = ['mean', 'max', 'weighted']

logging.basicConfig(level=logging.DEBUG)

class PreprocessData:
    def __init__(self, text):
        self.text = text
        
    def preprocess(self):
        text = self.text
        if params['normalize_unicode']:
            text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces/newlines
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        text = re.sub(r'•', ' ', text)  # Remove bullet points
        text = re.sub(r'[\*\-–—]', ' ', text)  # Remove common special characters
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove all other special characters
        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
        text = text.strip()  # Remove leading/trailing whitespace
        sentences = nltk.sent_tokenize(text)  # Sentence tokenization

        if params['remove_stop_words']:
            stop_words = set(nltk.corpus.stopwords.words('english'))
            sentences = [' '.join([word for word in nltk.word_tokenize(sentence) if word.lower() not in stop_words]) for sentence in sentences]
        
        if params['lemmatize']:
            lemmatizer = nltk.stem.WordNetLemmatizer()
            sentences = [' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(sentence)]) for sentence in sentences]

        return " ".join(sentences)

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
        self.model_name = model_name

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
        
        if 't5' in self.model_name.lower():
            outputs = self.model(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                decoder_input_ids=inputs['input_ids']
            )
        else:
            outputs = self.model(**inputs)
        
        if params['pooling_strategy'] == 'mean':
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()
        elif params['pooling_strategy'] == 'max':
            embeddings = outputs.last_hidden_state.max(dim=1)[0].detach().numpy().tolist()
        else:
            # Implement weighted pooling if needed
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()  # Placeholder for weighted pooling
        return embeddings

def save_embeddings_locally(embeddings, text, file_path=EMBEDDINGS_FILE):
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

def auto_set_settings(text_length):
    if text_length > 5000:
        params.update({
            'do_lower_case': True,
            'strip_accents': True,
            'do_basic_tokenize': True,
        })
    else:
        params.update({
            'do_lower_case': False,
            'strip_accents': False,
            'do_basic_tokenize': False,
        })

def save_preset(file_path=PRESET_FILE):
    with open(file_path, 'w') as f:
        json.dump(params, f)
    return {"message": "Preset saved."}

def load_preset(file_path=PRESET_FILE):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            preset = json.load(f)
        params.update(preset)
        return {"message": "Preset loaded."}
    return {"error": "Preset file not found."}

def delete_preset(file_path=PRESET_FILE):
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": "Preset file deleted."}
    return {"error": "Preset file not found."}

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

        auto_set_settings(len(preprocessed_text))

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

    auto_set_settings(len(preprocessed_text))

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

def apply_settings(model_name, chunk_length, truncation, padding, max_length, batch_size, return_tensors, add_special_tokens, stride, is_split_into_words, return_attention_mask, return_token_type_ids, return_length, verbose, use_fast, add_prefix_space, do_lower_case, strip_accents, do_basic_tokenize, never_split, pad_token, normalize_unicode, remove_stop_words, lemmatize, dynamic_padding, synonym_replacement, back_translation, cross_encoder_model, pooling_strategy, fine_tune, context_window_size, context_window_stride, correct_spelling_grammar, custom_pretrained_embeddings):
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
    params['normalize_unicode'] = normalize_unicode
    params['remove_stop_words'] = remove_stop_words
    params['lemmatize'] = lemmatize
    params['dynamic_padding'] = dynamic_padding
    params['data_augmentation']['synonym_replacement'] = synonym_replacement
    params['data_augmentation']['back_translation'] = back_translation
    params['cross_encoder_model'] = cross_encoder_model
    params['pooling_strategy'] = pooling_strategy
    params['fine_tune'] = fine_tune
    params['context_window_size'] = int(context_window_size)
    params['context_window_stride'] = int(context_window_stride)
    params['correct_spelling_grammar'] = correct_spelling_grammar
    params['custom_pretrained_embeddings'] = custom_pretrained_embeddings
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

                    ### Tokenizer Settings

                    - **Model Name**: Select from a variety of pre-trained models like `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`, `gpt2`, `t5-small`, and `albert-base-v2`. 
                    - **When to use each model**: 
                        - `distilbert-base-uncased` is lightweight and fast, suitable for smaller tasks or limited computational resources.
                        - `bert-base-uncased` offers a balance between performance and accuracy, good for general-purpose use.
                        - `roberta-base` is robust and handles complex language patterns, ideal for nuanced text analysis.
                        - `gpt2` generates more coherent text, useful for text generation tasks.
                        - `t5-small` excels at text-to-text transformations, like summarization or translation.
                        - `albert-base-v2` is optimized for efficiency and memory, good for scenarios with tight resource constraints.

                    - **Chunk Length**: Specify the length of text chunks to be processed.
                    - **Impact**: A smaller chunk length can reduce memory usage and ensure that text fits within the model's max length. A larger chunk length can capture more context but may require more memory.

                    - **Truncation**: Enable or disable truncation of text chunks that exceed the maximum length.
                    - **Impact**: Enabling truncation ensures that text exceeding the max length is cut off, preventing errors. Disabling truncation might cause the model to fail if the text is too long.

                    - **Padding**: Choose padding strategy (`max_length` to pad all sequences to the same length).
                    - **Impact**: Padding to `max_length` ensures uniform input sizes, which can stabilize model performance. Dynamic padding can save memory when input sizes vary.

                    - **Max Length**: Set the maximum length for text sequences.
                    - **Impact**: Shorter max length reduces memory usage and processing time but may cut off relevant information. Longer max length captures more context but requires more resources.

                    - **Batch Size**: Define the number of sequences processed at once.
                    - **Impact**: Larger batch sizes improve throughput but require more memory. Smaller batch sizes reduce memory usage but may increase processing time.

                    - **Return Tensors**: Select the tensor type for returned outputs (`pt`, `tf`, `np`).
                    - **Impact**: `pt` for PyTorch, `tf` for TensorFlow, and `np` for NumPy. Choose based on your preferred framework for further processing.

                    - **Add Special Tokens**: Include special tokens in the tokenized sequences.
                    - **Impact**: Special tokens like `[CLS]` and `[SEP]` are necessary for many models to function correctly. Disabling this might be useful for custom tokenization schemes.

                    - **Stride**: Specify the stride for overlapping text chunks.
                    - **Impact**: A non-zero stride helps in creating overlapping chunks, which can preserve context between chunks but increases the number of chunks processed.

                    - **Is Split Into Words**: Determine if input text is split into words.
                    - **Impact**: Enable for word-level tokenization, useful for languages or tasks where preserving word boundaries is crucial.

                    - **Return Attention Mask**: Enable the return of attention masks.
                    - **Impact**: Necessary for models that require attention masks to ignore padding tokens. Disabling can save memory if attention masks are not needed.

                    - **Return Token Type IDs**: Include token type IDs in the output.
                    - **Impact**: Needed for models handling multiple segments like question-answer pairs. Disabling reduces the size of the returned data.

                    - **Return Length**: Return the length of each sequence.
                    - **Impact**: Useful for debugging and analysis, enabling this provides additional information about each sequence's length.

                    - **Verbose**: Enable verbose output for debugging.
                    - **Impact**: Helpful during development for detailed logs, but can be disabled in production to reduce log clutter.

                    - **Use Fast Tokenizer**: Use the fast version of the tokenizer.
                    - **Impact**: Fast tokenizers significantly speed up tokenization with minimal differences in output, recommended for most cases.

                    - **Add Prefix Space**: Add a space before the first token.
                    - **Impact**: Useful for certain tokenizers that require space before special tokens. Often used with GPT-2 and similar models.

                    - **Do Lower Case**: Convert all characters to lowercase.
                    - **Impact**: Ensures uniformity in case-sensitive tasks. Disable if case information is important.

                    - **Strip Accents**: Remove accents from characters.
                    - **Impact**: Simplifies text by removing accents, useful for languages where accents are not critical to meaning.

                    - **Do Basic Tokenize**: Perform basic tokenization before wordpiece tokenization.
                    - **Impact**: Basic tokenization splits text into words and punctuation, which is useful for many NLP tasks. Disabling may be useful for pre-tokenized text.

                    - **Never Split**: List of tokens that should never be split.
                    - **Impact**: Specify tokens like names or special phrases that should remain intact during tokenization. E.g., `["JohnDoe", "New York"]`.

                    - **Pad Token**: Define the token used for padding.
                    - **Impact**: The padding token can be customized to fit the model's requirements. Common options are `[PAD]`, `0`, or any other token that fits the model's vocabulary.

                    - **Normalize Unicode**: Convert all characters to Unicode NFKC form.
                    - **Impact**: Ensures consistency in text representation by normalizing characters.

                    - **Remove Stop Words**: Remove common stop words from the text.
                    - **Impact**: Reduces noise in the text, potentially improving the focus on meaningful words.

                    - **Lemmatize**: Reduce words to their base or root form.
                    - **Impact**: Ensures consistency by treating different forms of a word as the same token.

                    - **Dynamic Padding**: Use dynamic padding based on the batch's maximum sequence length.
                    - **Impact**: Saves memory and reduces padding noise by padding only to the length of the longest sequence in a batch.

                    - **Synonym Replacement**: Replace words with their synonyms.
                    - **Impact**: Increases the diversity of the training data by introducing variations in wording.

                    - **Back Translation**: Translate text to another language and back to the original language.
                    - **Impact**: Generates paraphrases of the text, further diversifying the training data.

                    - **Cross Encoder Model**: Use a cross-encoder model for embeddings.
                    - **Impact**: Considers pairs of sentences simultaneously, producing more context-aware embeddings.

                    - **Pooling Strategy**: Choose the pooling strategy for generating embeddings.
                    - **Mean Pooling**: Averages the token embeddings, providing a balanced representation of the text.
                    - **Max Pooling**: Selects the maximum value across the token embeddings, capturing the most salient features.
                    - **Weighted Pooling**: Uses learned weights to combine token embeddings, potentially improving representation by emphasizing important tokens.

                    - **Fine Tune**: Fine-tune the model on domain-specific data.
                    - **Impact**: Improves the model's performance on specific tasks by adapting it to the nuances of the target domain.

                    - **Context Window Size**: Specify the size of the context window for text processing.
                    - **Impact**: Ensures that important context is captured across chunk boundaries.

                    - **Context Window Stride**: Specify the stride for the context window.
                    - **Impact**: Controls the overlap between context windows, balancing context retention and processing efficiency.

                    - **Correct Spelling Grammar**: Enable spelling and grammar correction.
                    - **Impact**: Ensures that the input text is clean and standardized before processing.

                    - **Custom Pretrained Embeddings**: Use custom pre-trained embeddings.
                    - **Impact**: Boosts performance for specific tasks by leveraging embeddings pre-trained on relevant corpora.

                    ### Changing Parameters

                    To change the parameters, navigate to the "Settings" tab in the interface. Adjust the settings as needed for your specific use case, keeping the impacts mentioned above in mind. 

                    - **Example Use Case**: If you're processing a document that includes names and places, add those names and places to the `never_split` parameter to ensure they remain intact.
                    - **Custom Pad Token**: If your model uses a different token for padding, set the `pad_token` parameter accordingly.

                    By understanding and adjusting these settings, you can tailor the text processing to better fit your needs, improving the efficiency and accuracy of your embeddings.

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
                normalize_unicode = gr.Checkbox(value=params['normalize_unicode'], label='Normalize Unicode')
                remove_stop_words = gr.Checkbox(value=params['remove_stop_words'], label='Remove Stop Words')
                lemmatize = gr.Checkbox(value=params['lemmatize'], label='Lemmatize')
                dynamic_padding = gr.Checkbox(value=params['dynamic_padding'], label='Dynamic Padding')
                synonym_replacement = gr.Checkbox(value=params['data_augmentation']['synonym_replacement'], label='Synonym Replacement')
                back_translation = gr.Checkbox(value=params['data_augmentation']['back_translation'], label='Back Translation')
                cross_encoder_model = gr.Textbox(value=params['cross_encoder_model'], label='Cross Encoder Model')
                pooling_strategy = gr.Dropdown(choices=pooling_choices, value=params['pooling_strategy'], label='Pooling Strategy')
                fine_tune = gr.Checkbox(value=params['fine_tune'], label='Fine Tune')
                context_window_size = gr.Number(value=params['context_window_size'], label='Context Window Size')
                context_window_stride = gr.Number(value=params['context_window_stride'], label='Context Window Stride')
                correct_spelling_grammar = gr.Checkbox(value=params['correct_spelling_grammar'], label='Correct Spelling Grammar')
                custom_pretrained_embeddings = gr.Textbox(value=params['custom_pretrained_embeddings'], label='Custom Pretrained Embeddings')
                update_settings = gr.Button('Apply changes')

            with gr.Tab("Manage Embeddings"):
                delete_embeddings_btn = gr.Button('Delete embeddings')
                generate_preset_btn = gr.Button('Generate preset')
                delete_preset_btn = gr.Button('Delete preset')

            last_updated = gr.Markdown()

    update_file.click(feed_file_into_collector, [file_input, chunk_len, chunk_sep], last_updated, show_progress=False)
    fetch_embeddings_btn.click(fetch_embeddings, [], last_updated, show_progress=False)
    update_settings.click(apply_settings, [
        model_name, chunk_len, truncation, padding, max_length, batch_size, return_tensors, add_special_tokens, stride, 
        is_split_into_words, return_attention_mask, return_token_type_ids, return_length, verbose, use_fast, add_prefix_space, 
        do_lower_case, strip_accents, do_basic_tokenize, never_split, pad_token, normalize_unicode, remove_stop_words, lemmatize, 
        dynamic_padding, synonym_replacement, back_translation, cross_encoder_model, pooling_strategy, fine_tune, context_window_size, 
        context_window_stride, correct_spelling_grammar, custom_pretrained_embeddings
    ], last_updated, show_progress=False)
    delete_embeddings_btn.click(delete_embeddings_and_respond, [], last_updated, show_progress=False)
    generate_preset_btn.click(save_preset, [], last_updated, show_progress=False)
    delete_preset_btn.click(delete_preset, [], last_updated, show_progress=False)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
