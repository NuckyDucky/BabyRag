# 🚀 BabyRag

## 🌟 Introduction
Welcome to **BabyRag**, part of my submission for the Hackster.io AMD Pervasive AI Contest 2023 - [AI.You](https://www.hackster.io/nicholasbscott92/ai-you-6aceaf). This project creates a coherent RAG (Retrieval-Augmented Generation) style infrastructure, inspired by the ambitions of superboogav2 for the Oobabooga Web-UI. Using cutting-edge AI and AMD’s powerful processing capabilities, BabyRag enhances text processing applications with advanced text embedding and retrieval techniques.

## 🔥 Features
- **📝 Text Embedding and Retrieval**: Utilizes BERT models for generating and retrieving text embeddings.
- **💻 Enhanced Interaction**: Implements a Flask-based web interface for dynamic interaction with the AI.
- **⚡ Real-time Processing**: Leverages the computing power of AMD hardware for efficient data handling and processing.
- **🔧 Advanced Tokenizer Features**: Supports advanced tokenizer features like lowercasing, stripping accents, basic tokenization control, custom token splits, and padding tokens.

## 🛠️ Hardware and Software Requirements
### 🖥️ Hardware:
- AMD Ryzen processors (recommended for optimal performance)
### 🧰 Software:
- Python 3.8+
- PyTorch, Transformers, Flask
- Other dependencies listed in `requirements.txt`

## 📥 Installation Instructions
_Preinstall: Configure your AMD hardware. For NPU users, this means rebooting into UEFI and enabling the NPU._

1. Install [Oobabooga's text-generation-webUI](https://github.com/oobabooga/text-generation-webui).
2. Clone the repository in your extensions folder: `git clone https://github.com/your-repository/BabyRag.git`
3. Activate the .env (usually `cmd_os.bat` for Windows users, `CMD_windows.bat`).
4. Navigate to `extensions/BabyRag` and install required Python packages: `pip install -r requirements.txt`
5. Reopen the webui at `localhost:7860`
6. Navigate to the settings tab and make sure BabyRag is selected. Then select 'Save UI Defaults...' ![Settings Image](https://github.com/user-attachments/assets/8d846909-9177-496b-b22d-ada1910b56d0) (I have openai and listen enabled so I can use this as a discord bot.)
7. Finally, select apply changes and restart.

## 🚀 Usage
Start the server and upload your files, similar to how you would with superboogav2. You don't need to fetch your data, but if you do, it generates a cool JSON with your embeddings. Oobabooga's Web-UI has custom chat generation that only works for 'chat' mode, but this will expand as familiarity with the infrastructure grows.

To emulate yourself, just upload a personal document containing a letter from you or written in your passive/active voice. Use your best roleplay model and ask it to imitate you and generate a letter/article etc.

### Load a Model
If you're using an AMD NPU, consider a model with these parameters for reasonable generation times.
![Model Parameters Image](https://github.com/user-attachments/assets/61907668-6d46-4989-9b61-feebb8a38b52)

### Tips:
- Keep your file size at most 100k if you're not using a GPU or have any GPU acceleration.
- Summarizing text before processing can yield quicker generations but might affect performance.

Once the extension loads, navigate to the chat page. Under 'Generate' you'll see:
![Generate Image 1](https://github.com/user-attachments/assets/be48a231-a705-4904-af8d-73b4692d102d)
![Generate Image 2](https://github.com/user-attachments/assets/d2f2feb8-24c7-4c6c-a69f-605ee2be9189)

Select your file and hit: **Load Data**. Ensure 'chat' is selected (may not work with chat-instruct or instruct) and begin talking.

## 📂 Load Data
Load Data lets you parse your embeddings as a `.json`. This is helpful if you find a file isn't parsing properly.
![Load Data Image](https://github.com/user-attachments/assets/0a3ed221-aec7-4ae2-aa58-459cfe70c5ab)

## 🧠 Model Zoo
[Try Out A Model Today](https://huggingface.co/)

## 🔧 Tokenizing
The BabyRag project includes advanced tokenizer features to enhance text preprocessing. Here is an explanation of each tokenizer setting:

- **model_name**: The name of the model to be used for generating embeddings, e.g., `'bert-base-uncased'`.
- **chunk_length**: Length of the text chunks to be processed at a time. This helps in managing large texts.
- **truncation**: If set to `True`, long sequences will be truncated to fit the `max_length`.
- **padding**: If set to `True`, sequences will be padded to the `max_length` with the specified `pad_token`.
- **max_length**: The maximum length of the tokenized sequences.
- **batch_size**: Number of sequences to be processed at a time.
- **return_tensors**: Specifies the format of the returned tensors (`'pt'` for PyTorch, `'tf'` for TensorFlow, `'np'` for NumPy).
- **add_special_tokens**: If set to `True`, special tokens like `[CLS]` and `[SEP]` will be added to the sequences.
- **stride**: The number of tokens to shift the window for overlapping chunks.
- **is_split_into_words**: If set to `True`, the input text is split into words.
- **return_attention_mask**: If set to `True`, returns the attention mask for the sequences.
- **return_token_type_ids**: If set to `True`, returns the token type IDs for the sequences.
- **return_length**: If set to `True`, returns the length of the sequences.
- **verbose**: If set to `True`, enables verbose logging during tokenization.
- **use_fast**: If set to `True`, uses the fast version of the tokenizer.
- **add_prefix_space**: If set to `True`, adds a space before each word during tokenization.
- **do_lower_case**: Converts all characters to lowercase before tokenization. Useful for models trained on lowercase data.
- **strip_accents**: Removes accents and other diacritical marks from text. This can be set to `None` (no stripping), `True` (strip accents), or `False` (do not strip accents).
- **do_basic_tokenize**: Determines whether to perform basic tokenization before applying wordpiece tokenization. Basic tokenization includes splitting text into words, handling punctuation, etc.
- **never_split**: A list of tokens that should never be split during tokenization. Useful for keeping certain tokens intact.
- **pad_token**: Specifies the token to be used for padding sequences to the same length. Commonly used pad tokens include `[PAD]`.

These settings provide fine-grained control over the tokenization process, enabling more precise and effective text preprocessing.

## 🐞 Known Issues
1. Sometimes the web-UI restarts too fast, causing multiple Gradio instances to conflict and throw errors. To avoid this, press `Ctrl+C` in the terminal and restart the server.
2. The data parsing is inefficient due to using cosine similarity and being unoptimized. If a file takes a while to load, consider lowering the context size for now. Future updates will include a smarter comparison feature.

## 🙌 Special Thanks
- **AMD and Hackster.io** for sponsoring this contest!
- **Oobabooga** for making this web-UI so easy to work with.
- **FuriousPandas** 🐼 for teaching me everything I know.
- **My Wonderful Husband**, whose unwavering support helped me get through this project with ease. ❤️

## Proof of Concept:
![image](https://github.com/user-attachments/assets/8a918bc2-284b-47fe-815b-5543d5754371)
![image](https://github.com/user-attachments/assets/fa69a05e-d81f-45b0-af31-fcfd2c659f34)
![image](https://github.com/user-attachments/assets/ec5c2a3d-b4b2-49ba-ad30-6f68da804f6d)
![image](https://github.com/user-attachments/assets/06f25929-03d0-466b-baff-42aa0db36d19)



