# 🚀 BabyRag

## 🌟 Introduction
Welcome to **BabyRag**, part of my submission for the Hackster.io AMD Pervasive AI Contest 2023 - [AI.You](https://www.hackster.io/contests/amd2023/hardware_applications/17172). This project creates a coherent RAG (Retrieval-Augmented Generation) style infrastructure, inspired by the ambitions of superboogav2 for the Oobabooga Web-UI. Using cutting-edge AI and AMD’s powerful processing capabilities, BabyRag enhances text processing applications with advanced text embedding and retrieval techniques.

## 🔥 Features
- **📝 Text Embedding and Retrieval**: Utilizes BERT models for generating and retrieving text embeddings.
- **💻 Enhanced Interaction**: Implements a Flask-based web interface for dynamic interaction with the AI.
- **⚡ Real-time Processing**: Leverages the computing power of AMD hardware for efficient data handling and processing.

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
6. Navigate to the settings tab and make sure BabyRag is selected. Then select 'Save UI Defaults...' ![Settings Image](https://github.com/user-attachments/assets/8d846909-9177-496b-b22d-ada1910b56d0)
7. Finally, select apply changes and restart.

## 🚀 Usage
Start the server and upload your files, similar to how you would with superbooga/''V2. You don't need to fetch your data, but if you do, it generates a cool JSON with your embeddings. Oobabooga's Web-UI has custom chat generation that only works for 'chat' mode, but this will expand as familiarity with the infrastructure grows.

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
