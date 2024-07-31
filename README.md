# BabyRag

## Introduction
Welcome to BabyRag, part of my submission for the Hackster.io AMD Pervasive AI Contest 2023 - [AI.You](https://www.hackster.io/contests/amd2023/hardware_applications/17172). This project is designed to create a coherent RAG (Retrieval-Augmented Generation) style infrastructure, inspired by the ambitions of superboogav2 for the Oobabooga Web-UI. Using cutting-edge AI and AMD’s powerful processing capabilities, BabyRag aims to enhance text processing applications with advanced text embedding and retrieval techniques.

## Features
- **Text Embedding and Retrieval**: Utilizes BERT models for generating and retrieving text embeddings.
- **Enhanced Interaction**: Implements a Flask-based web interface for dynamic interaction with the AI.
- **Real-time Processing**: Leverages the computing power of AMD hardware for efficient data handling and processing.

## Hardware and Software Requirements
### Hardware:
- AMD Ryzen processors (recommended for optimal performance)
### Software:
- Python 3.8+
- PyTorch, Transformers, Flask
- Other dependencies listed in `requirements.txt`

## Installation Instructions
Preinstall: Configure your AMD hardware. For NPU users, this means rebooting into UEFI and enabling the NPU.
1. Install [Oobabooga's text-generation-webUI](https://github.com/oobabooga/text-generation-webui).
2. Clone the repository in your extensions folder: `git clone https://github.com/your-repository/BabyRag.git`
3. Activate the .env (usually cmd_os.bat) for windows users, its CMD_windows.bat.
4. CD: navigate to extensions/Baby Rag and Install required Python packages: `pip install -r requirements.txt`
5. Reopen the webui at localhost:7860
6. Navigate to the settings tab and make sure BabyRag is selected. Then select 'Save UI Defaults...' ![image](https://github.com/user-attachments/assets/8d846909-9177-496b-b22d-ada1910b56d0)
7. Finally, select apply changes and restart

## Usage
To start the server and upload your files, very similarly to how you would do with superbooga/''V2. You don't need to fetch your data, but if you do it generates a cool json with your embeddings. Oobabooga's Web-UI has custom chat generation
only work for 'chat' mode, but I plan to expand this as I become more familiar with the infrastructure.

Load a model: If you're following along with an AMD NPU, consider using a model with these parameters to keep generation time reasonable.
![image](https://github.com/user-attachments/assets/61907668-6d46-4989-9b61-feebb8a38b52)

I would also keep your file size at most 100k if you're not using a GPU or have any type of GPU accelleration. The model attempts to preprocess your text but summarizing it yourself or getting another GPT to could give quicker generations but possibly at the risk of lowered performance.

Once the extension loads, navigate to the chat page. Under 'Generate' you'll see:
![image](https://github.com/user-attachments/assets/be48a231-a705-4904-af8d-73b4692d102d)
![image](https://github.com/user-attachments/assets/d2f2feb8-24c7-4c6c-a69f-605ee2be9189)

Select your file and hit: Load Data
From there, make sure you have 'chat' selected (may not work with chat-instruct or instruct) and begin talking.

## Load Data 

Load Data lets you parse your embeddings as a .json. This is helpful if you find a file isn't parsing properly.
![image](https://github.com/user-attachments/assets/0a3ed221-aec7-4ae2-aa58-459cfe70c5ab)


