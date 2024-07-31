# BabyRag

## Introduction
Welcome to BabyRag, part of my submission for the AMD Hackster.io Pervasive AI Contest 2024 - [AI.You](https://www.hackster.io/contests/amd2023/hardware_applications/17172). This project is designed to create a coherent RAG (Retrieval-Augmented Generation) style infrastructure, inspired by the ambitions of superboogav2 for the Oobabooga Web-UI. Using cutting-edge AI and AMD’s powerful processing capabilities, BabyRag aims to enhance text processing applications with advanced text embedding and retrieval techniques.

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
3. Install required Python packages: `pip install -r requirements.txt`

## Usage
To start the server and upload your files, very similarly to how you would do with superbooga/''V2. You don't need to fetch your data, but if you do it generates a cool json with your embeddings. Oobabooga's Web-UI has custom chat generation
only work for 'chat' mode, but I plan to expand this as I become more familiar with the infrastructure.
