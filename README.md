# 🚀 BabyRag 2.0

## 🌟 Introduction
Welcome to **BabyRag 2.0**, an enhanced version of the original BabyRag project. This iteration brings significant improvements, including the use of ONNX models for efficient inference, integration with Gradio for an interactive web interface, and enhanced hardware acceleration capabilities leveraging AMD's powerful processing units. BabyRag continues to provide a coherent RAG (Retrieval-Augmented Generation) style infrastructure, enhancing text processing applications with advanced text embedding and retrieval techniques.

## 🔥 Features
- **📝 Text Embedding and Retrieval**: Utilizes ONNX models for generating and retrieving text embeddings, providing faster inference times.
- **💻 Interactive Gradio Interface**: Implements a Gradio-based web interface for dynamic interaction with the AI.
- **⚡ Hardware Acceleration**: Leverages AMD hardware, including support for GPU acceleration, for efficient data handling and processing.
- **🔄 Model Management**: Introduces model selection and loading/unloading capabilities within the interface.
- **🔧 Hardware Backend Selection**: Automatically selects the optimal hardware backend (CPU or GPU) for running the models.

## 🛠️ Hardware and Software Requirements
### 🖥️ Hardware:
- AMD Ryzen processors or AMD GPUs (recommended for optimal performance)

### 🧰 Software:
- Python 3.8+
- PyTorch
- Transformers
- ONNX Runtime
- Gradio
- Other dependencies listed in `requirements.txt`

## 📥 Installation Instructions
_Preinstall: Configure your AMD hardware. For NPU users, this means rebooting into UEFI and enabling the NPU._

1. Clone the repository: `git clone https://github.com/your-repository/BabyRag.git`
2. Navigate to the project directory: `cd BabyRag`
3. Install the required Python packages: `pip install -r requirements.txt`
4. Run the Gradio app: `python babyrag.py`
5. Access the web interface at `http://localhost:1234` (or the port specified in the code).

## 🚀 Usage

### Starting the Application
- Run `python babyrag.py` to start the Gradio interface.
- The application will automatically detect your hardware capabilities and select the appropriate backend (CPU or GPU).

### Selecting and Loading Models
1. **Select Model Folder**: Click on **"Select Model Folder"** to choose the directory where your ONNX models are stored.
2. **Load a Model**:
   - Choose a model from the **"Available Models"** dropdown.
   - Click **"Load Model"** to load the selected model.
   - The status textbox will display whether the model was loaded successfully.

### Generating Embeddings
1. Navigate to the **"Embeddings Generator"** section.
2. Enter the text you want to generate embeddings for in the **"Input Text"** textbox.
3. Click **"Generate Embeddings"** to obtain the embeddings.
4. The embeddings will be displayed in the **"Generated Embeddings"** output section.

### Chatting with the Model
1. Scroll down to the **"Chat with the Model"** section.
2. In the **"Your Message"** textbox, type your message.
3. Click **"Send"** to interact with the model.
4. The conversation will appear in the chatbot interface.

## 🧠 Model Zoo
[Try Out A Model Today](https://huggingface.co/) (Ensure that your models are converted to ONNX format and stored in the selected model folder.)

## 🔧 Hardware Backend Selection
The application includes a **HardwareManager** that automatically selects the optimal backend for running the models:

- **CPU**: Default backend if no GPU is detected.
- **GPU**: If a compatible GPU is detected (e.g., CUDA-compatible GPU), the application will use it for acceleration.

*Note*: Support for AMD NPUs or other hardware accelerators may require additional configuration or updates.

## 🐞 Known Issues
1. **Hardware Acceleration Limitations**: Currently, the application supports CPU and CUDA-compatible GPUs. AMD GPU acceleration via ROCm or support for AMD NPUs is not yet implemented.
2. **Model Compatibility**: Ensure that your models are converted to ONNX format and are compatible with the ONNX Runtime.
3. **Large Models and Texts**: Processing very large models or lengthy texts may result in increased memory usage and slower performance.

## 🙌 Special Thanks
- **AMD and Hackster.io** for sponsoring this contest!
- **Oobabooga** for making this web-UI so easy to work with.
- **FuriousPandas** 🐼 for teaching me everything I know.
- **My Wonderful Husband**, whose unwavering support helped me get through this project with ease. ❤️

## Proof of Concept
All of the following images are produced by this RAG concept alone, albeit each one using a different methodology or even a different tokenizer model.

![image](https://github.com/user-attachments/assets/8a918bc2-284b-47fe-815b-5543d5754371)
![image](https://github.com/user-attachments/assets/fa69a05e-d81f-45b0-af31-fcfd2c659f34)
![image](https://github.com/user-attachments/assets/ec5c2a3d-b4b2-49ba-ad30-6f68da804f6d)
![image](https://github.com/user-attachments/assets/06f25929-03d0-466b-baff-42aa0db36d19)

### Current Features
![image](https://github.com/user-attachments/assets/7e1d7a9c-2d69-4713-a23f-9be9ce4cac3c)
