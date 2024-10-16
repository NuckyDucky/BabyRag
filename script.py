import os
import json
import gradio as gr
import torch
import onnxruntime as ort
import logging
from transformers import AutoTokenizer
from tkinter import filedialog, Tk

logging.basicConfig(level=logging.DEBUG)

class HardwareManager:
    def __init__(self):
        self.backend = "cpu"  # Default to CPU

    def select_backend(self):
        try:
            if torch.cuda.is_available():
                self.backend = "cuda"
            else:
                self.backend = "cpu"
        except ImportError:
            self.backend = "cpu"

    def get_backend(self):
        return self.backend

class TextEmbedding:
    def __init__(self, model_path, hardware_manager):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.hardware_manager = hardware_manager
        self.backend = hardware_manager.get_backend()

        if self.backend == "cuda":
            # Load model using ONNX Runtime with CUDA support
            self.session = ort.InferenceSession(f"{model_path}/model.onnx", providers=["CUDAExecutionProvider"])
        else:
            self.session = ort.InferenceSession(f"{model_path}/model.onnx", providers=["CPUExecutionProvider"])

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='np', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_ids})
        embeddings = outputs[0].mean(axis=1).tolist()
        return embeddings

# Functions for model management
def select_model_folder():
    root = Tk()
    root.withdraw()  # Hide the root window
    folder_selected = filedialog.askdirectory()  # Open a dialog to select folder
    root.destroy()
    return folder_selected

def update_model_folder():
    global model_folder
    model_folder = select_model_folder()
    if model_folder:
        model_choices = [name for name in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, name))]
        return gr.Dropdown.update(choices=model_choices), f"Model folder selected: {model_folder}"
    else:
        return gr.Dropdown.update(choices=[]), "No folder selected."

def load_or_unload_model(model_name):
    global embedding_model, model_loaded
    if not model_loaded:
        if model_name:
            selected_model_path = os.path.join(model_folder, model_name)
            embedding_model = TextEmbedding(selected_model_path, hw_manager)
            model_loaded = True
            status = f"Model '{model_name}' loaded successfully."
            button_label = "Unload Model"
        else:
            status = "No model selected."
            button_label = "Load Model"
    else:
        # Unload the model
        embedding_model = None
        model_loaded = False
        status = "Model unloaded."
        button_label = "Load Model"
    return status, gr.Button.update(value=button_label)

def generate_embeddings_interface(text_input):
    if model_loaded and embedding_model:
        embeddings = embedding_model.get_embeddings(text_input)
        return embeddings
    else:
        return {"error": "Model not loaded"}

def generate_reply(prompt):
    # Placeholder function for generating replies
    if model_loaded and embedding_model:
        # Implement your model's reply generation logic here
        return f"Model response to: {prompt}"
    else:
        return "Model not loaded."

# Main execution
if __name__ == "__main__":
    hw_manager = HardwareManager()
    hw_manager.select_backend()
    model_loaded = False
    model_folder = ""
    embedding_model = None  # Initialize embedding_model

    # Allow the user to set the Gradio server port
    gradio_port = 1234  # Default port

    # Gradio-based web UI
    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
        gr.Markdown("<h1 style='text-align: center;'>BabyRag 2.0: Text Embedding and Chatbot Generator</h1>")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Management")
                select_folder_button = gr.Button("Select Model Folder")
                model_dropdown = gr.Dropdown(choices=[], label="Available Models")
                load_button = gr.Button("Load Model")
                status_text = gr.Textbox(value="", label="Status", interactive=False)
            with gr.Column(scale=2):
                gr.Markdown("### Embeddings Generator")
                text_input = gr.Textbox(label="Input Text")
                generate_button = gr.Button("Generate Embeddings")
                embeddings_output = gr.JSON(label="Generated Embeddings")
                generate_button.click(generate_embeddings_interface, inputs=text_input, outputs=embeddings_output)
        
        gr.Markdown("---")
        gr.Markdown("### Chat with the Model")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Your Message")
        send_button = gr.Button("Send")

        def chat_interface(user_message, history):
            if model_loaded and embedding_model:
                bot_response = generate_reply(user_message)
                history.append((user_message, bot_response))
                return history, ""
            else:
                history.append((user_message, "Model not loaded."))
                return history, ""

        send_button.click(chat_interface, inputs=[msg, chatbot], outputs=[chatbot, msg])

        # Button interactions
        select_folder_button.click(fn=update_model_folder, inputs=None, outputs=[model_dropdown, status_text])
        load_button.click(fn=load_or_unload_model, inputs=model_dropdown, outputs=[status_text, load_button])

    # Launch Gradio app
    demo.launch(server_name="0.0.0.0", server_port=gradio_port)
