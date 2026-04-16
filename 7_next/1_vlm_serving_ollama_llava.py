import base64
from io import BytesIO
from PIL import Image
import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# In terminal, run ollama pull moondream 
MODEL_NAME = "llava" # moondream, llava model is better than small VLM like moondream.

llm = ChatOllama(model=MODEL_NAME, temperature=0)

def pil_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")    # IPL image to Base64 string (for Ollama API spec)

def app_interface(image, prompt):
    if image is None:
        return "Upload image."
    if not prompt:
        prompt = ""
        
    try:
        image_b64 = pil_to_base64(image)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ]
        )
        
        # Request message to Ollama and get response
        response = llm.invoke([message])
        return response.content
        
    except Exception as e:
        return f"Error: {str(e)}\n\n(Tip: Make sure Ollama is running in the background and the '{MODEL_NAME}' model is pulled.)"

prompt_examples = """
### Test Prompts
* **Describe the image**: Provide a general description.
* **Find the chair**: Locate the chair in the image.
* **What is on the floor?**: Identify objects on the floor.
* **How many people are there?**: Count the number of people in the image.
* **What is the person doing?**: Describe the actions of the person in the image.
* **What is the color of the car?**: Identify the color of the car in the image.
"""

with gr.Blocks() as demo:
    gr.Markdown(f"# Image Analysis ({MODEL_NAME})")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload image")
            prompt_input = gr.Textbox(
                label="Query", 
                placeholder="Enter prompt here",
                value="Describe the image."
            )
            gr.Markdown(prompt_examples)
            submit_btn = gr.Button("Start Analysis", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Result", lines=10)
            
    submit_btn.click(
        fn=app_interface,
        inputs=[image_input, prompt_input],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(share=False)