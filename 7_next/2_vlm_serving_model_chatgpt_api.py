import os
import base64
import json
import re
from io import BytesIO
from PIL import Image, ImageDraw
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0,
    max_tokens=2048
)

def pil_to_base64(img: Image.Image) -> str:
    max_size = 1024
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_document(image, prompt):
    if image is None: return None, "Upload image"

    if not prompt:
        prompt = """
        이 이미지(그림/문서/영수증/웹페이지)를 분석해서 다음을 완벽하게 수행해 줘:
        
        1. [시각적 요소 묘사]: 이미지에 있는 그림, 로고, 표, 사람 등 텍스트가 아닌 시각적 요소가 무엇인지 설명해 줘.
        2. [OCR 추출]: 이미지에 있는 모든 텍스트를 레이아웃에 맞게 그대로 추출해 줘.
        ```
        """

    try:
        image_b64 = pil_to_base64(image)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        )
        
        response = llm.invoke([message])
        output_text = response.content

        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)
        
        return output_text
        
    except Exception as e:
        return image, f"Error occurred during API call: {str(e)}"

prompt_examples = """
### 실무 프롬프트 예시
* **이 이미지(그림/문서/영수증/웹페이지)를 분석:** 위 입력창을 비워두면 자동으로 실행됩니다.
* **JSON 추출:** 이 영수증에서 `{"상호명": "", "결제총액": "", "결제일자": ""}` 형태로 데이터만 뽑아줘.
"""

with gr.Blocks() as demo:
    gr.Markdown("# Smart document analysis (GPT-4o)")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload document image")
            prompt_input = gr.Textbox(
                label="Command (leave empty for automatic OCR detection)", 
                placeholder="e.g., Extract only the store name and price in JSON format",
                lines=3
            )
            gr.Markdown(prompt_examples)
            submit_btn = gr.Button("Start Analysis", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Markdown(label="Analysis Result")
            
    submit_btn.click(
        fn=analyze_document,
        inputs=[image_input, prompt_input],
        outputs=[output_text]
    )

if __name__ == "__main__":
    demo.launch(share=False)