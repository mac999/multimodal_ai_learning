import os
import base64
import json
import re
from io import BytesIO
from PIL import Image, ImageDraw
import gradio as gr

# LangChain의 최신 Google GenAI 모듈. pip install langchain-google-genai gradio pillow google-generativeai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
print("사용 가능한 모델 목록:")
for m in genai.list_models():
	if 'generateContent' in m.supported_generation_methods:
		print(m.name)

llm = ChatGoogleGenerativeAI(
	model="gemini-3.1-flash-lite-preview", # gemini-2.5-flash, gemini-2.5-flash-lite, gemini-3.1-flash-lite-preview
	temperature=0,
	max_tokens=2048
)

def pil_to_base64(img: Image.Image) -> str:
	max_size = 512
	if max(img.size) > max_size:
		img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
		
	buffered = BytesIO()
	img.save(buffered, format="JPEG", quality=85)
	return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_document(image, prompt):
	if image is None: return None, "Upload image"
	
	if not prompt:
		prompt = """
Analyze this image and perform the following tasks perfectly:
1. [Describe Visual Elements]: Describe the content of the image.
2. [Detect Objects]: Find the locations of major objects (logos, signatures, key paragraphs, etc.) within the image.
Organize the results in a visually appealing Markdown format, and **be sure to include the following JSON block at the end of the result:**
All box_2d coordinate values ​​must be output as ratios of integers between 0 and 1000, regardless of resolution.
```json
{
	"bboxes": [
	{"label": "Object", "box_2d": [ymin, xmin, ymax, xmax]}
	]
}
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
		response_text = response.content
		output_text = response_text[0]['text']

		output_image = image.copy()
		draw = ImageDraw.Draw(output_image)
		width, height = output_image.size
		
		# 정규표현식으로 마크다운 내의 JSON 블록 추출
		match = re.search(r'```json\s*(.*?)\s*```', output_text, re.DOTALL)
		if match:
			try:
				data = json.loads(match.group(1))
				if "bboxes" in data:
					for obj in data["bboxes"]:
						ymin, xmin, ymax, xmax = obj["box_2d"]
						label = obj.get("label", "Object")
						x1 = int(xmin / 1000 * width)
						y1 = int(ymin / 1000 * height)  
						x2 = int(xmax / 1000 * width)
						y2 = int(ymax / 1000 * height)
						
						draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
						draw.text((x1, max(0, y1 - 15)), label, fill="red", height=150)
			except Exception as e:
				print(f"JSON 파싱/Bbox 렌더링 에러: {e}")

		return output_image, output_text
		
	except Exception as e:
		return image, f"Error occurred during API call: {str(e)}"

prompt_examples = """
### Practical Prompt Examples
* **Image/Text/BBOX Separation (Default):** If you leave the input field above blank, it will run automatically.
* **JSON Extraction:** Extract only the data from this receipt in the format `{"Business Name": "", "Total Payment Amount": "", "Payment Date": ""}`.
* **Specific Object Detection:** Find only the locations of human faces in the image and set them to `box_2d` JSON.
"""

with gr.Blocks() as demo:
	gr.Markdown("# Smart document analysis & Object Detection")
	
	with gr.Row():
		with gr.Column(scale=1):
			image_input = gr.Image(type="pil", label="Upload document image")
			prompt_input = gr.Textbox(
				label="Command (leave empty for automatic OCR & BBox detection)", 
				placeholder="e.g., Extract only the store name and price in JSON format",
				lines=3
			)
			gr.Markdown(prompt_examples)
			submit_btn = gr.Button("Start Analysis", variant="primary")
			
		with gr.Column(scale=1):
			output_image = gr.Image(type="pil", label="Detected Objects")
			output_text = gr.Markdown(label="Analysis Result")
			
	submit_btn.click(
		fn=analyze_document,
		inputs=[image_input, prompt_input],
		outputs=[output_image, output_text]
	)

if __name__ == "__main__":
	demo.launch(share=False)