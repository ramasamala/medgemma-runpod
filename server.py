from transformers import AutoProcessor, AutoModelForImageTextToText
from flask import Flask, request, jsonify
import torch, base64, io, os, time
from PIL import Image

os.environ['HF_HOME'] = '/workspace/model-cache'
token = os.environ.get('HF_TOKEN')

print('Loading MedGemma 4B...')
processor = AutoProcessor.from_pretrained('google/medgemma-4b-it', token=token)
model = AutoModelForImageTextToText.from_pretrained('google/medgemma-4b-it', torch_dtype=torch.bfloat16, token=token, device_map='auto')
print(f'Model loaded on {model.device}')

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    return response

@app.route('/medgemma', methods=['OPTIONS'])
def medgemma_options():
    return '', 204

DEFAULT_PROMPT = (
    "You are an expert medical imaging analyst. "
    "Analyze this medical image in detail. Describe the imaging modality, "
    "anatomical region, any notable findings, abnormalities, or observations. "
    "Provide a structured assessment."
)

@app.route('/medgemma', methods=['POST'])
def analyze():
    try:
        data = request.json
        image_b64 = data.get('image_base64')
        prompt = data.get('prompt', DEFAULT_PROMPT)
        max_tokens = data.get('max_tokens', 1024)

        if not image_b64:
            return jsonify({'error': 'image_base64 is required'}), 400

        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert('RGB')

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        t0 = time.time()
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
        inference_ms = int((time.time() - t0) * 1000)

        generated = output_ids[0][inputs['input_ids'].shape[-1]:]
        text = processor.decode(generated, skip_special_tokens=True)

        return jsonify({'analysis': text, 'inference_time_ms': inference_ms, 'model': 'google/medgemma-4b-it'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
