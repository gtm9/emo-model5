from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
from flasgger import Swagger, swag_from
import time
import psutil
import config
import json
import os

# Load the Hugging Face pipeline for text generation
model_name = "SamLowe/roberta-base-go_emotions"
model_path = 'models/transformers/' # will be created automatically if not exists

model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
classifier.save_pretrained(model_path)


load_dotenv()

app = Flask(__name__)
CORS(app)

#Get the environment variables
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

@app.route("/")
def hello_world():
    return "Hello, World"

model_path = './models/transformers/'
model = TFAutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
print("----------- transformer model loaded ------------")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
print("----------- transformer tokenizer loaded ------------")
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, top_k=None)

@app.route('/predict', methods=[ 'POST'])
def calc_emo():
    data = request.json

    if 'text' not in data:
        return jsonify({'error': 'Missing "text" parameter'}), 400

    text = data['text']
    print(text)
    
    # Initialize CPU usage monitoring
    start_time = time.time()
    start_cpu = psutil.cpu_percent()

    # the classifier
    result = classifier(text)
    print(result)

    # Calculate time and CPU usage
    elapsed_time = time.time() - start_time
    cpu_percent = psutil.cpu_percent() - start_cpu

    # Calculate memory usage
    process = psutil.Process()
    memory_info = process.memory_info()

    memory_usage = f"Memory Usage (RSS): {memory_info.rss / 1024 / 1024:.2f} MB"
    time_elapsed = f"Time Elapsed: {elapsed_time:.2f} seconds"
    minutes_since_midnight = elapsed_time / 60
    minutes_since_midnight_string = f"Minutes since midnight: {minutes_since_midnight:.2f} minutes"
    cpu_usage = f"CPU Usage: {cpu_percent:.2f}%"

    return jsonify({
        'emotions': result,
        'Memory Usage (RSS)': memory_usage,
        'Time Elapsed': time_elapsed,
        'Minutes since midnight': minutes_since_midnight_string,
        'CPU Usage': cpu_usage
    })

@app.after_request
def after_request(response):
    if response and response.get_json():
        data = response.get_json()

        data["time_request"] = int(time.time())
        data["version"] = config.VERSION

        response.set_data(json.dumps(data))

    return response

@app.route("/version", methods=["GET"], strict_slashes=False)
def version():
  response_body = {
      "success": 1,
  }
  return jsonify(response_body)

if __name__ == "__main__":
    app.run()
