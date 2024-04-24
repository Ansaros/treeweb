import os
import torch
from PIL import Image
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'models/yolov5.pt'  
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'image' not in request.files:
        return 'No image file uploaded'
  
    image = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    img = Image.open(image_path)
    results = model(img)

    results.render()  
    results_img = Image.fromarray(results.render()[0])  

    output_image_filename = "result_" + image.filename
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_image_filename)
    results_img.save(output_image_path)
        
    os.remove(image_path)

    return render_template('result.html', image_path=output_image_filename)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)