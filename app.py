from flask import Flask, request, jsonify
from mangum import Mangum
import cv2
import numpy as np
from simple_facerec import SimpleFacerec
from Model.database import get_database

app = Flask(__name__)

# Initialize the face recognizer
sfr = SimpleFacerec()
sfr.load_encoding_images("Images/")

@app.route('/')
def hello():
    return "Server is running successfully"

@app.route('/scan', methods=['POST'])
def scan_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    face_locations, face_names = sfr.detect_known_faces(img)

    results = []
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        results.append({
            "name": name,
            "location": {"top": top, "right": right, "bottom": bottom, "left": left}
        })

    return jsonify(results)

handler = Mangum(app)

if __name__ == '__main__':
    app.run(debug=True)