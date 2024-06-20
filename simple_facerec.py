import face_recognition
import cv2
import os
import glob
import numpy as np
from Model.database import get_database

class SimpleFacerec:
    def __init__(self):
        self.db = get_database()
        self.faces_collection = self.db['faces']
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        self.load_known_faces_from_db()

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images_path)} encoding images found.")

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            face_encodings = face_recognition.face_encodings(rgb_img)
            if face_encodings:
                img_encoding = face_encodings[0]

                self.faces_collection.insert_one({
                    'name': filename,
                    'encoding': img_encoding.tolist()
                })
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)
                print(f"Loaded encoding for {filename}")
            else:
                print(f"No faces found in {filename}, skipping.")
        print("Encoding images loaded")

    def load_known_faces_from_db(self):
        faces = self.faces_collection.find()
        for face in faces:
            self.known_face_encodings.append(np.array(face['encoding']))
            self.known_face_names.append(face['name'])

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names