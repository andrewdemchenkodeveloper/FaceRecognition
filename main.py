import os
import face_recognition

from PIL import Image
from time import time

#start timer
start_time = time()
print ('Start recognition')

all_persons = []

persons_folder = './persons'
photos_folder = './photos'
results_folder = './results'

# encode person faces
for filename in os.listdir(persons_folder):
    person = face_recognition.load_image_file(os.path.join(persons_folder, filename))
    person_encoding = face_recognition.face_encodings(person)[0]

    all_persons.append(person_encoding)

# find and save similar faces
for filename in os.listdir(photos_folder):
    for people in all_persons:
        unknown_picture = face_recognition.load_image_file(os.path.join(photos_folder, filename))
        unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

        results = face_recognition.compare_faces([people], unknown_face_encoding)

        if results[0]:
            unknown_picture = Image.fromarray(unknown_picture)
            unknown_picture.save(os.path.join(results_folder, filename))
        else:
            pass

#end timer
end_time = time() - start_time
print ('Recognition ended in {} seconds'.format(round(end_time, 2)))
