import os
import argparse
import face_recognition

from PIL import Image
from time import time

#start timer
start_time = time()
print ('Start recognition')

# add argument parser for images
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--person', help='path to name_surname.jpg file')
ap.add_argument('-f', '--folder', help='path to files folder')
args = vars(ap.parse_args())

name_folder = args['person'].split('.')

persons_folder = './persons'
photos_folder = args['folder']

try:
    results_folder = os.stat('{}_folder'.format(name_folder[0]))
except:
    results_folder = os.mkdir('{}_folder'.format(name_folder[0]))

# encode person face
person = face_recognition.load_image_file(os.path.join(persons_folder, args['person']))
person_encoding = face_recognition.face_encodings(person)[0]

# find and save similar faces
for filename in os.listdir(photos_folder):
    unknown_picture = face_recognition.load_image_file(os.path.join(photos_folder, filename))

    try:
        unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]
    except(IndexError):
        continue

    results = face_recognition.compare_faces([person_encoding], unknown_face_encoding)

    if results[0]:
        unknown_picture = Image.fromarray(unknown_picture)
        unknown_picture.save(os.path.join('{}_folder'.format(name_folder[0]), filename))
        print ('Find {} on {}'.format(name_folder[0], filename))
    else:
        pass

#end timer
end_time = time() - start_time
print ('Recognition ended in {} seconds'.format(round(end_time, 2)))
