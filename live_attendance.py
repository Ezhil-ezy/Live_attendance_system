import os
import cv2
import numpy as np 
import face_recognition
from datetime import datetime
'''
face_detection were done with the help face_recognition and for measurements numpy are handy tools.
Note :  it only shows best matches based on the accuracy of the measures.
!!! too long distance to be avoided for the webcam drawn pictures.

'''
# file_path.
path = 'face_venv/fd_images'

images = [] # empty_list_at_beginning
names_ = [] # filtered_by_file_name.and
mylist = os.listdir(path) # passing_path_of_the_folder.
print(path)
# print(mylist)

# to_get_all_the_images_and_file_names
for image in mylist:
  current_images = cv2.imread(f'{path}/{image}')
  images.append(current_images)
  names_.append(os.path.splitext(image)[0])
print(names_)

def find_encodings(images):
  encode_list = []
  for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(img)[0]
    encode_list.append(enc)
  return encode_list

encode_list_known = find_encodings(images)
print('encoding_completed.....')

def mark_attendance(name):
  with open('face_venv/attendance.csv', 'r+') as f:
    my_data_list = f.readlines()
    name_list = []
    for line in my_data_list:
      entry = line.split(',')
      name_list.append(entry[0])
    
    if name not in name_list:
      now = datetime.now()
      date_string = now.strftime('%H:%M:%S')
      f.writelines(f'\n{name}, {date_string}')


web_cam = cv2.VideoCapture(0)

while True:

  success, img = web_cam.read() # to_read_an_image
  img_sze = cv2.resize(img, (0, 0), None, 0.25, 0.25)
  img_sze = cv2.cvtColor(img_sze, cv2.COLOR_BGR2RGB)

  face_frame = face_recognition.face_locations(img_sze)
  enc_frame = face_recognition.face_encodings(img_sze, face_frame)

  for encode_face, face_location in zip(enc_frame, face_frame):

    matches = face_recognition.compare_faces(encode_list_known, encode_face)
    facedis = face_recognition.face_distance(encode_list_known, encode_face)
    print(facedis)

    match_index = np.argmin(facedis)

    if matches[match_index]:
      name = names_[match_index].upper()
      print(name)
      y1, x2, y2, x1 = face_location
      y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 
      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) , 1)
      cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
      cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
      
      mark_attendance(name)

  cv2.imshow('face_detection', img)
  cv2.waitKey(0)


