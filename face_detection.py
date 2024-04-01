import cv2
import numpy as np 
import face_recognition

# first_image
img_ezhil = face_recognition.load_image_file('face_venv/fd_images/ezhil.png')
img_ezhil = cv2.cvtColor(img_ezhil, cv2.COLOR_BGR2RGB)

# test_image
img_test = face_recognition.load_image_file('face_venv/fd_images/test_.png')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# drawing_the_rect_or_encoding_the_face
face_location = face_recognition.face_locations(img_ezhil)[0]
encode_ezhil = face_recognition.face_encodings(img_ezhil)[0]
cv2.rectangle(img_ezhil, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (0, 255, 255), 1)
#print(face_location)

# drawing_the_rect_or_encoding_the_face
face_location_test = face_recognition.face_locations(img_test)[0]
encode_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (face_location_test[3], face_location_test[0]), (face_location_test[1], face_location_test[2]), (0, 255, 255), 1)

# comparing_the_faces___
compare = face_recognition.compare_faces([encode_ezhil], encode_test)
face_distance = face_recognition.face_distance([encode_ezhil], encode_test)
print(compare, face_distance)

cv2.putText(img_test, f'{compare} {round(face_distance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('ezhil', img_ezhil)
cv2.imshow('test', img_test)

cv2.waitKey(0)