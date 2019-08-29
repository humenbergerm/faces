import os
import os.path
import pickle
import csv
import numpy as np
import dlib
from collections import Counter, OrderedDict
import cv2
from shapely.geometry import Polygon, Point
import copy
import subprocess

def mkdir_p(path):
  if not os.path.isdir(path):
    os.makedirs(path)

def get_images_in_dir_rec(path_to_dir):
  files = []
  for root, dirnames, filenames in os.walk(os.path.normpath(path_to_dir)):
    for j in filenames:
      if j.lower().endswith(('.jpg', '.jpeg', '.png')):
        files.append(os.path.join(root, j))

  return files

def get_files_in_dir_by_name_rec(path_to_dir, filename):
  files = []
  for root, dirnames, filenames in os.walk(os.path.normpath(path_to_dir)):
    for j in filenames:
      if j == filename:
        files.append(os.path.join(root, j))

  return files

def is_valid_timestamp(date):
  if len(date) == 19 \
    and int(date[:4]) > 1900 \
    and int(date[5:7]) in range(1, 13) \
    and int(date[8:10]) in range(1, 32) \
    and int(date[11:13]) in range(0, 24) \
    and int(date[14:16]) in range(0, 60) \
    and int(date[17:19]) in range(0, 60):
    return True
  else:
    return False

def get_files_in_dir(path_to_dir, ext):
  files = []

  entries = os.listdir(os.path.normpath(path_to_dir))
  for f in entries:
      if os.path.isfile(os.path.join(path_to_dir,f)):
        if f.lower().endswith(ext):
            files.append(os.path.join(path_to_dir,f))

  return sorted(files)

def export_persons_to_csv(preds_per_person, preds_per_person_path):
  if len(preds_per_person) == 0:
    print('nothing to export, preds_per_person is empty')
    return
  for p in preds_per_person:
    if len(preds_per_person[p]) == 0:
      print('No faces found in {}. Files will be deleted.'.format(p))
      bin_file = os.path.join(preds_per_person_path, p + '.bin')
      if os.path.isfile(bin_file):
        os.remove(bin_file)
      csv_file = os.path.join(preds_per_person_path, p + '.csv')
      if os.path.isfile(csv_file):
        os.remove(csv_file)
    else:
      export_face_to_csv(preds_per_person_path, preds_per_person, p)

def export_face_to_csv(preds_per_person_path, preds_per_person, face):

  print('exporting {} ({})'.format(face, len(preds_per_person[face])))
  # save everything in one pickle file
  pkl_path = os.path.join(preds_per_person_path, face + '.bin')
  with open(pkl_path, "wb") as fp:
    pickle.dump(preds_per_person[face], fp)

  # open/create csv file
  csv_path = os.path.join(preds_per_person_path, face + '.csv')
  with open(csv_path, "w") as csvfile:
    filewriter = csv.writer(csvfile, delimiter=';')
    header = []
    header.append('name')
    header.append('location in image')
    header.append('image path')
    header.append('flag')
    header.append('timestamp')
    filewriter.writerow(header)
    for c in preds_per_person[face]:
      face_row = []
      for i in range(len(c)):
        if i == 0:
          face_row.append(c[i][0])
          # face_row.append(c[i][1])
          face_row.append(str(c[i][1][0]) + ' ' + str(c[i][1][1]) + ' ' + str(c[i][1][2]) + ' ' + str(c[i][1][3]))
          continue
        if i == 1:
          face_row.append(os.path.basename(c[i]))
          continue
        if i == 2:
          # skip the descriptor because it is saved in a separate file
          continue
        face_row.append(c[i])
      filewriter.writerow(face_row)

def load_faces_from_csv(preds_per_person_path):

  print('Loading the faces from {}.'.format(preds_per_person_path))

  if preds_per_person_path == None:
    print('path to db is needed')
    exit()
  if not os.path.isdir(preds_per_person_path):
    print('path to db is not a valid directory')
    exit()

  preds_per_person = {}

  csv_files = get_files_in_dir(preds_per_person_path, '.csv')
  bin_files = get_files_in_dir(preds_per_person_path, '.bin')
  if len(csv_files) == 0:
    print('no csv files found in {}'.format(preds_per_person_path))
    return preds_per_person
  for f in csv_files:
    name = os.path.splitext(os.path.basename(f))[0]
    print('loading {}'.format(name))
    descs = pickle.load(open(os.path.join(preds_per_person_path, name + '.bin'), "rb"))
    if 1:
      for i in bin_files:
        filename = os.path.splitext(os.path.basename(i))[0]
        # check if multiple .bin files should be combined
        # bin files can be combined if the 'other' files are exactly one character longer, e.g.: martin.bin and martin1.bin
        if name in filename and len(name) + 1 == len(filename):
          accept = input('Add ' + i + ' to ' + name + ' (y/n)?')
          if accept == 'y':
            tmp = pickle.load(open(i, "rb"))
            descs += tmp
    with open(f, 'r') as csvfile:
      filereader = csv.reader(csvfile, delimiter=';')
      preds_per_person[name] = []
      for i, row in enumerate(filereader):
        # skip first row (header)
        if i == 0:
          continue
        if len(row) == 1:
          row_str = row[0].split(';')
          face_loc = row_str[0], tuple(map(int, row_str[1].split(' ')))
          img_name = os.path.basename(row_str[2])
        elif len(row) == 5:
          face_loc = row[0], tuple(map(int, row[1].split(' ')))
          img_name = os.path.basename(row[2])
        else:
          print('invalid csv row format')
          continue
        if 0:  # use this if the order between the csv and the bin file(s) needs to be restored; e.g. after manual changes in the files
          found_face = find_face(descs, face_loc, img_name)
        else:
          found_face = descs[i - 1]
        if len(found_face) != 0:
          if os.path.isfile(found_face[1]):
            preds_per_person[name].append(
              ((name, face_loc[1]), found_face[1], found_face[2], found_face[3], found_face[4]))
          else:
            print('file {} does not exist'.format(found_face[1]))
        else:
          print('{} not found'.format(img_name))

  return preds_per_person

def get_folders_in_path(path):
  entries = os.listdir(os.path.normpath(path))
  dirs = []
  for f in entries:
    if os.path.isdir(os.path.join(path, f)):
      dirs.append(os.path.join(path, f))

  return dirs

def get_folders_in_dir_rec(path_to_dir):
  dirs = []
  for root, dirnames, filenames in os.walk(os.path.normpath(path_to_dir)):
    for j in dirnames:
      if os.path.isdir(os.path.join(root, j)):
        dirs.append(os.path.join(root, j))

  return dirs

def sort_folders_by_nr_of_images(folder_to_sort):
  dirs = get_folders_in_path(folder_to_sort)

  cnt = Counter()
  for d in dirs:
    num_of_images = len(get_images_in_dir(d))
    cnt[d] = num_of_images

  print("Number of images in folders: {}".format(sum(cnt.values())))

  cnt_ordered = OrderedDict(cnt.most_common())

  counter = 0
  for i in cnt_ordered:
    folder_name = os.path.join(folder_to_sort, str(counter) + "_nr_of_images_" + str(cnt_ordered[i]))
    os.rename(i, folder_name)
    # if (counter == 0):
    #    copytree(folder_name, folder_to_sort)
    counter += 1

def get_images_in_dir(path_to_dir):
  files = []

  entries = os.listdir(os.path.normpath(path_to_dir))
  for f in entries:
      if os.path.isfile(os.path.join(path_to_dir,f)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            files.append(os.path.join(path_to_dir,f))

  return files

def resizeCV(img, h):
  height, width = img.shape[:2]

  s = h / height
  height *= s
  width *= s

  return cv2.resize(img, (int(width), int(height)))

def is_valid_roi(x, y, w, h, img_shape):
  if x >= 0 and x < img_shape[1] and \
    y >= 0 and y < img_shape[0] and \
    w > 0 and h > 0 and \
    x + w < img_shape[1] and \
    y + h < img_shape[0]:
    return True
  return False

def detect_faces_in_image(img_path, detector, facerec, sp, use_entire_image=False):
  img = dlib.load_rgb_image(img_path)

  if use_entire_image:
    dets = [dlib.rectangle(0, 0, img.shape[1] - 1, img.shape[0] - 1)]
  else:
    dets = detector(img, 1)

  #print("Number of faces detected: {}".format(len(dets)))

  locations = []
  descriptors = []

  # process each face we found
  for k, d in enumerate(dets):
    # get the landmarks/parts for the face in box d.
    shape = sp(img, d)

    # face alignment is done automatically by dlib as pre-processing step in compute_face_descriptor
    # aligned_face = dlib.get_face_chip(img, shape)
    # cv2.imshow('aligned', aligned_face)
    # cv2.waitKey(0)

    # locations: list of tuples (t,r,b,l)
    # descriptors: list of float64 np arrays
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    descriptors.append(np.array(face_descriptor))
    locations.append((d.top(), d.right(), d.bottom(), d.left()))

  return locations, descriptors

def detect_faces_in_image_cv2(img_path, net, facerec, sp):
  opencvImage = cv2.imread(img_path)
  h, w = opencvImage.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(opencvImage, (300,300)), 1.0, (300, 300), (103.93, 116.77, 123.68))

  net.setInput(blob)
  detections = net.forward()

  locations = []
  descriptors = []
  for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (x1, y1, x2, y2) = box.astype("int")
      d = dlib.rectangle(x1, y1, x2, y2)
      shape = sp(opencvImage, d)
      face_descriptor = facerec.compute_face_descriptor(opencvImage, shape)
      descriptors.append(np.array(face_descriptor))
      locations.append((d.top(), d.right(), d.bottom(), d.left()))

  return locations, descriptors

def initialize_face_data(preds_per_person, cls):
  face_locations = []
  face_encodings = []
  for p in preds_per_person[cls]:
    face_locations.append(p[0][1])
    face_encodings.append(p[2])

  return face_locations, face_encodings

def delete_element_preds_per_person(preds_per_person, cls, ix):
  preds_per_person[cls].pop(ix)

def move_class(preds_per_person, cls, new_cls):
  accept = input('All members from class {} will be moved to class {}. (y/n)?'.format(cls, new_cls))
  if accept == 'y':
    for i, p in enumerate(preds_per_person[cls]):
      insert_element_preds_per_person(preds_per_person, cls, i, new_cls, 1)

    preds_per_person[cls] = []

def insert_element_preds_per_person(preds_per_person, cls, ix, new_cls, conf=-1):
  tmp = preds_per_person[cls][ix]
  if conf == -1:
    conf = tmp[3]
  if preds_per_person.get(new_cls) == None:
    preds_per_person[new_cls] = []
  preds_per_person[new_cls].append(((new_cls, tmp[0][1]), tmp[1], tmp[2], conf, tmp[4]))

def count_preds_status(preds_per_person):
  count_ignored = 0
  count_confirmed = 0
  for i in preds_per_person:
    if i[3] == 1:
      count_confirmed += 1
    if i[3] == 2:
      count_ignored += 1
  count_not_ignored = len(preds_per_person) - count_confirmed - count_ignored

  return count_confirmed, count_ignored, count_not_ignored

def resizeCV(img, w):
  height, width = img.shape[:2]

  s = w / width
  height *= s
  width *= s

  return cv2.resize(img, (int(width), int(height)))

def evaluate_key(args, key, preds_per_person, cls, ix, save, names, dets, det_file_map):
  if key == 99:  # key 'c'
    new_name = guided_input(preds_per_person)
    if new_name != "":
      save.append(copy.deepcopy(preds_per_person))
      # add pred in new list
      if preds_per_person.get(new_name) == None:
        preds_per_person[new_name] = []
      insert_element_preds_per_person(preds_per_person, cls, ix, new_name, 1)
      # delete pred in current list
      delete_element_preds_per_person(preds_per_person, cls, ix)
      print("face changed: {} ({})".format(new_name, len(preds_per_person[new_name])))
  elif key == 109:  # key 'm'
    new_name = guided_input(preds_per_person)
    if new_name != "":
      save.append(copy.deepcopy(preds_per_person))
      move_class(preds_per_person, cls, new_name)
      print("class moved: {} -> {}".format(cls, new_name))
  elif key == 117:  # key 'u'
    save.append(copy.deepcopy(preds_per_person))
    new_name = 'unknown'
    # add pred in new list
    if preds_per_person.get(new_name) == None:
      preds_per_person[new_name] = []
    insert_element_preds_per_person(preds_per_person, cls, ix, new_name)
    # delete pred in current list
    delete_element_preds_per_person(preds_per_person, cls, ix)
    print("face changed: {} ({})".format(new_name, len(preds_per_person[new_name])))
  elif key == 47:  # key '/'
    save.append(copy.deepcopy(preds_per_person))
    tmp = preds_per_person[cls][ix]
    if tmp[3] == 0:
      preds_per_person[cls][ix] = tmp[0], tmp[1], tmp[2], 1, tmp[4]
    elif tmp[3] == 1:
      preds_per_person[cls][ix] = tmp[0], tmp[1], tmp[2], 0, tmp[4]
    print("face confirmed: {} ({})".format(tmp[0], len(preds_per_person[cls])))
  elif key >= 48 and key <= 57:  # keys '0' - '9'
    save.append(copy.deepcopy(preds_per_person))
    new_name = names[key - 48]
    insert_element_preds_per_person(preds_per_person, cls, ix, new_name, 1)
    # delete pred in current list
    delete_element_preds_per_person(preds_per_person, cls, ix)
    print("face confirmed: {} ({})".format(new_name, len(preds_per_person[new_name])))
  elif key == 100:  # key 'd'
    if 1:
      save.append(copy.deepcopy(preds_per_person))
      new_name = 'deleted'
      # add pred in new list
      if preds_per_person.get(new_name) == None:
        preds_per_person[new_name] = []
      insert_element_preds_per_person(preds_per_person, cls, ix, new_name)
      # delete pred in current list
      delete_element_preds_per_person(preds_per_person, cls, ix)
    else:
      save.append(copy.deepcopy(preds_per_person))
      # delete face
      delete_element_preds_per_person(preds_per_person, cls, ix)
    print("face deleted")
  elif key == 116:  # key 't'
    subprocess.call(["open", "-R", preds_per_person[cls][ix][1]])
  elif key == 97:  # key 'a'
    # delete all faces of this class in the current image
    save.append(copy.deepcopy(preds_per_person))
    i = 0
    while i < len(preds_per_person[cls]):
      compare_path = preds_per_person[cls][i][1]
      if compare_path == preds_per_person[cls][ix][1]:
        delete_element_preds_per_person(preds_per_person, cls, i)
      else:
        i += 1
    # delete detections as well
    if len(dets) != 0:
      delete_detections_of_file(dets, preds_per_person[cls][ix][1])
      print("all faces in {} deleted".format(preds_per_person[cls][ix][1]))
    else:
      print('detections not deleted from detections.bin')
  elif key == 105:  # key 'i'
    # delete all faces of this class in the current image AND set it to be ignored in the future (also for detection)
    save.append(copy.deepcopy(preds_per_person))
    i = 0
    while i < len(preds_per_person[cls]):
      compare_path = preds_per_person[cls][i][1]
      if compare_path == preds_per_person[cls][ix][1]:
        delete_element_preds_per_person(preds_per_person, cls, i)
      else:
        i += 1
    # ignore detections in the future
    if len(dets) != 0:
      ignore_detections_of_file(dets, preds_per_person[cls][ix][1])
      # print("all faces in {} deleted and image will be ignored".format(image_path))
    else:
      print('detections not deleted from detections.bin')
  elif key == 115:  # key 's'
    export_persons_to_csv(preds_per_person, args.db)
    if args.dets != None:
      save_detections(dets, det_file_map)
    print('saved')

def get_rect_from_pts(pts, ws):
  top = min(pts[0][1], pts[1][1])
  left = min (pts[0][0], pts[1][0])
  bottom = max(pts[0][1], pts[1][1])
  right = max(pts[0][0], pts[1][0])
  return (int(top/ws), int(right/ws), int(bottom/ws), int(left/ws))

clicked_cls = ''
clicked_idx = 0
clicked_names = []
refPt = []
sp = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

def click(event, x, y, flags, params):

  global clicked_cls, clicked_idx, clicked_names

  global refPt

  image = params[0]
  preds_per_person = params[1]
  face_indices = params[2]
  ws = params[3]
  svm_clf = params[4]
  main_face = params[5]
  main_idx = params[6]

  if flags == (cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON):
    draw_rects(face_indices, preds_per_person, main_face, main_idx, ws, image)
    pt = Point(x,y)
    for i in face_indices:
      cls, idx = i
      (top, right, bottom, left) = preds_per_person[cls][idx][0][1]
      top = int(top * ws)
      right = int(right * ws)
      bottom = int(bottom * ws)
      left = int(left * ws)
      p = Polygon([(left, top), (right, top), (right, bottom), (left, bottom)])
      if p.contains(pt):
        cv2.rectangle(image, (left, top), (right, bottom), (255, 128, 0), 1)
        clicked_cls = cls
        clicked_idx = idx
        print('clicked class: {}, clicked index: {}'.format(clicked_cls, clicked_idx))
        clicked_names, probs = predict_face_svm(preds_per_person[clicked_cls][clicked_idx][2], svm_clf)
        break
      # else:
        # cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
        # draw_rects(face_indices, preds_per_person, main_face, main_idx, ws, image)
  elif event == cv2.EVENT_LBUTTONDOWN:
    refPt = [(x, y)]
  elif event == cv2.EVENT_LBUTTONUP:
    refPt.append((x, y))
    # draw a rectangle around the region of interest
    cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 1)
    p = Polygon([refPt[0], (refPt[1][0], refPt[0][1]), refPt[1], (refPt[0][0], refPt[1][1])])
    if p.area >= 10:
      cv2.imshow("faces", image)
      cv2.waitKey(1)
      new_name = guided_input(preds_per_person)
      new_loc = get_rect_from_pts(refPt, ws) # order in preds_per_person: (top(), right(), bottom(), left())
      d = dlib.rectangle(new_loc[3], new_loc[0], new_loc[1], new_loc[2])
      opencvImage = cv2.imread(preds_per_person[main_face][main_idx][1])
      shape = sp(opencvImage, d)
      face_descriptor = facerec.compute_face_descriptor(opencvImage, shape)
      new_desc = np.array(face_descriptor)
      if preds_per_person.get(new_name) == None:
        preds_per_person[new_name] = []
      preds_per_person[new_name].append(((new_name, new_loc), preds_per_person[main_face][main_idx][1], new_desc, 1, preds_per_person[main_face][main_idx][4]))
    refPt = []

  cv2.imshow("faces", image)

def show_detections_on_image(locations, img_path, waitkey=True):
  opencvImage = cv2.imread(img_path)

  height, width = opencvImage.shape[:2]
  ws = 600.0 / float(height)
  opencvImage = cv2.resize(opencvImage, (int(width * ws), int(height * ws)))

  for l in locations:
    (top, right, bottom, left) = l
    top = int(top * ws)
    right = int(right * ws)
    bottom = int(bottom * ws)
    left = int(left * ws)
    cv2.rectangle(opencvImage, (left, top), (right, bottom), (0, 255, 0), 1)

  cv2.imshow("detections", opencvImage)
  if waitkey:
    return cv2.waitKey(0)
  else:
    return cv2.waitKey(1)

def draw_rect(image, loc, scale, color):
  (top, right, bottom, left) = loc
  top = int(top * scale)
  right = int(right * scale)
  bottom = int(bottom * scale)
  left = int(left * scale)
  cv2.rectangle(image, (left, top), (right, bottom), color, 1)

def draw_rects(face_indices, preds_per_person, main_face, main_idx, ws, image):
  for i in face_indices:
    cls, idx = i
    if cls == 'unknown':
      color = (0, 0, 255)
    else:
      color = (0, 255, 0)
    draw_rect(image, preds_per_person[cls][idx][0][1], ws, color)

  draw_rect(image, preds_per_person[main_face][main_idx][0][1], ws, (255, 0, 0))

def show_faces_on_image(svm_clf, names, main_face, main_idx, preds_per_person, face_indices, img_path, waitkey=True, text = ''):
  # initilize the "clicked face" with the current face (main face)
  global clicked_cls, clicked_idx, clicked_names
  clicked_cls = main_face
  clicked_idx = main_idx
  clicked_names = names

  opencvImage = cv2.imread(img_path)

  height, width = opencvImage.shape[:2]
  ws = 600.0 / float(height)
  opencvImage = cv2.resize(opencvImage, (int(width * ws), int(height * ws)))

  confirmed = preds_per_person[main_face][main_idx][3]
  if confirmed >= 1:
    color = (0, 255, 0)
  else:
    color = (255, 0, 0)
  cv2.putText(opencvImage, main_face, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
  if text != None:
    cv2.putText(opencvImage, text, (20, opencvImage.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

  draw_rects(face_indices, preds_per_person, main_face, main_idx, ws, opencvImage)

  cv2.namedWindow("faces")
  cv2.setMouseCallback("faces", click, (opencvImage, preds_per_person, face_indices, ws, svm_clf, main_face, main_idx))

  cv2.imshow("faces", opencvImage)
  if waitkey:
    return cv2.waitKey(0), clicked_cls, clicked_idx, clicked_names
  else:
    return cv2.waitKey(1), clicked_cls, clicked_idx, clicked_names

# if index is -1: use all elements of predictions, if not, only use one (given by the index)
def show_prediction_labels_on_image(predictions, pil_image, confirmed=None, index=-1, img_path=None, text=None, force_name=''):
    if img_path != None:
        opencvImage = cv2.imread(img_path)
    else:
        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    height, width = opencvImage.shape[:2]
    ws = 600.0 / float(height)
    opencvImage = cv2.resize(opencvImage, (int(width * ws), int(height * ws)))

    if index != -1:
        name, (top, right, bottom, left) = predictions[index]
        if force_name != '':
          name = force_name
        top = int(top * ws)
        right = int(right * ws)
        bottom = int(bottom * ws)
        left = int(left * ws)
        cv2.rectangle(opencvImage, (left, top), (right, bottom), (0, 255, 0), 1)
        if confirmed >= 1:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.putText(opencvImage, name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        if text != None:
            cv2.putText(opencvImage, text, (20, opencvImage.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    else:
        for name, (top, right, bottom, left) in predictions:
            top = int(top * ws)
            right = int(right * ws)
            bottom = int(bottom * ws)
            left = int(left * ws)
            cv2.rectangle(opencvImage, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(opencvImage, name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            if text != None:
                cv2.putText(opencvImage, text, (20, opencvImage.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow("detections", opencvImage)
    return cv2.waitKey(0)

def guided_input(persons, add_persons=None):
  options = list(persons.keys())
  if add_persons != None:
    for p in add_persons:
      options.append(p)

  user_input = input("Enter new name: ")

  filtered_names = []

  for i in options:
    if user_input in i:
      filtered_names.append(i)

  if len(filtered_names) > 1:
    # Deal with more that one team.
    filtered_names.append('abort')
    print('There is more than one person starting with "{}"'.format(user_input))
    print('Select the correct person from these choices: ')
    for index, name in enumerate(filtered_names):
      print("{}: {}".format(index, name))

    index = input("Enter choice number: ")
    if not index.isdigit():
      print('aborted')
      return ""
    else:
      index = int(index)

    if index == len(filtered_names) - 1:
      new_name = ""
      print('aborted')
    else:
      new_name = filtered_names[index]
      print('Selected person: {}'.format(new_name))
  elif len(filtered_names) == 1:
    # Only one person found
    new_name = filtered_names[0]
  elif len(filtered_names) == 0:
    print('Generate a new person {}?'.format(user_input))
    print('1 ... yes, 0 ... no')
    index = input("Enter choice number: ")
    if not index.isdigit():
      print('aborted')
      index = 0
    else:
      index = int(index)
    if index == 1:
      new_name = user_input
    else:
      new_name = ""

  return new_name

def load_detections(path):
  dets = {}
  files = get_files_in_dir_by_name_rec(path, 'detections.bin')
  for f in files:
    dirname = os.path.dirname(f)
    dets[dirname] = pickle.load(open(f, "rb"))

  return dets

def load_detections_as_single_dict(path):
  dets = {}
  det_file_map = {}
  files = get_files_in_dir_by_name_rec(path, 'detections.bin')
  for f in files:
    print('loading {}'.format(os.path.dirname(f)))
    det_file_map[f] = []
    #dirname = os.path.dirname(f)
    tmp = pickle.load(open(f, "rb"))
    for t in tmp:
      det_file_map[f].append(t)
      dets[t] = tmp[t]

  return dets, det_file_map

def save_detections(dets, det_file_map):
  #dets_per_folder = {}
  for dm in det_file_map:
    dets_to_save = {}
    for d in det_file_map[dm]:
      if dets.get(d) != None:
        dets_to_save[d] = dets[d]

    with open(dm, "wb") as fp:
      pickle.dump(dets_to_save, fp)

    # dirname = os.path.dirname(d)
    # if len(dets_per_folder[dirname]) == 0:
    #   dets_per_folder[dirname][d] = dets[d]
    # dets_per_folder[dirname]

    # outfile = os.path.join(d, 'detections.bin')
    # with open(outfile, "wb") as fp:
    #   pickle.dump(dets[d], fp)

def delete_detections_of_file(dets, filepath):
  if dets.get(filepath) != None:
    dets.pop(filepath)
    print('detections in {} deleted'.format(filepath))
  else:
    print('{} not found in detections'.format(filepath))

def ignore_detections_of_file(dets, filepath):
  if dets.get(filepath) != None:
    dets[filepath] = ([],[])
    print('detections in {} will be ignored'.format(filepath))
  else:
    print('{} not found in detections'.format(filepath))

def predict_face_svm(enc, svm, print_top=True):
  preds = svm.predict_proba(enc.reshape(1, -1))[0]
  sorted = np.argsort(-preds)

  names = []
  probs = []
  for s in range(10):
    ix = sorted[s]
    names.append(svm.classes_[ix])
    probs.append(preds[ix])

  if print_top:
    print_top_svm(names, probs)

  return names, probs

def print_top_svm(names, probs):
  print('\n')
  for i,(n,p) in enumerate(zip(names,probs)):
    print('{}: name: {}, prob: {}'.format(i, n, p))
  print('\n')

def save_face_crop(face_path, img_path, loc):
    img = cv2.imread(img_path)

    x = loc[3]
    y = loc[0]
    w = loc[1] - x
    h = loc[2] - y
    if is_valid_roi(x, y, w, h, img.shape):
      roi = img[y:y + h, x:x + w]
      cv2.imwrite(face_path, roi)
      return True

    return False

def save_face_crop_aligned(sp, face_path, img_path, loc):
  img = cv2.imread(img_path)
  # loc = (d.top(), d.right(), d.bottom(), d.left())
  d = dlib.rectangle(loc[3], loc[0], loc[1], loc[2])
  shape = sp(img, d)

  aligned_face = dlib.get_face_chip(img, shape)
  cv2.imwrite(face_path, aligned_face)

def get_faces_in_files(preds_per_person):
  faces_files = {}
  for p in preds_per_person:
    for i,f in enumerate(preds_per_person[p]):
      if f[0][0] != 'deleted':
        if faces_files.get(f[1]) == None:
          faces_files[f[1]] = []
        faces_files[f[1]].append((p, i))
  return faces_files