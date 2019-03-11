import os
import os.path
import pickle
import csv
import numpy as np
import dlib
from collections import Counter, OrderedDict
import cv2

def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def get_images_in_dir_rec(path_to_dir):
  files = []
  for root, dirnames, filenames in os.walk(os.path.normpath(path_to_dir)):
    for j in filenames:
      if j.lower().endswith(('.jpg', '.jpeg', '.png')):
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
    export_face_to_csv(preds_per_person_path, preds_per_person, p)

def export_face_to_csv(preds_per_person_path, preds_per_person, face):

  print('exporting {}'.format(face))
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
    print('--db is needed')
    exit()
  if not os.path.isdir(preds_per_person_path):
    print('--db is not a valid directory')
    exit()

  preds_per_person = {}

  csv_files = get_files_in_dir(preds_per_person_path, '.csv')
  bin_files = get_files_in_dir(preds_per_person_path, '.bin')
  if len(csv_files) == 0:
    print('no csv files found in {}'.format(preds_per_person_path))
    return preds_per_person
  for f in csv_files:
    name = os.path.splitext(os.path.basename(f))[0]
    # print('loading {}'.format(name))
    descs = pickle.load(open(os.path.join(preds_per_person_path, name + '.bin'), "rb"))
    for i in bin_files:
      filename = os.path.splitext(os.path.basename(i))[0]
      # check if multiple .bin files should be combined
      # bin files can be combined if the 'other' files are exactly one character longer, e.g.: martin.bin and martin1.bin
      if name in filename and len(name) + 1 == len(filename):
        accept = input('Add ' + i + ' to ' + name + ' (y/n)?')
        if accept == 'y':
          tmp = pickle.load(open(i, "rb"))
          descs += tmp
    # bin_path = os.path.join(preds_per_person_path, name + '.bin')
    # descs = pickle.load(open(bin_path, "rb"))
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
          preds_per_person[name].append(
            ((name, face_loc[1]), found_face[1], found_face[2], found_face[3], found_face[4]))
        else:
          print('{} not found'.format(img_name))

  return preds_per_person

def sort_folders_by_nr_of_images(folder_to_sort):
  entries = os.listdir(os.path.normpath(folder_to_sort))
  dirs = []
  for f in entries:
    if os.path.isdir(os.path.join(folder_to_sort, f)):
      dirs.append(os.path.join(folder_to_sort, f))

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

def detect_faces_in_image(img_path, detector, facerec, sp, use_entire_image=False, img_height=150):
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

    # locations: list of tuples (t,r,b,l)
    # descriptors: list of float64 np arrays
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    descriptors.append(np.array(face_descriptor))
    locations.append((d.top(), d.right(), d.bottom(), d.left()))

  return locations, descriptors