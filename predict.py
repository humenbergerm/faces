import os.path
import pickle
import argparse
import copy
import subprocess

import utils

def predict_class(args, knn_clf, svm_clf):

  cls = args.cls
  print('Detecting faces in class {} using knn.'.format(cls))
  preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)

  if preds_per_person.get(cls) == None:
    print('class {} not found'.format(cls))
    exit()

  face_locations = []
  face_encodings = []
  face_path = []
  for p in preds_per_person[cls]:
    face_locations.append(p[0][1])
    face_encodings.append(p[2])
    face_path.append(p[1])

  print('{} members of {}'.format(len(face_locations), cls))
  if len(face_locations) == 0:
    return

  distance_threshold = 0.3

  # Use the KNN model to find the best matches for the test face
  closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=3)
  are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

  predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                 zip(knn_clf.predict(face_encodings), face_locations, are_matches)]

  # assumption: predictions has the same length like the class members
  for i,p in enumerate(preds_per_person[cls]):
    preds_per_person[cls][i] = (predictions[i][0], p[0][1]), p[1], p[2], p[3], p[4], p[5]

  ix = 0
  key = 0
  save = []
  nr_of_faces = len(preds_per_person[cls])

  while key != 27 and nr_of_faces > 0:

    nr_of_faces = len(preds_per_person[cls])  # because it might have changed (e.g. if a face got deleted)
    if nr_of_faces == 0:
      print('no more faces of class {} found'.format(cls))
      break

    if ix >= nr_of_faces:
      ix = 0
    elif ix < 0:
      ix = nr_of_faces - 1

    name = preds_per_person[cls][ix][0][0]
    print('predicted: {}'.format(name))

    while len(save) > 10:
      save.pop(0)

    faces_files = utils.get_faces_in_files(preds_per_person)

    names, probs = utils.predict_face_svm(preds_per_person[cls][ix][2], svm_clf, print_top=True)
    if name == "unknown":
      if probs[0] >= 0.95:
        name = names[0]
        print('{} > 0.95'.format(name))
      else:
        utils.insert_element_preds_per_person(preds_per_person, cls, ix, name, conf=0)
        preds_per_person[cls].pop(ix)

    if name != cls and name != 'unknown':
      if args.confirm:
        image_path = preds_per_person[cls][ix][1]
        key, clicked_class, clicked_idx, clicked_names = utils.show_faces_on_image(svm_clf, names, cls, ix,
                                                                                   preds_per_person,
                                                                                   faces_files[image_path], image_path,
                                                                                   waitkey=True, text=name)
        deleted_elem_of_cls = utils.evaluate_key(args, key, preds_per_person, clicked_class, clicked_idx, save, clicked_names, faces_files)

        if deleted_elem_of_cls > 0 and clicked_idx <= ix and clicked_class == cls:
          ix -= deleted_elem_of_cls

        if key == 46 or key == 47:  # key '.' or key '/'
          ix += 1
        elif key == 44:  # key ','
          ix -= 1
        elif key == 98:  # key 'b'
          if len(save) > 0:
            preds_per_person = copy.deepcopy(save.pop())
            print("undone last action")
        elif key == 111: # 'o'
          # move to new class
          utils.insert_element_preds_per_person(preds_per_person, cls, ix, name, conf=1)
          preds_per_person[cls].pop(ix)
      else:
        # move to new class
        utils.insert_element_preds_per_person(preds_per_person, cls, ix, name, conf=0)
        preds_per_person[cls].pop(ix)
    else:
      ix += 1

  utils.export_persons_to_csv(preds_per_person, args.imgs_root, args.db)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cls', type=str, required=True,
                      help="Class to predict, such as unknown or detected.")
  parser.add_argument('--knn', type=str, required=True,
                      help="Path to knn model file (e.g. knn.clf).")
  parser.add_argument('--svm', type=str, required=True,
                      help="Path to svm model file (e.g. svm.clf).")
  parser.add_argument('--db', type=str, required=True,
                      help="Path to folder with predicted faces (.csv files).")
  parser.add_argument('--imgs_root', type=str, required=True,
                      help="Root directory of your image library.")
  parser.add_argument('--confirm', help='Each newly found face needs to be confirmed.',
                      action='store_true')
  args = parser.parse_args()

  if not os.path.isdir(args.db):
    utils.mkdir_p(args.db)

  if os.path.isfile(args.knn):
    with open(args.knn, 'rb') as f:
      knn_clf = pickle.load(f)
  else:
    print('args.knn ({}) is not a valid file'.format(args.knn))
    exit()

  if os.path.isfile(args.svm):
    with open(args.svm, 'rb') as f:
      svm_clf = pickle.load(f)
  else:
    print('args.svm ({}) is not a valid file'.format(args.svm))
    exit()

  print('Predicting faces of class {}'.format(args.cls))
  predict_class(args, knn_clf, svm_clf)

  print('Done.')

if __name__ == "__main__":
  main()