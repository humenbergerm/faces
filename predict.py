import os.path
import pickle
from PIL import Image
import argparse
from datetime import datetime
import copy
import subprocess

import utils

def confirm_face(preds_per_person, predictions, name, i, ins, names, args, svm_clf):

  # names, probs = utils.predict_face_svm(preds_per_person[name][ins][2], svm_clf)

  key = utils.show_prediction_labels_on_image(predictions, None, preds_per_person[name][ins][3], i,
                                              preds_per_person[name][ins][1], '', force_name=name)

  if key == 99:  # key 'c'
    new_name = utils.guided_input(preds_per_person)
    if new_name != "":
      # add pred in new list
      if preds_per_person.get(new_name) == None:
        preds_per_person[new_name] = []
      # print(preds_per_person[name][-1])
      # print(preds_per_person[new_name][-1])
      utils.insert_element_preds_per_person(preds_per_person, name, ins, new_name, 1)
      # print(preds_per_person[new_name][-1])
      # delete pred in current list
      face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, name, ins)
      # print(preds_per_person[name][-1])
      print("face changed: {} ({})".format(new_name, len(preds_per_person[new_name])))
  elif key == 100:  # key 'd'
    new_name = 'deleted'
    # add pred in new list
    if preds_per_person.get(new_name) == None:
      preds_per_person[new_name] = []
    utils.insert_element_preds_per_person(preds_per_person, name, ins, new_name)
    # delete pred in current list
    face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, name, ins)
    print("face deleted")
  elif key == 47:  # key '/'
    tmp = preds_per_person[name][ins]
    if tmp[3] == 0:
      preds_per_person[name][ins] = tmp[0], tmp[1], tmp[2], 1, tmp[4]
    elif tmp[3] == 1:
      preds_per_person[name][ins] = tmp[0], tmp[1], tmp[2], 0, tmp[4]
    print("face confirmed: {} ({})".format(tmp[0], len(preds_per_person[name])))
  elif key >= 48 and key <= 57:  # keys '0' - '9'
    new_name = names[key - 48]
    utils.insert_element_preds_per_person(preds_per_person, name, ins, new_name, 1)
    # delete pred in current list
    face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, name, ins)
    print("face confirmed: {} ({})".format(new_name, len(preds_per_person[new_name])))

  return key

def predict_class(args, knn_clf, svm_clf):

  cls = args.detections
  print('Detecting faces in class {} using knn.'.format(cls))
  preds_per_person = utils.load_faces_from_csv(args.db)

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
  unknown_counter = 0
  new_counter = 0
  known_counter = 0
  pos = 0

  for i, (name, (top, right, bottom, left)) in enumerate(predictions):

    names, probs = utils.predict_face_svm(preds_per_person[cls][pos][2], svm_clf, print_top=True)
    if name == "unknown":
      if probs[0] >= 0.95:
        name = names[0]
        print('{} > 0.95'.format(name))
      else:
        unknown_counter += 1
        pos += 1

    if name != cls:
      save = copy.deepcopy(preds_per_person)
      # move to new class
      new_counter += 1
      tmp = preds_per_person[cls][pos]
      if preds_per_person.get(name) == None:
        preds_per_person[name] = []
      # print(preds_per_person[name][-1])
      preds_per_person[name].append(((name, tmp[0][1]), tmp[1], tmp[2], tmp[3], tmp[4]))
      preds_per_person[cls].pop(pos) # this is why pos is needed
      print('{} found'.format(name))

      if args.confirm:
        repeat = True
        while repeat == True:
          key = confirm_face(preds_per_person, predictions, name, i, -1, names, args, svm_clf)
          if key == 27: # key 'esc'
            preds_per_person = save
            utils.export_persons_to_csv(preds_per_person, args.db)
            return 0
          elif key == 116:  # key 't'
            image_path = preds_per_person[name][-1][1]
            subprocess.call(["open", "-R", image_path])
          else:
            repeat = False
    else:
      known_counter += 1
      # pos += 1

  utils.export_persons_to_csv(preds_per_person, args.db)

  print("predicted faces of class {}".format(cls))
  print("{} new face(s) found. They were moved to their class.".format(new_counter))
  print("{} face(s) unknown. They are not changed!".format(unknown_counter))
  print("{} face(s) unchanged.".format(known_counter))

def predict_image(descriptors, locations, knn_clf, distance_threshold=0.3):

    # Predict classes and remove classifications that aren't within the threshold
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(descriptors, n_neighbors=3)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(locations))]

    predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(descriptors), locations, are_matches)]
    #print("predicted faces...")

    return predictions

def predict_faces(args, knn_clf, svm_clf, detections):

    if len(detections) == 0:
        print('no detections found')
        exit()

    # if args.recompute:
    #   preds_per_person = {}
    # else:
    preds_per_person = utils.load_faces_from_csv(args.db)

    counter = 0

    # detections_save = detections.copy()
    for n, image_file in enumerate(detections):
        print("{}/{}".format(n, len(detections)))
        # detections_save.pop(image_file)

        locations = detections[image_file][0]
        descriptors = detections[image_file][1]
        full_file_path = image_file
        if not os.path.isfile(full_file_path):
          continue

        if len(locations) == 0:
          #print('no faces found')
          continue

        # in order to use the exif timestamp, all timestamps and dates etc in the exif data from the images need to be fixed first
        timeStamp = datetime.now()
        no_timestamp = True
        if full_file_path.lower().endswith(('.jpg')):
            pil_image = Image.open(full_file_path)
            exif = pil_image._getexif()
            if exif != None:
                if exif.get(36868) != None:
                    date = exif[36868]
                    if utils.is_valid_timestamp(date):
                        timeStamp = datetime.strptime(date, '%Y:%m:%d %H:%M:%S')
                        no_timestamp = False

        predictions = predict_image(descriptors, locations, knn_clf)

        for id, (name, (top, right, bottom, left)) in enumerate(predictions):

            if preds_per_person.get(name) == None:
              preds_per_person[name] = []

            found = 0
            #found_at = ''
            # if not args.recompute:
            for y in preds_per_person:
              for x in preds_per_person[y]:
                if x[0][1] == predictions[id][1] and x[1] == image_file:
                  found = 1
                  found_at = y
                  print('Found old face {}.'.format(y))
                  break
              if found == 1:
                break

            if found == 0:
                names, probs = utils.predict_face_svm(descriptors[id], svm_clf, print_top=True)
                if name == 'unknown':
                  if probs[0] >= 0.95:
                    name = names[0]
                    print('{} > 0.95'.format(name))

                print('Found new face {}.'.format(name))
                save = copy.deepcopy(preds_per_person)
                counter = counter + 1
                if len(preds_per_person[name]) == 0 or no_timestamp:
                    preds_per_person[name].append([(name, predictions[id][1]), image_file, descriptors[id], 0, timeStamp])
                    ins = -1
                else:
                    inserted = False
                    for ins, pr in enumerate(preds_per_person[name]):
                        if timeStamp <= pr[4]:
                            preds_per_person[name].insert(ins, [(name, predictions[id][1]), image_file, descriptors[id], 0, timeStamp])
                            inserted = True
                            break
                    if not inserted:
                        preds_per_person[name].append([(name, predictions[id][1]), image_file, descriptors[id], 0, timeStamp])

                if args.confirm:
                  repeat = True
                  while repeat == True:
                    key = confirm_face(preds_per_person, predictions, name, id, ins, names, args, svm_clf)
                    if key == 27:  # key 'esc'
                      preds_per_person = save
                      utils.export_persons_to_csv(preds_per_person, args.db)
                      return 0
                    elif key == 116:  # key 't'
                      image_path = preds_per_person[name][ins][1]
                      subprocess.call(["open", "-R", image_path])
                    else:
                      repeat = False
            # else:
            #     print('face already in database ({})'.format(found_at))

        if n % 1000 == 0 and n != 0:
            utils.export_persons_to_csv(preds_per_person, args.db)
            print('saved')

    utils.export_persons_to_csv(preds_per_person, args.db)
    print('Found {} new faces.'.format(counter))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--detections', type=str, required=True,
                      help="Path to detections.bin file(s) or name of an already predicted class, such as unknown.")
  parser.add_argument('--knn', type=str, required=True,
                      help="Path to knn model file (e.g. knn.clf).")
  parser.add_argument('--svm', type=str, required=True,
                      help="Path to svm model file (e.g. svm.clf).")
  parser.add_argument('--db', type=str, required=True,
                      help="Path to folder with predicted faces (.csv files).")
  parser.add_argument('--confirm', help='Each newly found face needs to be confirmed.',
                      action='store_true')
  # parser.add_argument('--recompute', help='Recompute detections.',
  #                     action='store_true')
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

  # if args.recompute:
  #   answer = input("You are about to delete and recompute the content of args.db. Continue? y/n")
  #   if answer != 'y':
  #     print('Aborted.')
  #     exit()

  if os.path.isdir(args.detections):
    print('Predicting faces in {}'.format(args.detections))
    detections, det_file_map = utils.load_detections_as_single_dict(args.detections)
    predict_faces(args, knn_clf, svm_clf, detections)
  elif os.path.isfile(args.detections):
    detections = pickle.load(open(args.detections, "rb"))
    predict_faces(args, knn_clf, svm_clf, detections)
  else:
    print('Predicting faces of class {}'.format(args.detections))
    predict_class(args, knn_clf, svm_clf)

  print('Done.')

if __name__ == "__main__":
  main()