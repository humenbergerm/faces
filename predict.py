import os.path
import pickle
from PIL import Image
import argparse
from datetime import datetime

import utils

def predict_class(args, knn_clf):

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
  known_counter = 0
  pos = 0
  for name, (top, right, bottom, left) in predictions:
    if (name == "unknown"):
      unknown_counter += 1
      pos += 1
    else:
      # move to new class
      known_counter += 1
      tmp = preds_per_person[cls][pos]
      if preds_per_person.get(name) == None:
        preds_per_person[name] = []
      preds_per_person[name].append(((name, tmp[0][1]), tmp[1], tmp[2], tmp[3], tmp[4]))
      preds_per_person[cls].pop(pos)
      print('{} found'.format(name))

  utils.export_persons_to_csv(preds_per_person, args.db)

  print("predicted faces of class {}".format(cls))
  print("{} new face(s) found. They were moved to their class.".format(known_counter))
  print("{} face(s) unknown. They are not changed!".format(unknown_counter))

def predict_image(descriptors, locations, knn_clf, distance_threshold=0.3):

    # Predict classes and remove classifications that aren't within the threshold
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(descriptors, n_neighbors=3)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(locations))]

    predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(descriptors), locations, are_matches)]
    #print("predicted faces...")

    return predictions

def predict_faces(args, knn_clf, detections):

    if len(detections) == 0:
        print('no detections found')
        exit()

    if args.recompute:
      preds_per_person = {}
    else:
      preds_per_person = utils.load_faces_from_csv(args.db)

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
            if not args.recompute:
              for y in preds_per_person:
                for x in preds_per_person[y]:
                  if x[0][1] == predictions[id][1] and x[1] == image_file:
                    found = 1
                    found_at = y
                    break
                if found == 1:
                  break

            if found == 0:
                print('Found new face {}.'.format(name))
                if len(preds_per_person[name]) == 0 or no_timestamp:
                    preds_per_person[name].append([predictions[id], image_file, descriptors[id], 0, timeStamp])
                else:
                    inserted = False
                    for ins, pr in enumerate(preds_per_person[name]):
                        if timeStamp <= pr[4]:
                            preds_per_person[name].insert(ins, [predictions[id], image_file, descriptors[id], 0, timeStamp])
                            inserted = True
                            break
                    if not inserted:
                        preds_per_person[name].append([predictions[id], image_file, descriptors[id], 0, timeStamp])
            else:
                print('face already in database ({})'.format(found_at))

        if n % 10000 == 0:
            utils.export_persons_to_csv(preds_per_person, args.db)
            print('saved')

    utils.export_persons_to_csv(preds_per_person, args.db)

    # if len(detections_save) != 0:
    #   with open(args.detections, "wb") as fp:
    #     pickle.dump(detections_save, fp)
    # else:
    #   print('All detections processed. To make sure you do not do it again, delete {}.'.format(args.detections))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--detections', type=str, required=True,
                      help="Path to detections.bin file(s) or name of an already predicted class, such as unknown.")
  parser.add_argument('--knn', type=str, required=True,
                      help="Path to knn model file (e.g. knn.clf).")
  parser.add_argument('--db', type=str, required=True,
                      help="Path to folder with predicted faces (.csv files).")
  parser.add_argument('--recompute', help='Recompute detections.',
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

  if args.recompute:
    answer = input("You are about to delete and recompute the content of args.db. Continue? y/n")
    if answer != 'y':
      print('Aborted.')
      exit()

  if os.path.isdir(args.detections):
    print('Predicting faces in {}'.format(args.detections))
    detections = utils.load_detections_as_single_dict(args.detections)
    predict_faces(args, knn_clf, detections)
  elif os.path.isfile(args.detections):
    detections = pickle.load(open(args.detections, "rb"))
    predict_faces(args, knn_clf, detections)
  else:
    print('Predicting faces of class {}'.format(args.detections))
    predict_class(args, knn_clf)

  print('Done.')

if __name__ == "__main__":
  main()