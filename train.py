import math
from sklearn import neighbors
import os
import os.path
import pickle
import argparse
import dlib
from sklearn.svm import SVC

import utils


def train(args):
  X = []
  y = []
  train_dir = args.traindir

  detector = dlib.get_frontal_face_detector()
  sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
  facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

  if len(utils.get_files_in_dir(train_dir, '.csv')) != 0:
    # train using csv files
    print('Training using .csv files.')

    preds_per_person = utils.load_faces_from_csv(train_dir)
    for p in preds_per_person:
      if p != 'unknown':
        for l in preds_per_person[p]:
          X.append(l[2])
          y.append(p)

    if len(X) == 0:
      print('No faces found in database {}'.format(train_dir))
      return

  elif len(os.listdir(train_dir)) != 0:
    # train using train folder
    # Loop through each person in the training set
    print('Training using faces in subfolders.')
    for class_dir in os.listdir(train_dir):
      if not os.path.isdir(os.path.join(train_dir, class_dir)):
        continue

      images = utils.get_images_in_dir(os.path.join(train_dir, class_dir))
      if len(images) == 0:
        continue

      print('adding {} to training data'.format(class_dir))
      # Loop through each training image for the current person
      for img_path in images:
        locations, descriptors = utils.detect_faces_in_image(img_path, detector, facerec, sp, use_entire_image=True)

        # Add face descriptor for current image to the training set
        X.append(descriptors[0])
        y.append(class_dir)
      print('{} faces used for training'.format(len(images)))
  else:
    print('Training directory does not contain valid training data.')
    return

  # Determine how many neighbors to use for weighting in the KNN classifier
  n_neighbors = int(round(math.sqrt(len(X))))
  print("Chose n_neighbors automatically:", n_neighbors)

  # Create and train the KNN classifier
  print("Training model with KNN ...")
  knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
  knn_clf.fit(X, y)

  # Save the trained KNN classifier
  with open(os.path.join(args.outdir, 'knn.clf'), 'wb') as f:
    pickle.dump(knn_clf, f)

  # train the svm
  print("Training model with an SVM ...")
  recognizer = SVC(C=1.0, kernel="linear", probability=True)
  recognizer.fit(X, y)

  # Save the trained SVM
  with open(os.path.join(args.outdir, 'svm.clf'), 'wb') as f:
    pickle.dump(recognizer, f)

  print('Trained models with {} faces'.format(len(X)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--traindir', type=str, required=True,
                             help="Path to folder containing subfolders to train faces.")
    parser.add_argument('--outdir', type=str, required=True,
                             help="Path to store the trained models.")
    args = parser.parse_args()

    if not os.path.isdir(args.traindir):
        print('args.traindir needs to be a valid folder')
        exit()

    if not os.path.isdir(args.outdir):
        utils.mkdir_p(args.outdir)

    print('Training knn and svm model using data in {}.'.format(args.traindir))
    train(args)
    print('Done.')