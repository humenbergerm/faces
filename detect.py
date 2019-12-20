import os
import os.path
import pickle
import argparse
import dlib
import cv2
from shapely.geometry import Polygon

import utils

processed_faces = 0

def detect_faces(args):

    detection_status_file = os.path.join(args.db, 'detection_status.bin')
    if os.path.isfile(detection_status_file):
      detection_status = pickle.load(open(detection_status_file, 'rb'))
      # detection_status = {}
      # for ds in detection_status_fromfile:
      #   new_path = os.path.relpath(ds.replace('OneDrive', 'odrive/OneDrive'), args.imgs_root)
      #   detection_status[new_path] = detection_status_fromfile[ds]
    else:
      detection_status = {}

    preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)
    faces_files = utils.get_faces_in_files(preds_per_person)

    # initialize counters
    total_faces = len(utils.get_images_in_dir_rec(args.input))

    dirs = utils.get_folders_in_dir_rec(args.input)
    # if there is no sub-folder, process input folder
    if len(dirs) == 0:
      dirs.append(args.input)

    for d in dirs:
      # check if folder contains images, if not -> skip
      imgs = utils.get_images_in_dir(d)
      if len(imgs) == 0:
        continue
      detect_faces_in_folder(args, preds_per_person, faces_files, imgs, detection_status, total_faces)

def detect_faces_in_folder(args, preds_per_person, faces_files, files, detection_status, total_faces):
    global processed_faces
    detection_status_file = os.path.join(args.db, 'detection_status.bin')
    # detections_path = os.path.join(output_path, 'detections.bin')
    # if os.path.isfile(detections_path):
    #     print('loading {}'.format(detections_path))
    #     detections = pickle.load(open(detections_path, "rb"))
    # else:
    #     detections = {}

    # opencv face detector
    modelFile = "models/opencv_face_detector_uint8.pb"
    configFile = "models/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    detector = dlib.get_frontal_face_detector()
    # detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
    sp = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

    # files = utils.get_images_in_dir_rec(os.path.normpath(folder))

    # Find all the faces and compute 128D face descriptors for each face.
    counter = 1
    # total = len(files)
    changed = False
    for n, f in enumerate(files):
        # print("Processing file ({}/{}): {}".format(counter, total, f))
        print("Processing file ({}/{}): {}".format(processed_faces, total_faces, f))
        counter += 1
        processed_faces = processed_faces+1

        rel_path = os.path.relpath(f, args.imgs_root)
        if detection_status.get(rel_path) != None and not args.recompute:
          print('file already processed, skipping, ...')
          continue

        changed = True

        # if detections.get(f) != None and not args.recompute:
        #     print('file already processed, skipping, ...')
        #     # locs, descs = detections[os.path.abspath(f)]
        #     # utils.show_detections_on_image(locs, f)
        #     continue

        locations_cv2, descriptors_cv2, imagesize = utils.detect_faces_in_image_cv2(f, net, facerec, sp, detector)
        print('cv2: {} detection(s) found'.format(len(locations_cv2)))
        # utils.show_detections_on_image(locations_cv2, f)

        locations, descriptors, imagesize = utils.detect_faces_in_image(f, detector, facerec, sp)
        print('dlib: {} detection(s) found'.format(len(locations)))
        # utils.show_detections_on_image(locations, f)

        # utils.show_detections_on_image(locations+locations_cv2, f)

        # merge detections dlib and cv2
        locs, descs = utils.merge_detections(locations, descriptors, locations_cv2, descriptors_cv2)
        print('total: {} detection(s) found'.format(len(locs)))

        # merge detections with already saved ones
        # detections[os.path.abspath(f)] = (locs, descs)
        # utils.show_detections_on_image(detections[os.path.abspath(f)][0], f)
        # utils.show_detections_on_image(detections[os.path.abspath(f)][0] + locs, f)
        # if detections.get(os.path.abspath(f)) != None:
        #   locs, descs = utils.merge_detections(detections[os.path.abspath(f)][0], detections[os.path.abspath(f)][1], locs, descs)

        # utils.show_detections_on_image(detections[os.path.abspath(f)][0], f)

        # save dets to preds_per_person class "detected"
        timeStamp = utils.get_timestamp(f)
        cls = 'detected'
        for l,d in zip(locs, descs):
          utils.add_new_face(preds_per_person, faces_files, cls, l, d, f, timeStamp, imagesize)

        detection_status[rel_path] = {'cv2': 1, 'dlib': 1}

        if n % 100 == 0 and n != 0 and changed:
          utils.export_persons_to_csv(preds_per_person, args.imgs_root, args.db)
          with open(detection_status_file, 'wb') as fp:
            pickle.dump(detection_status, fp)
          changed = False

    if changed:
      utils.export_persons_to_csv(preds_per_person, args.imgs_root, args.db)
      with open(detection_status_file, 'wb') as fp:
        pickle.dump(detection_status, fp)

def check_detections(detections):
  for d in list(detections):
    if not os.path.isfile(d):
      del detections[d]
      print('deleted detection {} because file does not exist'.format(d))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help="Input image directory. Recursive processing is supported.")
    parser.add_argument('--imgs_root', type=str, required=True,
                        help="Root directory of your image library.")
    # parser.add_argument('--outdir', type=str, required=True,
    #                     help="Output directory.")
    parser.add_argument('--db', type=str, required=True,
                        help="Path to folder with predicted faces (.csv files).")
    parser.add_argument('--recompute', help='Recompute detections.',
                        action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print('args.input needs to be a valid folder containing images')
        exit()

    # if not os.path.isdir(args.outdir):
    #     utils.mkdir_p(args.outdir)

    print('Detecting faces in {}'.format(args.input))
    detect_faces(args)
    print('Done.')

if __name__ == "__main__":
    main()