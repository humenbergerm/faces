import os
import os.path
import pickle
import argparse
import dlib

import utils

def detect_faces(args):
    detections_path = os.path.join(args.outdir, "detections.bin")
    if os.path.isfile(detections_path):
        print('loading {}'.format(detections_path))
        detections = pickle.load(open(detections_path, "rb"))
    else:
        detections = {}

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

    files = utils.get_images_in_dir_rec(os.path.normpath(args.input))

    # Find all the faces and compute 128D face descriptors for each face.
    counter = 1
    total = len(files)
    for n, f in enumerate(files):
        print("Processing file ({}/{}): {}".format(counter, total, f))
        counter += 1

        if detections.get(f) != None and not args.recompute:
            print('file already processed, skipping, ...')
            continue

        locations, descriptors = utils.detect_faces_in_image(f, detector, facerec, sp)

        detections[os.path.abspath(f)] = (locations, descriptors)

        if n % 100 == 0:
            with open(detections_path, "wb") as fp:
                pickle.dump(detections, fp)
                print('saved')

    with open(detections_path, "wb") as fp:
        pickle.dump(detections, fp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help="Input image directory.")
    parser.add_argument('--outdir', type=str, required=True,
                        help="Output directory.")
    parser.add_argument('--recompute', help='Recompute detections.',
                        action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print('args.input needs to be a valid folder containing images')
        exit()

    if not os.path.isdir(args.outdir):
        utils.mkdir_p(args.outdir)

    print('Detecting faces in {}'.format(args.input))
    detect_faces(args)
    print('Done.')

if __name__ == "__main__":
    main()