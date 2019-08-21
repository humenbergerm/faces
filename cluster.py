import os
import os.path
import pickle
import argparse
import dlib
import cv2
import numpy as np

import utils

def cluster_faces_in_class(args):
  preds_per_person = utils.load_faces_from_csv(args.db)
  if preds_per_person.get(args.detections) == None:
    print('Class {} not found.'.format(args.detections))
    return

  descriptors = []
  for i,p in enumerate(preds_per_person[args.detections]):
    descriptors.append(dlib.vector(p[2]))

  # cluster the faces
  print('clustering...')
  labels = dlib.chinese_whispers_clustering(descriptors, args.threshold)
  num_classes = len(set(labels))
  print("Number of clusters: {}".format(num_classes))

  all_indices = []
  all_lengths = []
  for j in range(0, num_classes):
    class_length = len([label for label in labels if label == j])
    if class_length >= args.min_members:
      indices = []
      for i, label in enumerate(labels):
        if label == j:
          indices.append(i)
      all_indices.append(indices)
      all_lengths.append(class_length)

  sort_index = np.argsort(np.array(all_lengths))[::-1]

  # Move the clustered faces to individual groups
  print('Moving the clustered faces to the database.')
  to_delete = []
  for i in sort_index[:10]:
    cluster_name = "group_" + str(i)

    to_delete += all_indices[i]
    for index in all_indices[i]:
      utils.insert_element_preds_per_person(preds_per_person, args.detections, index, cluster_name)

  to_delete = sorted(to_delete)
  to_delete.reverse()
  for i in to_delete:
    utils.delete_element_preds_per_person(preds_per_person, args.detections, i)

  utils.export_persons_to_csv(preds_per_person, args.db)

def cluster_faces(args):
    cluster_root_path = args.outdir

    detections_path = args.detections
    detections, det_file_map = utils.load_detections_as_single_dict(detections_path)

    descriptors = []
    locations = []
    images = []
    for d in detections:
        locs = detections[d][0]
        desc = detections[d][1]
        for lo, de in zip(locs, desc):
            descriptors.append(dlib.vector(de))
            locations.append(lo)
            images.append(d)

    # cluster the faces
    labels = dlib.chinese_whispers_clustering(descriptors, args.threshold)
    num_classes = len(set(labels))
    print("Number of clusters: {}".format(num_classes))

    counter_unclustered = 0
    counter_clustered = 0
    for j in range(0, num_classes):
        class_length = len([label for label in labels if label == j])
        if class_length >= args.min_members:
            indices = []
            for i, label in enumerate(labels):
                if label == j:
                    indices.append(i)

            # folder_name = os.path.splitext(os.path.basename(f).replace(" ", "_"))[0] + "_group_"
            folder_name = "group_"
            cluster_path = os.path.join(cluster_root_path, folder_name + str(j))
            if not os.path.isdir(cluster_path):
                os.makedirs(cluster_path)

            # Save the extracted faces
            print('Saving faces in clusters to output folder {}'.format(cluster_path))
            for i, index in enumerate(indices):
                img_path = images[index]
                img = cv2.imread(img_path)
                file_name = os.path.join(cluster_path, 'face_' + folder_name + str(counter_clustered) + '.jpg')

                x = locations[index][3]
                y = locations[index][0]
                w = locations[index][1] - x
                h = locations[index][2] - y
                if utils.is_valid_roi(x, y, w, h, img.shape):
                  roi = img[y:y + h, x:x + w]
                  cv2.imwrite(file_name, roi)
                  counter_clustered += 1
        else:
            indices = []
            for i, label in enumerate(labels):
                if label == j:
                    indices.append(i)

            folder_name = "unclustered"
            cluster_path = os.path.join(cluster_root_path, folder_name)
            if not os.path.isdir(cluster_path):
                os.makedirs(cluster_path)

            # Save the extracted faces
            print('Saving unclustered faces to output folder {}'.format(cluster_path))
            for i, index in enumerate(indices):
              img_path = images[index]
              img = cv2.imread(img_path)
              file_name = os.path.join(cluster_path, 'face_' + folder_name + str(counter_unclustered) + '.jpg')

              x = locations[index][3]
              y = locations[index][0]
              w = locations[index][1] - x
              h = locations[index][2] - y
              if utils.is_valid_roi(x, y, w, h, img.shape):
                roi = img[y:y + h, x:x + w]
                cv2.imwrite(file_name, roi)
                counter_unclustered += 1

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--detections', type=str, required=True,
                      help="Path to detections.bin files.")
  parser.add_argument('--outdir', type=str, required=True,
                      help="Path to folder with clustered faces.")
  parser.add_argument('--threshold', type=float, default=0.45,
                      help="Threshold for clustering (default=0.45). A larger value decreases the number of resulting clusters.")
  parser.add_argument('--min_members', type=float, default=2,
                      help="Minimum number of members for a cluster to be accepted.")
  parser.add_argument('--db', type=str, required=False,
                      help="Path to folder with predicted faces (.csv files).")
  parser.add_argument('--recompute', help='Recompute detections.',
                      action='store_true')
  args = parser.parse_args()

  if os.path.isdir(args.detections):
    if not os.path.isdir(args.outdir):
      utils.mkdir_p(args.outdir)

    print('Clustering faces in {}'.format(args.detections))
    cluster_faces(args)
    utils.sort_folders_by_nr_of_images(args.outdir)
  else:
    if not os.path.isdir(args.db):
      print('{} is no valid directory.'.format(args.db))
      exit()
    print('Clustering faces in class {}'.format(args.detections))
    cluster_faces_in_class(args)

  print('Done.')

if __name__ == "__main__":
  main()