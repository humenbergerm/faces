import fnmatch
import os
import sys
import argparse
import utils

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', type=str, required=True,
                      help="Input image directory. Recursive processing is supported.")
  # parser.add_argument('--imgs_root', type=str, required=True,
  #                     help="Root directory of your image library.")
  # parser.add_argument('--db', type=str, required=True,
  #                     help="Path to folder with predicted faces (.csv files).")
  # parser.add_argument('--recompute', help='Recompute detections.',
  #                     action='store_true')
  args = parser.parse_args()

  matches = []
  for root, dirnames, filenames in os.walk(args.input):
    for dirname in fnmatch.filter(dirnames, '*'):
      matches.append(os.path.join(root, dirname))

  matches.append(args.input)

  if len(matches) != 0:
    print('Recursive processing...')
    for i in matches:
      process_exif(i)

def process(i):
  print('Processing images in directory: {}'.format(i))
  foldername = os.path.basename(i)
  cur_dir = os.path.join(i, "*").replace(" ", "\ ")

  print('Renaming...')
  exifargs = "exiftool '-filename<" + foldername.replace(" ", "_") + "_${DateTimeOriginal}.%e' -d %Y-%m-%d-%H.%M.%S%%-c " + cur_dir
  os.system(exifargs)

def process_exif(i):
  files = utils.get_images_in_dir(i)
  foldername = os.path.basename(i)

  for f in files:
    ext = os.path.splitext(os.path.basename(f))[1]
    timestamp = utils.get_timestamp(f)
    new_filename = foldername.replace(" ", "_") + '_' + str(timestamp).replace(' ', '-').replace(':', '.') + ext
    new_filename = os.path.join(i, new_filename)
    print(new_filename)
    os.rename(f, new_filename)

if __name__ == "__main__":
    main()