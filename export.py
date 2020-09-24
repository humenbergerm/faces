import os.path
import argparse
import csv
import json
import dlib
import cv2
from PIL import Image
import subprocess

import utils
import exif
from libxmp import XMPFiles, consts
from libxmp import XMPMeta

def export_album(args):
  preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)
  if len(preds_per_person) == 0:
    print('no faces loaded')
    exit()

  album_dir = os.path.join(args.outdir, 'faces')
  utils.mkdir_p(album_dir)

  for p in preds_per_person:
    print('exporting {}'.format(p))
    face_dir = os.path.join(album_dir, p)
    utils.mkdir_p(face_dir)
    for f in preds_per_person[p]:
      symlinkname = os.path.join(face_dir, os.path.basename(f[1]))
      if not os.path.islink(symlinkname):
        os.symlink(f[1], symlinkname)

  return album_dir

def export_to_json(args):
  preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)
  if len(preds_per_person) == 0:
    print('no faces loaded')
    exit()

  json_dir = os.path.join(args.db, 'exif_json')
  if not os.path.isdir(json_dir):
    utils.mkdir_p(json_dir)

  for p in preds_per_person:
    if p == 'deleted' or p == 'unknown':
      continue

    print('exporting {}'.format(p))

    for f in preds_per_person[p]:
      # check mask
      if args.mask_folder != None:
        if os.path.dirname(f[1]) != args.mask_folder:
          continue
      if not os.path.isfile(f[1]):
        continue
      json_path = json_dir + f[1][:-3] + 'json'
      if os.path.isfile(json_path):
        continue
      if not os.path.isdir(os.path.dirname(json_path)):
        utils.mkdir_p(os.path.dirname(json_path))

      arg_str = 'exiftool -json "' + f[1] + '" > "' + json_path + '"'
      os.system(arg_str)

def save_to_exif(args):
  face_prefix = 'f '
  # json_dir = os.path.join(args.db, 'exif_json')

  preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)
  if len(preds_per_person) == 0:
    print('no faces loaded')
    exit()

  keywords_files = {}

  for p in preds_per_person:
    #print('exporting {}'.format(p))
    for f in preds_per_person[p]:
      # check mask
      if args.mask_folder != None:
        if os.path.dirname(f[1]) != args.mask_folder:
          continue

      if os.path.isfile(f[1]):
        if keywords_files.get(f[1]) == None:
          keywords_files[f[1]] = []
        if p != 'unknown' and p != 'deleted':
          keywords_files[f[1]].append(face_prefix + p)

  if args.mask_folder == None:
    all_images = utils.get_images_in_dir_rec(args.imgs_root)
  else:
    all_images = utils.get_images_in_dir_rec(args.mask_folder)

  for i,k in enumerate(all_images):
    if args.mask_folder != None:
      if os.path.dirname(k) != args.mask_folder:
        continue
    changed = False
    print('processing exif {}/{} ... {}'.format(i, len(all_images), k))
    ex = exif.ExifEditor(k)
    # test = ex.getDictTags()
    tag = ex.getTag('Description')
    if tag != os.path.basename(os.path.dirname(k)):
      ex.setTag('Description', os.path.basename(os.path.dirname(k)))
      print('updated tag <Description>')

    # get face keywords (they start with 'f ')
    kw_faces_exif = []
    kw_others = []
    kws = ex.getKeywords()

    # multiple keywords found
    for kw in kws:
      if kw[:2] == face_prefix:
        kw_faces_exif.append(kw)
      else:
        kw_others.append(kw)

    new_kws = []

    if keywords_files.get(k) == None:
      if args.overwrite:
        changed = True
    else:
      if set(keywords_files[k]) != set(kw_faces_exif):
        new_kws = keywords_files[k]
        if not args.overwrite:
          new_kws = new_kws + kw_others
        changed = True

    if changed:
      ex.setKeywords(new_kws)

    else:
      print('no change in exif found')

def export_face_crops(args):
  preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)
  if len(preds_per_person) == 0:
    print('no faces loaded')
    exit()

  sp = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")

  for p in preds_per_person:
    if p == 'unknown' or p == 'deleted':
      continue
    face_dir = os.path.join(args.outdir, p)
    if not os.path.isdir(face_dir):
      utils.mkdir_p(face_dir)
    print('Writing {}'.format(p))
    for i,f in enumerate(preds_per_person[p]):
      face_path = os.path.join(face_dir, '{}_{:06d}.jpg'.format(p, i))
      if not os.path.isfile(face_path):
        if 1:
          utils.save_face_crop(face_path, f[1], f[0][1])
        else:
          utils.save_face_crop_aligned(sp, face_path, f[1], f[0][1])

def export_thumbnails(args):
  preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)
  files_faces = utils.get_faces_in_files(preds_per_person)
  if len(preds_per_person) == 0:
    print('no faces loaded')
    exit()

  # face_prefix = 'f '
  size = (1024, 1024)

  for f in files_faces:
    rel_path = os.path.relpath(f, args.imgs_root)
    out_path = os.path.join(args.outdir, rel_path)
    print('Writing {}'.format(f))
    if not os.path.isdir(os.path.dirname(out_path)):
      os.makedirs(os.path.dirname(out_path))

    keywords = []
    for i in files_faces[f]:
      cls, idx = i
      if cls != 'unknown' and cls != 'detected' and cls != 'deleted':
        keywords.append(cls)
    if len(keywords) == 0:
      print('only ignored keywords found -> skipping')
      continue

    if os.path.isfile(out_path):
      print('skipping')
      continue

    im = cv2.imread(f)
    im = utils.resizeCV(im, size[1])
    if f.lower().endswith(('.jpg', '.jpeg')):
      cv2.imwrite(out_path, im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    elif f.lower().endswith(('.png')):
      cv2.imwrite(out_path, im, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    else:
      print('unsupported file format of {}'.format(f))
      exit()

def export_thumbnails_of_all_images_form_root(args):
  images = utils.get_images_in_dir_rec(args.imgs_root)

  size = (1024, 1024)

  for f in images:
    rel_path = os.path.relpath(f, args.imgs_root)
    out_path = os.path.join(args.outdir, rel_path)

    if os.path.isfile(out_path):
      print('skipping')
      continue

    print('Writing {}'.format(f))
    if not os.path.isdir(os.path.dirname(out_path)):
      os.makedirs(os.path.dirname(out_path))

    # if not utils.autorotate_and_resize(f, out_path, size):
    #   path = f
    # else:
    #   path = out_path

    utils.autorotate_and_resize(f, out_path, size)

    # im = cv2.imread(path)
    # im = utils.resizeCV(im, size[1])
    # if f.lower().endswith(('.jpg', '.jpeg')):
    #   cv2.imwrite(out_path, im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    # elif f.lower().endswith(('.png')):
    #   cv2.imwrite(out_path, im, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    # else:
    #   print('unsupported file format of {}'.format(f))
    #   exit()

def prepare_face_name(str):
  face_prefix = 'f '
  return face_prefix + str

def prepare_face_names(faces_list):
  faces = []
  if len(faces_list) == 1:
    face_name = prepare_face_name(faces_list[0][0])
    faces.append(face_name)
  else:
    for i in faces_list:
      face_name = prepare_face_name(i[0])
      if not face_name in faces:
        faces.append(face_name)

  return faces

def export_to_csv(args):
  preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)
  files_faces = utils.get_faces_in_files(preds_per_person, ignore_unknown=True)
  faces_csv_path = os.path.join(args.outdir, 'faces.csv')
  faces_input_csv_path = os.path.join(args.outdir, 'faces_exiftool.csv')

  if os.path.isfile(faces_input_csv_path):
    files_faces_csv = utils.load_faces_from_keywords_csv(faces_input_csv_path)
  else:
    files_faces_csv = None

  with open(faces_csv_path, 'w+') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    header = ['SourceFile','Keywords']
    # header = ['SourceFile','Subject','XPKeywords','LastKeywordXMP','LastKeywordIPTC','UserComment']
    filewriter.writerow(header)
    for e,f in enumerate(files_faces):
      print('{}/{}'.format(e,len(files_faces)))
      if os.path.dirname(f) != args.mask_folder and args.mask_folder != None:
        continue
      relpath = './' + os.path.relpath(f, args.imgs_root)
      row = [relpath]
      faces = []
      if len(files_faces[f]) == 1:
        face_name = prepare_face_name(files_faces[f][0][0])
        faces.append(face_name)
      else:
        str = ''
        tmp_faces = []
        for i in files_faces[f]:
          face_name = prepare_face_name(i[0])
          if not face_name in tmp_faces:
            str += face_name + ','
            tmp_faces.append(face_name)
        faces.append(str[:-1])
      if files_faces_csv != None:
        if files_faces_csv.get(relpath) != None:
          if files_faces_csv[relpath].replace(' ', '') == faces[0].replace(' ', ''):
            continue
      row += faces
      # row += ['-','-','-','-','-']
      filewriter.writerow(row)

def get_xmp_keywords(xmp):
  nr_of_elements = xmp.count_array_items(consts.XMP_NS_DC, 'subject')
  keywords = []
  for i in range(1, nr_of_elements+1):
    keywords.append(xmp.get_array_item(consts.XMP_NS_DC, 'subject', i))

  return keywords

def export_to_xmp(args):
  preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)
  if len(preds_per_person) == 0:
    print('no faces loaded')
    exit()
  files_faces = utils.get_faces_in_files(preds_per_person, ignore_unknown=True)

  for f in files_faces:
    xmp_path = os.path.splitext(f)[0] + '.xmp'
    if os.path.exists(xmp_path):
      print('modifying existing file')
      with open(xmp_path, 'r') as fptr:
        strbuffer = fptr.read()
      xmp = XMPMeta()
      xmp.parse_from_str(strbuffer)
    else:
      print('creating new file')
      xmpfile = XMPFiles(file_path=f, open_forupdate=True)
      xmp = xmpfile.get_xmp()

    xmp_keywords = get_xmp_keywords(xmp)

    faces = prepare_face_names(files_faces[f])
    if not sorted(faces) == sorted(xmp_keywords):
      print('test')

    if len(xmp_keywords) == 0:
      print(xmp_keywords)

    # with open(xmp_path, 'w') as fptr:
    #   fptr.write(xmp.serialize_to_str(omit_packet_wrapper=True))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--method', type=str, required=True,
                      help="Method of export: 0 ... album, 1 ... exif, 2 ... face crops to folder, 3 ... thumbnails, 4 ... one csv file, 5 ... thumbnails of all images, 6 ... XMP files")
  parser.add_argument('--db', type=str, required=True,
                      help="Path to folder with predicted faces (.csv files).")
  parser.add_argument('--outdir', type=str,
                      help="Output directory.")
  parser.add_argument('--mask_folder', type=str, required=False, default=None,
                      help="Mask folder for faces. Only faces of images within this folder will be processed.")
  parser.add_argument('--imgs_root', type=str, required=True,
                      help="Root directory of your image library.")
  parser.add_argument('--overwrite', help='Overwrite all keywords in EXIF data.', default=False,
                      action='store_true')
  args = parser.parse_args()

  if not os.path.isdir(args.db):
    print('args.db is not a valid directory')

  if args.outdir == None:
    print('Provide output directory.')
    exit()
  if not os.path.isdir(args.outdir):
    utils.mkdir_p(args.outdir)

  if args.method == '0':
    print('Exporting faces as album.')
    album_dir = export_album(args)

    sigal_dir = os.path.join(args.outdir, 'sigal')
    cmd_str = ['sigal ', 'build ', '--config ', 'sigal.conf.py ', '--title ', 'FACES ', album_dir, ' ', sigal_dir]
    # pSigal = subprocess.Popen(cmd_str)
    # pSigal.wait()

    print('To generate a Sigal album use: {}'.format(''.join(str(e) for e in cmd_str)))
    print('Show album with: sigal serve -c sigal.conf.py {}'.format(sigal_dir))
  elif args.method == '1':
    # print('Exporting all exif from the images.')
    # export_to_json(args)
    print('Saving all faces to the images exif data.')
    save_to_exif(args)
  elif args.method == '2':
    print('Exporting all face crops to {}.'.format(args.outdir))
    export_face_crops(args)
  elif args.method == '3':
    print('Exporting all face pictures as low quality thumbails to {}.'.format(args.outdir))
    export_thumbnails(args)
  elif args.method == '4':
    print('Exporting all faces in one csv file. This can be imported using exiftool.')
    export_to_csv(args)
  elif args.method == '5':
    print('Exporting all images as low quality thumbails to {}.'.format(args.outdir))
    export_thumbnails_of_all_images_form_root(args)
  elif args.method == '6':
    print('Exporting keywords to XMP files.')
    export_to_xmp(args)

  print('Done.')

if __name__ == "__main__":
  main()