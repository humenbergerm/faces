import os.path
import argparse
import json
#import subprocess

import utils

def export_album(args):
  preds_per_person = utils.load_faces_from_csv(args.db)
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
  preds_per_person = utils.load_faces_from_csv(args.db)
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
  json_dir = os.path.join(args.db, 'exif_json')

  preds_per_person = utils.load_faces_from_csv(args.db)
  if len(preds_per_person) == 0:
    print('no faces loaded')
    exit()

  keywords_files = {}

  for p in preds_per_person:
    #print('exporting {}'.format(p))
    for f in preds_per_person[p]:
      if os.path.isfile(f[1]):
        if keywords_files.get(f[1]) == None:
          keywords_files[f[1]] = []
        if p != 'unknown' and p != 'deleted':
          keywords_files[f[1]].append(face_prefix + p)

  for i,k in enumerate(keywords_files):
    changed = False
    print('processing exif {}/{} ... {}'.format(i, len(keywords_files), k))
    json_path = json_dir + k[:-3] + 'json'
    if not os.path.isfile(json_path):
      continue
    with open(json_path) as fp:
      exif_image = json.load(fp)

    if exif_image[0].get('SourceFile') == None or exif_image[0].get('SourceFile') == '':
      exif_image[0]['SourceFile'] = k
      changed = True
    if 'ImageDescription' not in exif_image[0] or exif_image[0]['ImageDescription'] != os.path.basename(os.path.dirname(k)):
      exif_image[0]['ImageDescription'] = os.path.basename(os.path.dirname(k))
      changed = True
    if exif_image[0].get('Keywords') == None:
      exif_image[0]['Keywords'] = []
      changed = True

    # for kw in keywords_files[k]:
    #   if kw not in exif_image[0]['Keywords']:
    #     exif_image[0]['Keywords'].append(kw)
    #     changed = True

    # get face keywords (they start with 'f ')
    kw_faces_exif = []
    kw_others = []
    for kw in exif_image[0]['Keywords']:
      if kw[:2] == face_prefix:
        kw_faces_exif.append(kw)
      else:
        kw_others.append(kw)

    if set(keywords_files[k]) != set(kw_faces_exif):
      exif_image[0]['Keywords'] = keywords_files[k] + kw_others
      changed = True
    if changed:
      with open(json_path, 'w') as fp:
        json.dump(exif_image, fp)
      arg_str = 'exiftool -json="' + json_path + '" "' + k + '" -overwrite_original'
      os.system(arg_str)
    else:
      print('no change in exif found')

    #TODO: show changed keywords on images

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--method', type=str, required=True,
                      help="Method of export: 0 ... album, 1 ... exif")
  parser.add_argument('--db', type=str, required=True,
                      help="Path to folder with predicted faces (.csv files).")
  parser.add_argument('--outdir', type=str,
                      help="Output directory.")
  args = parser.parse_args()

  if not os.path.isdir(args.db):
    print('args.db is not a valid directory')

  if args.method == '0':
    if args.outdir == None:
      print('Provide output directory.')
      exit()
    if not os.path.isdir(args.outdir):
      utils.mkdir_p(args.outdir)

    print('Exporting faces as album.')
    album_dir = export_album(args)

    sigal_dir = os.path.join(args.outdir, 'sigal')
    cmd_str = ['sigal ', 'build ', '--config ', 'sigal.conf.py ', '--title ', 'FACES ', album_dir, ' ', sigal_dir]
    # pSigal = subprocess.Popen(cmd_str)
    # pSigal.wait()

    print('To generate a Sigal album use: {}'.format(''.join(str(e) for e in cmd_str)))
    print('Show album with: sigal serve -c sigal.conf.py {}'.format(sigal_dir))
  elif args.method == '1':
    print('Exporting all exif from the images.')
    export_to_json(args)
    print('Saving all faces to the images exif data.')
    save_to_exif(args)
    print('Done.')

if __name__ == "__main__":
  main()