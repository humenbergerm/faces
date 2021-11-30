import os.path
import argparse
import csv
import json
import dlib
import cv2
from PIL import Image
import subprocess
import pickle

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
        # print('exporting {}'.format(p))
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

    for i, k in enumerate(all_images):
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
    tmp_faces, img_labels = utils.load_img_labels(args.imgs_root)
    faces = utils.FACES(tmp_faces)

    sp = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")

    for name in faces.dict_by_name:
        if name == 'unknown' or name == 'deleted':
            continue
        face_dir = os.path.join(args.outdir, name)
        if not os.path.isdir(face_dir):
            utils.mkdir_p(face_dir)
        print('Writing {}'.format(name))
        for i, f in enumerate(faces.dict_by_name[name]):
            face_path = os.path.join(face_dir, '{}_{:06d}.jpg'.format(name, i))
            if not os.path.isfile(face_path) or args.overwrite:
                if 0:
                    utils.save_face_crop(face_path, faces.get_face_path(f), faces.get_loc(f))
                else:
                    utils.save_face_crop_aligned(sp, face_path, faces.get_face_path(f), faces.get_loc(f))


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


def prepare_name(str, prefix):
    return prefix + str


def prepare_names(names_list, prefix):
    names = []
    if len(names_list) == 1:
        name = prepare_name(names_list[0], prefix)
        names.append(name)
    else:
        for i in names_list:
            name = prepare_name(i, prefix)
            if not name in names:
                names.append(name)

    return names


def getfile_sensitive(path):
    directory, filename = os.path.split(path)
    directory = (directory or '.')
    ext = os.path.splitext(filename)[1]
    for f in os.listdir(directory):
        newpath = os.path.join(directory, f)
        if os.path.isfile(newpath) and os.path.splitext(f)[0] == os.path.splitext(filename)[0] and os.path.splitext(f)[1].lower() == ext.lower():
            if ext == os.path.splitext(f)[1]:
                return path
            else:
                return os.path.splitext(newpath)[0] + os.path.splitext(f)[1]


def export_to_csv(args):
    tmp_faces, img_labels = utils.load_img_labels(args.imgs_root)
    faces = utils.FACES(tmp_faces)

    faces_csv_path = os.path.join(args.outdir, 'faces.csv')
    faces_input_csv_path = os.path.join(args.outdir, 'faces_exiftool.csv')

    if os.path.isfile(faces_input_csv_path):
        files_faces_csv = utils.load_faces_from_keywords_csv(faces_input_csv_path)
    else:
        files_faces_csv = None

    with open(faces_csv_path, 'w+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        header = ['SourceFile', 'Keywords']
        # header = ['SourceFile','Subject','XPKeywords','LastKeywordXMP','LastKeywordIPTC','UserComment']
        filewriter.writerow(header)

        if files_faces_csv != None:
            for f in files_faces_csv:
                full_path = os.path.join(args.imgs_root, f[2:])
                real_image_path = getfile_sensitive(full_path)
                relpath = './' + os.path.relpath(real_image_path, args.imgs_root)
                if os.path.splitext(relpath)[1].lower() in ['.jpg', '.png'] and full_path not in faces.dict_by_files:
                    row = [relpath, '-']
                    filewriter.writerow(row)

        for e, f in enumerate(faces.dict_by_files):
            print('{}/{}'.format(e, len(faces.dict_by_files)))
            if os.path.dirname(f) != args.mask_folder and args.mask_folder != None:
                continue
            real_image_path = getfile_sensitive(f)
            relpath = './' + os.path.relpath(real_image_path, args.imgs_root)
            relpath_lower = os.path.splitext(relpath)[0] + os.path.splitext(relpath)[1].lower()
            row = [relpath]
            face_names = []
            tmp_faces = []
            if len(faces.dict_by_files[f]) == 1:
                face_name = prepare_name(faces.get_face(faces.dict_by_files[f][0]).name, 'f ')
                tmp_faces.append(face_name)
                if face_name not in ['f unknown', 'f deleted']:
                    face_names.append(face_name)
            else:
                str = ''
                for i in faces.dict_by_files[f]:
                    face_name = prepare_name(faces.get_face(i).name, 'f ')
                    if not face_name in tmp_faces and face_name not in ['f unknown', 'f deleted']:
                        str += face_name + ','
                        tmp_faces.append(face_name)
                if str != '':
                    face_names.append(str[:-1])
            if len(face_names) != 0:
                if files_faces_csv != None:
                    if relpath in files_faces_csv:
                        if sorted(list(files_faces_csv[relpath].split(', '))) == sorted(tmp_faces):
                            continue
                    elif relpath_lower in files_faces_csv:
                        if sorted(list(files_faces_csv[relpath_lower].split(', '))) == sorted(tmp_faces):
                            continue
                row += face_names
                filewriter.writerow(row)


def get_xmp_keywords(xmp):
    nr_of_elements = xmp.count_array_items(consts.XMP_NS_DC, 'subject')
    keywords = []
    for i in range(1, nr_of_elements + 1):
        keywords.append(xmp.get_array_item(consts.XMP_NS_DC, 'subject', i))

    return keywords


def get_keywords(xmp):
    keywords = get_xmp_keywords(xmp)
    faces = []
    tags = []
    categories = []
    miscs = []
    labels = []
    for k in keywords:
        if k[0:2] == 'f ':
            faces.append(k)
        elif k[0:2] == 't ':
            tags.append(k)
        elif k[0:2] == 'c ':
            categories.append(k)
        elif k[0:2] == 'l ':
            labels.append(k)
        else:
            miscs.append(k)

    return faces, tags, categories, labels, miscs


# def export_to_xmp(args):
#     preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)
#     if len(preds_per_person) == 0:
#         print('no faces loaded')
#         exit()
#     files_faces = utils.get_faces_in_files(preds_per_person, ignore_unknown=True)
#
#     for f in files_faces:
#         if os.path.dirname(f) != args.mask_folder and args.mask_folder != None:
#             continue
#         xmp_path = os.path.splitext(f)[0] + '.xmp'
#         if os.path.exists(xmp_path):
#             with open(xmp_path, 'r') as fptr:
#                 strbuffer = fptr.read()
#             xmp = XMPMeta()
#             xmp.parse_from_str(strbuffer)
#         else:
#             xmpfile = XMPFiles(file_path=f, open_forupdate=True)
#             xmp = xmpfile.get_xmp()
#
#         xmp_keywords = get_xmp_keywords(xmp)
#
#         faces = prepare_face_names(files_faces[f])
#         if not sorted(faces) == sorted(xmp_keywords):
#             xmp.delete_property(consts.XMP_NS_DC, 'subject')
#             xmp_keywords = get_xmp_keywords(xmp)
#             new_xmp_keywords = []
#             for face in faces:
#                 if not face in xmp_keywords:
#                     new_xmp_keywords.append(face)
#
#             for face in new_xmp_keywords:
#                 xmp.append_array_item(consts.XMP_NS_DC, 'subject', face,
#                                       {'prop_array_is_ordered': True, 'prop_value_is_array': True})
#
#             print('modifying existing file: {}'.format(os.path.basename(xmp_path)))
#             with open(xmp_path, 'w') as fptr:
#                 fptr.write(xmp.serialize_to_str(omit_packet_wrapper=True))
#         elif not os.path.exists(xmp_path):
#             print('creating new file: {}'.format(os.path.basename(xmp_path)))
#             with open(xmp_path, 'w') as fptr:
#                 fptr.write(xmp.serialize_to_str(omit_packet_wrapper=True))


def export_to_xmp_files(args):
    tmp_faces, img_labels = utils.load_img_labels(args.imgs_root)
    faces = utils.FACES(tmp_faces)

    if len(faces.dict_by_files) == 0:
        print('no faces loaded')
        exit()

    total_images = utils.get_images_in_dir_rec(args.imgs_root)

    for f in total_images:
        img_path = os.path.splitext(f)[0] + os.path.splitext(f)[1].lower()

        if os.path.dirname(img_path) != args.mask_folder and args.mask_folder != None or img_path.lower().endswith('.png'):
            continue
        xmp_path = os.path.splitext(f)[0] + '.xmp'

        if os.path.exists(xmp_path):
            with open(xmp_path, 'r') as fptr:
                strbuffer = fptr.read()
            xmp = XMPMeta()
            xmp.parse_from_str(strbuffer)
        else:
            xmpfile = XMPFiles(file_path=f, open_forupdate=True)
            xmp = xmpfile.get_xmp()

        # print(f)
        kw_faces, kw_tags, kw_categories, kw_labels, kw_miscs = get_keywords(xmp)
        kw_tags = []
        kw_categories = []
        kw_miscs = []

        if img_path in faces.dict_by_files:
            names = faces.get_names(faces.dict_by_files[img_path])
            # remove detected, deleted and unknown
            unwanted_names = {'detected', 'deleted', 'unknown'}
            names = [ele for ele in names if ele not in unwanted_names]
            face_names = prepare_names(names, 'f ')
        else:
            # if not os.path.exists(xmp_path):
            #     continue
            face_names = []

        labels = []
        if img_path in img_labels:
            if len(img_labels[img_path].tags) != 0:
                labels += prepare_names([t[0].lower() for t in img_labels[img_path].tags if t[1] >= 30], 'l ')

            if len(img_labels[img_path].categories) != 0:
                labels += prepare_names([c[0].lower() for c in img_labels[img_path].categories], 'l ')

            if hasattr(img_labels[img_path], 'gcloud_labels'):
                labels += prepare_names([l.lower() for l in img_labels[img_path].gcloud_labels], 'l ')
            if hasattr(img_labels[img_path], 'gcloud_objects'):
                labels += prepare_names([l.lower() for l in img_labels[img_path].gcloud_objects], 'l ')
            if hasattr(img_labels[img_path], 'gcloud_web'):
                labels += prepare_names([l.lower() for l in img_labels[img_path].gcloud_web], 'l ')

            if hasattr(img_labels[img_path], 'gcloud_landmarks'):
                labels += prepare_names([l.lower() for l in img_labels[img_path].gcloud_landmarks[::2]], 'l ')

            labels = list(set(labels))

        if sorted(face_names) != sorted(kw_faces) or sorted(labels) != sorted(kw_labels) or not os.path.exists(xmp_path):
            xmp.delete_property(consts.XMP_NS_DC, 'subject')

            for face in face_names:
                xmp.append_array_item(consts.XMP_NS_DC, 'subject', face,
                                      {'prop_array_is_ordered': True, 'prop_value_is_array': True})

            for l in labels:
                xmp.append_array_item(consts.XMP_NS_DC, 'subject', l,
                                      {'prop_array_is_ordered': True, 'prop_value_is_array': True})

            print('exporting file: {}'.format(os.path.basename(xmp_path)))
            with open(xmp_path, 'w') as fptr:
                fptr.write(xmp.serialize_to_str(omit_packet_wrapper=True))
        # elif not os.path.exists(xmp_path):
        #     print('creating new file: {}'.format(os.path.basename(xmp_path)))
            # with open(xmp_path, 'w') as fptr:
            #     fptr.write(xmp.serialize_to_str(omit_packet_wrapper=True))

def export_new_format(args):
    # faces = utils.load_img_labels(args.imgs_root)
    # d1, d2, d3 = utils.get_faces_dicts(faces)
    # utils.store_to_img_labels(faces, d1)

    preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)
    if len(preds_per_person) == 0:
        print('no faces loaded')
        exit()
    files_faces = utils.get_faces_in_files(preds_per_person, ignore_unknown=False)

    faces = []
    names = []
    for f in files_faces:
        (name, idx) = files_faces[f][0]
        img_timestamp = preds_per_person[name][idx][4]
        img_labels = utils.IMG_LABELS(img_timestamp)
        for (name, idx) in files_faces[f]:

            if not name in names:
                names.append(name)
                name_id = len(names) - 1
            else:
                name_id = names.index(name)

            ppp = preds_per_person[name][idx]
            loc = ppp[0][1]
            desc = ppp[2]
            timestamp = ppp[4]
            confirmed = ppp[3]
            face = utils.FACE(loc, desc, name_id, timestamp, confirmed)
            faces.append(face)
            img_labels.faces.append(face)

        # write img_label
        # bin_path = os.path.join(os.path.dirname(f), os.path.splitext(os.path.basename(f))[0] + '.pkl')
        bin_path = f + '.pkl'
        with open(bin_path, 'wb') as fid:
            pickle.dump(img_labels, fid)

        # with open(bin_path, 'rb') as fid:
        #   img_labels_test = pickle.load(fid)
        #   for face in img_labels_test.faces:
        #     face.path = f

    # write name_id mapping
    names_path = os.path.join(args.outdir, 'name_mapping.csv')
    with open(names_path, "w") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=';')
        for i, n in enumerate(names):
            filewriter.writerow([str(i), n])

    # names_test = {}
    # with open(names_path, 'r') as csvfile:
    #   filereader = csv.reader(csvfile, delimiter=';')
    #   for row in enumerate(filereader):
    #     names_test[row[0]] = row[1][1]
    # print('test')


def export_face_imgs_only(args):
    tmp_faces, img_labels = utils.load_img_labels(args.imgs_root)
    faces = utils.FACES(tmp_faces)

    for im in faces.dict_by_files:
        rel_path = os.path.relpath(im, args.imgs_root)
        symlinkname = os.path.join(args.outdir, rel_path)
        xmp_path = os.path.splitext(im)[0] + '.xmp'
        xmp_path_link = os.path.splitext(symlinkname)[0] + '.xmp'

        names = faces.get_names(faces.dict_by_files[im])
        if names.count('unknown') + names.count('deleted') + names.count('detected') == len(names):
            if os.path.exists(symlinkname):
                os.remove(symlinkname)
            if os.path.exists(xmp_path_link):
                os.remove(xmp_path_link)
        else:
            dirname = os.path.dirname(symlinkname)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            if not os.path.islink(symlinkname):
                os.symlink(im, symlinkname)
            if not os.path.islink(xmp_path_link):
                os.symlink(xmp_path, xmp_path_link)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        help="Method of export: 0 ... album, 1 ... exif, 2 ... face crops to folder, 3 ... thumbnails, 4 ... one csv file, 5 ... thumbnails of all images, 6 ... XMP files")
    parser.add_argument('--outdir', type=str,
                        help="Output directory.")
    parser.add_argument('--mask_folder', type=str, required=False, default=None,
                        help="Mask folder for faces. Only faces of images within this folder will be processed.")
    parser.add_argument('--imgs_root', type=str, required=True,
                        help="Root directory of your image library.")
    parser.add_argument('--overwrite', help='Overwrite all exiting data.', default=False,
                        action='store_true')
    args = parser.parse_args()


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
        export_to_xmp_files(args)
    elif args.method == '7':
        print('Exporting to new format.')
        export_new_format(args)
    elif args.method == '8':
        print('Exporting only face images.')
        export_face_imgs_only(args)

    print('Done.')


if __name__ == "__main__":
    main()
