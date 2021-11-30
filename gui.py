import os
from PIL import Image
import glob
import tkinter as tk
from PIL import ImageTk
import utils


# # process the interaction
# def event_action(event):
#     print(repr(event))
#     event.widget.quit()
#
#
# # clicks
# def clicked(event):
#     event_action(event)
#
#
# # keys
# def key_press(event):
#     event_action(event)
#
#
# # set up the gui
# window = tkinter.Tk()
# window.bind("<Button>", clicked)
# window.bind("<Key>", key_press)
#
# # get the list of images
# # files = glob.glob("/home/chrisg/Pictures/*.jpg")
# # files += glob.glob("/home/chrisg/Pictures/*.png")
# # files += glob.glob("/home/chrisg/Pictures/*.gif")
# # files.sort(key=os.path.getmtime, reverse=True)
#
# tmp_faces, img_labels = utils.load_img_labels("/Users/mhumenbe/Library/Mobile Documents/com~apple~CloudDocs/Fotos")
# # with open(os.path.join(args.imgs_root, 'faces_single_file.bin'), 'wb') as fid:
# #     pickle.dump(tmp_faces, fid)
# faces = utils.FACES(tmp_faces)
#
# idxs = sorted(faces.dict_by_name['unknown'], key=lambda x: faces.get_face(x).timestamp, reverse=True)
#         # files = faces.get_paths(faces.dict_by_name[args.face])
# files = faces.get_paths(idxs)
#
# # for each file, display the picture
# for i in range(0, len(files)):
#     file = files[i]
#     print(file)
#     window.title(file)
#     picture = Image.open(file).resize((300,300))
#     picture1 = Image.open(files[i+1]).resize((300, 300))
#     tk_picture = ImageTk.PhotoImage(picture)
#     tk_picture1 = ImageTk.PhotoImage(picture1)
#     picture_width = picture.size[0]
#     picture_height = picture.size[1]
#     picture1_width = picture1.size[0]
#     picture1_height = picture1.size[1]
#     window.geometry("{}x{}+100+100".format(picture_width+picture1_width+10, picture_height))
#     if i == 0:
#         image_widget = tkinter.Label(window, image=tk_picture)
#         image_widget1 = tkinter.Label(window, image=tk_picture1)
#     else:
#         image_widget.configure(image=tk_picture)
#         image_widget1.configure(image=tk_picture1)
#     image_widget.place(x=0, y=0, width=picture_width, height=picture_height)
#     image_widget1.place(x=picture_width+10, y=0, width=picture1_width, height=picture1_height)
#
#     # wait for events
#     window.mainloop()

tmp_faces, img_labels = utils.load_img_labels("/Users/mhumenbe/Library/Mobile Documents/com~apple~CloudDocs/Fotos")
faces = utils.FACES(tmp_faces)
idxs = sorted(faces.dict_by_name['unknown'], key=lambda x: faces.get_face(x).timestamp, reverse=True)
files = faces.get_paths(idxs)

root = tk.Tk()
root.geometry("400x400")


def showimg(e):
    n = lst.curselection()
    fname = lst.get(n)
    img = tk.PhotoImage(file=fname)
    lab.config(image=img)
    lab.image = img
    print(fname)


lst = tk.Listbox(root)
lst.pack(side="left", fill=tk.Y, expand=1)
namelist = files
for fname in namelist:
    lst.insert(tk.END, fname)
lst.bind("<<ListboxSelect>>", showimg)
img = tk.PhotoImage(Image.open(files[0]).resize((300,300)))
lab = tk.Label(root, text="hello", image=img)
lab.pack(side="left")

root.mainloop()