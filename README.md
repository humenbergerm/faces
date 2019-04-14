# faces
Face detection and recognition software to organize your personal photo gallery.

DRAFT: work in progress, testing in progress, documentation in progress

## Introduction

<img src="/data/overview.jpg" width="800"/>

This project builds on modern face detection and recognition algorithms to provide you an easy to use software to organize your personal photo gallery by people without uploading everything to the cloud. 

If you want to try it, I strongly recommend to play with the provided example first before you use it on your own images.

In theory, none of your images will be moved or deleted. I wanted to make sure that the initial folder structure etc. stays untouched. If you use the EXIF data export function you will obviously modify the EXIF data of your images.

However, please be careful and I cannot guarantee for anything :)

Furthermore, it is always a good idea to backup your data and images, especially the face database (see below), after and during long manual corrections or annotations. Obviously, I tried to reduce the manual work to a minimum but if you want to have a clean face database it will be inevitable. 

### Export Options

#### HTML Gallery
See example below. Use ```export.py --method 0```

#### EXIF Tags

In order to export the detected faces to the images' metadata, use ```export.py --method 1```.

This will first extract all metadata to a json file per image, second it will edit the "Keywords" data field and the "ImageDescription" data field, and third it will write the modified metadata to the images.

ATTENTION: In the current version, the two data fields will be overwritten. Thus, if you have anything stored there already, it will be lost. I will change this soon.

For metadata handling, we use [Exiftool](https://www.sno.phy.queensu.ca/~phil/exiftool/).

### Motivation

I was looking for a solution to organize my personal photo library and I also wanted to play with face detection algorithms. Furthermore, I wanted to have something easier to control (privacy, keep my own folder structure, do not modify my photos unless I want it to, etc.) than the various online solutions around.

### Inspiration

Obviously, the well working solutions from e.g. Google Photos and the recent advances in face detection and recognition inspired me. After some literature research, I found various sources such as [dlib](https://github.com/davisking/dlib) or [face_recognition](https://github.com/ageitgey/face_recognition) providing state of the art implementations. The latter provides a nice and clean way for face recognition using the command line but I was looking for a more complete solution from initial training till continous update of my photo database.

Inspired by this and motivated by my own idea of personal face recognition, I started to work on this project.

### Contribution

If you like this project and you feel motivated to contribute, let me know. Obviously, there is a lot of room for improvement in multiple aspects!

- Executables for Windows, Mac OS (there is a problem with PyInstaller + dlib), and Linux
- code optimization and cleaning (I certainly did not use the best coding practices and the full potential of python)
- parallel processing
- better/real GUI
- etc. 

I am also looking for an alternative photo gallery which also provides a nice keyword search. I would like to save all recognized faces in a database or directly in the image files and use this information to browse my pictures. The current solution with Sigal is nice but static.

### Classes
In this project a class is refered to as a unique face (or object) which can be trained for recognition. Sometimes I will use the word face and class interchangeably. Sorry for that. 

### Database Organization

All recognitions of a person are stored in a .csv and a .bin file. The .csv files can be edited manually to correct some errors which might have occurred. The detections are stored in detections.bin. This file can be deleted after calling ```predict```, see example.

### Installation

#### Requirements
- Python3 (tested with python 3.7 on Mac OS 10.14.3)
- [Homebrew](https://brew.sh/) (Mac OS only)

#### Mac OS (and Linux, but not tested yet)

1. Open a terminal

2. Generate and activate a virtual environement
```
pip3 install virtualenv
virtualenv venv_faces
source venv_faces/bin/activate
```
3. Install dependencies
```
pip3 install dlib opencv-python Pillow sklearn
```
[Exiftool](https://www.sno.phy.queensu.ca/~phil/exiftool/) if you want to use ```export.py``` for image metadata manopulation.

4. Clone repository
```
git clone https://github.com/humenbergerm/faces.git
```
5. Execute main script
```
python3 face.py
```
You should see something like this:
```
Usage: python3 face.py COMMAND

COMMAND:
detect 		... detect faces in images
cluster 	... group similar faces into clusters
train 		... train face recognition using faces in folders
show 		... show face recognition results
export 		... export face recognition results
```
If this is the case, you are all set and you can proceed with the example below.

## Example - Celebrities
In order to provide an easy to follow guide how to use these scripts to organize your personal photo gallery according to the people on the photos, I provide an example [dataset](https://www.microsoft.com/en-us/research/project/msra-cfw-data-set-of-celebrity-faces-on-the-web/) (part of the repositroy) consisting of 10 celebrities. Note that the pictures are already corretly sorted to make it easier for you to assess the recognition results. However, during the entire process we will ignore this. Thus, this example can be directly applied to you unorganized photo gallaery/library. All you need is one folder containing all the images you want to consider. Subfolders are supported.

We will first automatically extract the faces, define our training data, train the recognition models, and predict the faces.
Second, we will show how to add new images to the database.

### Initial Workflow

```data/celebrities``` contains our example dataset
```output``` will store the intermediate and final results

1. Detect the faces. This will give us the locations and the description of the faces in your images.
```
python3 face.py detect --input data/celebrities --outdir output

--input: path to your image library
--outdir: will contain the detections (detections.bin)
```
You will see something like:
```
Detecting faces in data/celebrities
Processing file (1/2174): data/celebrities/michelle obama/michelle_obama_97.jpg
saved
Processing file (2/2174): data/celebrities/michelle obama/michelle_obama_83.jpg
Processing file (3/2174): data/celebrities/michelle obama/michelle_obama_68.jpg
Processing file (4/2174): data/celebrities/michelle obama/michelle_obama_54.jpg
Processing file (5/2174): data/celebrities/michelle obama/michelle_obama_40.jpg
...
Done.
```
2. Cluster the detected faces. This will give us a set of folders containing the most similar images. Thus, each folder will correspond to one specific person.
```
python3 face.py cluster --detections output/detections.bin --outdir output/cluster --threshold 0.5

--threshold: Sensitivity of clustering (a real value between 0 and 1), the higher the less clusters. Thus, a higher value results in a higher number of different people. 
```
You will see something like this:
```
Clustering faces in output/detections.bin
Number of clusters: 55
Saving faces in clusters to output folder output/cluster/group_0
Saving faces in clusters to output folder output/cluster/group_1
Saving unclustered faces to output folder output/cluster/unclustered
Saving unclustered faces to output folder output/cluster/unclustered
...
Number of images in folders: 1673
Done.
```
3. Select the folders (people) you want to recognize in your images and give them a proper name (ideally the persons name which is on the pictures). This name will be used as class label for the faces in the folder.

The resulting folders from step 2 are sorted in descending order using the number of faces they contain. Here, I will just consider the top 10 clusters, rename them accordingly, and delete the remaing ones:
```
output/cluster/0_nr_of_images_638   -> aishwarya rai
output/cluster/1_nr_of_images_264   -> liv tyler
output/cluster/2_nr_of_images_217   -> bill gates
output/cluster/3_nr_of_images_131   -> al gore
output/cluster/4_nr_of_images_103   -> steve jobs
output/cluster/5_nr_of_images_88    -> michelle obama
output/cluster/6_nr_of_images_54    -> adam brody
output/cluster/7_nr_of_images_52    -> adrien brody
output/cluster/8_nr_of_images_37    -> aaron carter
output/cluster/9_nr_of_images_37    -> martina hingis
```
4. Train a svm and a knn model using the folders from step 3.
```
python3 face.py train --traindir output/cluster --outdir output/models

--traindir: folder containing the clustered faces
--outdir: target folder to save the trained face recognition models
```
You will see something like this:
```
Training knn and svm model using data in output/cluster.
Training using faces in subfolders.
adding michelle obama to training data
88 faces used for training
adding adam brody to training data
54 faces used for training
adding adrien brody to training data
52 faces used for training
...
Trained models with 1621 faces
Done.
```
5. Predict all faces using the trained models.
```
python3 face.py predict --detections output/detections.bin --knn output/models/knn.clf --db output/faces

--detections: detections.bin from step 1
--knn: knn.clf from step 4
--db: folder to store the face recognition results, thus, your face database
```
You will see something like this:
```
Predicting faces in output/detections.bin
loading output/detections.bin
Loading the faces from output/faces.
no csv files found in output/faces
0/2174
Found new face michelle obama.
exporting michelle obama
saved
1/2174
2/2174
Found new face michelle obama.
3/2174
Found new face michelle obama.
4/2174
...
exporting michelle obama
exporting unknown
exporting adam brody
exporting adrien brody
exporting bill gates
exporting martina hingis
exporting liv tyler
exporting al gore
exporting aishwarya rai
exporting aaron carter
exporting steve jobs
All detections processed. To make sure you do not do it again, delete output/detections.bin.
Done.
```
Note: all detections are stored in --db now. Once you processed any detections.bin file, you can delete it.

6. Optional: Since there will be wrong and missing detections you can now manually correct them.
```
python3 face.py show --face "liv tyler" --svm output/models/svm.clf --db output/faces

--face: person you want to show. e.g.: "liv tyler", "all" shows all persons in your database. Press esc to switch to the next person.
--db: face database from step 5
```
A window displaying the members of the target class will pop up and you will see something like this:
```
Showing detections of class liv tyler
Loading the faces from output/faces.
263 members of liv tyler
0: name: martina hingis, prob: 0.6585406451640466
1: name: liv tyler, prob: 0.28169683837462617
2: name: steve jobs, prob: 0.016849424576982804
3: name: aaron carter, prob: 0.012441532001513409
4: name: adam brody, prob: 0.007329329746059381
5: name: aishwarya rai, prob: 0.0064540872019714715
6: name: bill gates, prob: 0.005053292209619016
7: name: adrien brody, prob: 0.004709243538535866
8: name: al gore, prob: 0.003658970585157958
9: name: michelle obama, prob: 0.003266636601487665
```
The script ```show.py``` does not only display the members of the target class, it also uses the svm to compute the probabilities of the current image being a member of each class. We only print the top 10 in the terminal.  

This is the first image you probably get when you open the class "liv tyler" with the command from above:

<img src="/data/example_show.png" width="300"/>

As you know, this is not Liv Tyler but Martina Hingis. This can also be seen in the probabilities (martina hingis has 0.65, liv tyler only 0.28). Thus, this is a wrong recognition. We have three ways of correcting this:
1. (recommended if possible) Press the number key next to the printed probabilities. This will directly change the current face to this class. In this example press ```0```.

2. Press ```c``` and type the name of the new class. Here, type ```liv tyler```:
```
Enter new name: liv tyler
face changed: liv tyler (263)
```
Note: you do not need to type the entire name. Any number of letters is enough. If there are multiple options you will be able to select the correct one.

3. Press ```u``` to directly assign the face to the class ```unknown```. In this way you can assign it later. Or you can hope that it will be assigned automatically when you rerun the prediction using more/better training data.

Note: The window needs to be active for the keyboard buttons to work.

More information about database manipulation can be found below.

7. Export the predictions to a html gallery. 
```
python3 face.py export --method 0 --outdir output/album --db output/faces

--method: export method: 0 ... as folder with symbolic links to the original files (prepared for sigal)
--outdir: folder to store the album
--db: face database from above
```
You will see something like this:
```
Exporting faces as album.
Loading the faces from output/faces.
exporting aaron carter
exporting adam brody
exporting adrien brody
exporting aishwarya rai
exporting al gore
exporting bill gates
exporting liv tyler
exporting martina hingis
exporting michelle obama
exporting steve jobs
exporting unknown
To generate a Sigal album use: sigal build --config sigal.conf.py --title FACES output/album/faces output/album/sigal
Show album with: sigal serve -c sigal.conf.py output/album/sigal
Done.
```
Now, as writen at the end of the last script, run Sigal to generate the album:
1. If not done already, install [Sigal](http://sigal.saimon.org/en/latest/).
2. Build gallery:
```
sigal build --config sigal.conf.py --title FACES output/album/faces output/album/sigal
```
You will see something like this:
```
Collecting albums, done.
Processing files  [####################################]  1715/1715          

Done.
Processed 1715 images and 0 videos in 8.96 seconds.
```
3. Show gallery:
```
sigal serve -c sigal.conf.py output/album/sigal
```
You will see something like this:
```
DESTINATION : output/album/sigal
 * Running on http://127.0.0.1:8000/
```

Open http://127.0.0.1:8000 in your browser. It should look like this:

<img src="/data/example_sigal.png" width="600"/>

Now you have one album per person which you can easily browse. Remember, the images are links to the original files. 

8. Export the predictions to image metadata.
```
python3 export.py --method 1 --db /Users/mhumenbe/Code/faces/output/faces

--method: 1 writes the detected faces of each image into its metadata using Exiftool. 
``` 
You will see something like this:
```
Exporting all exif from the images.
Loading the faces from /Users/mhumenbe/Code/faces/output/faces.
exporting aaron carter
exporting adam brody
exporting adrien brody
exporting aishwarya rai
exporting al gore
exporting bill gates
exporting liv tyler
exporting martina hingis
exporting michelle obama
exporting steve jobs
exporting unknown
Saving all faces to the images exif data.
Loading the faces from /Users/mhumenbe/Code/faces/output/faces.
writing exif 0/1711
/Users/mhumenbe/Code/faces/data/celebrities/aaron carter/aaron_carter_30.jpg
new keywords: ['aaron carter']
    1 image files updated
...
writing exif 1602/1711
/Users/mhumenbe/Code/faces/data/celebrities/steve jobs/steve_jobs_44.jpg
new keywords: ['steve jobs']
    1 image files updated
no change in exif data found -> skipping
...
no change in exif data found -> skipping
Done.
```
All images which contain faces will have them stored in their metadata. In detail: the "IPTC Keywords" field. Furthermore, the field "ImageDiscription" will contain the name of the image's folder.

### Manipulate Recognized Faces in your Database 

#### Using ```show```:

The command ```show``` allows you to display and manipulate the face recognition results. The argument ```--face``` defines the class you open (see example above). If you provide 'all', all classes will be displayed one after the other. ```esc``` will jump to the next class. You can: 
- browse through the detections
- change the class of a face
- add a new class
- delete faces

##### Confirmed Images 

While browseing through the recognized faces using ```show```, you can maually confirm a recognition by pressing ```/```. This is useful if you want to verify the detections after prediction. By pressing button ```f```, you can fast-forward to the next face which was not confirmed yet.

##### Keyboard commands
```
.: next face
,: previous face
r: jump to a random face
esc: save and exit or switch to the next person
0 .. 9: move face to class written next to the number
u: move face to class "unknown"
c ... change class using the keyboard
s: save faces to args.db
/: ... set a face to 'confirmed'
f: ... fast-forward to the next unconfirmed face
d: ... delete a face (the face will not really be deleted, it will just be moved to the class 'deleted'
a: ... all faces in the image will be deleted (really deleted); this is useful for anonymous crowds you do not want to label manually)
b: ... undo last action
```

#### Using the .csv files

The .csv files can be edited manually. You can even merge 2 files, e.g. if you accidentally generated a person twice. If you want to merge 2 files, the according .bin files have to distinguish by exactly one letter, such as 'liv tyler' and 'liv tyler1'. 

### Extend your Gallery with new Images

If you want to extent your database and gallery with new pictures, just call ```detect``` and ```predict``` with the new image folder. The gallery can be updated using ```export```. Rerunning ```sigal build``` should just update the existing gallery. Please double-check this in the Sigal config file. 

#### Train the Recognition Models with Previous Results

Training the recognition models using the clustered faces is only necessary to initialize your database. As soon as this is done, you can use the recognition results (unknown faces will be ignored) to train the models.

```
python3 face.py train --traindir output/faces --outdir output/models_csv

--traindir: folder containing the recognition results, i.e. your faces database
--outdir: target folder to save the trained face recognition models
```
You will see something like this:
```
Training knn and svm model using data from output/faces.
Training using .csv files.
Loading the faces from output/faces.
Chose n_neighbors automatically: 40
Training model with KNN ...
Training model with an SVM ...
Trained models with 1609 faces
Done.
```

### Algorithm References

TODO 

- Face detector
- Face descriptor
- Clustering
- Matching
- Photo gallery, Sigal
- face_recognition
