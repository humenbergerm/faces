# faces
Face detection and recognition software to organize your personal photo gallery.

## Introduction
Not 100% accurate etc... Expetations...

### Classes

### Database Organization

### Requirements
- Python 3 (tested with python 3.7 on Mac OS 10.14.3)

## Example - Celebrities
In order to provide an easy to follow guide how to use these scripts to organize your personal photo gallery according to the people on the photos, I provide an example [dataset](https://www.microsoft.com/en-us/research/project/msra-cfw-data-set-of-celebrity-faces-on-the-web/) (part of the repositroy) consisting of 10 celebrities. Note that the pictures are already corretly sorted to make it easier for you to assess the recognition results. However, during the entire process we will ignore this. Thus, this example can be directly applied to you unorganized photo gallaery/library. All you need is one folder containing all the images you want to consider. Subfolders are supported.

### Overview
First initial workflow for training
Then extention ...

### Initial Workflow

```data/celecrities``` contains our example dataset
```output``` will store the intermediate and final results

1. Detect the faces. This will give us the locations and the description of the faces in your images.
```
python3 detect.py --input data/celebrities --outdir output

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
python3 cluster.py --detections output/detections.bin --outdir output/cluster --threshold 0.5

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
python3 train.py --traindir output/cluster --outdir output/models

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
python3 predict.py --detections output/detections.bin --knn output/models/knn.clf --db output/faces

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
python3 show.py --face "liv tyler" --svm output/models/svm.clf --db output/faces

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

More information about database manipulation can be found below.

7. Export the predictions, e.g., to a html gallery. 

### Manipulate Recognized Faces in your Database 

#### Using show.py
Keyboard commands:
```
esc: save and exit or switch to the next person
u: move face to class "unknown"
```

#### Using the .csv files

### Extend your Gallery with new Images

### Algorithm References
- Face detector
- Face descriptor
- Clustering
- Matching
- Photo gallery
- Other open source projects
