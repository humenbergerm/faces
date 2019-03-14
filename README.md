# faces
Face detection and recognition software to organize your personal photo gallery.

## Introduction

## Example - Celebrities
In order to provide an easy to follow guide how to use these scripts to organize your personal photo gallery according to the people on the photos, I provide an example [dataset](https://www.microsoft.com/en-us/research/project/msra-cfw-data-set-of-celebrity-faces-on-the-web/) (part of the repositroy) consisting of 10 celebrities. Note that the pictures are already corretly sorted to make it easier for you to assess the recognition results. However, during the entire process we will ignore this. Thus, this example can be directly applied to you unorganized photo gallaery/library. All you need is one folder containing all the images you want to consider. Subfolders are supported.

### Workflow

```data/celecrities``` contains our example dataset
```output``` will store the intermediate and final results

1. Detect the faces. This will give us the locations and the description of the faces in your images.
2. Cluster the detected faces. This will give us a set of folders containing the most similar images. Thus, each folder will correspond to one specific person.
3. Select the folders (people) you want to recognize in your images and give them a proper name (ideally the persons name which is on the pictures). This name will be used as class label for the faces in the folder.
4. Train a svm and a knn model using the folders from step 3.
5. Predict all faces using the trained models.
6. Optional: Since there will be fail detections you can now manually correct them.
7. Export the predictions, e.g., to a html gallery. 



### Algorithm References
- Face detector
- Face descriptor
- Clustering
- Matching
- Photo gallery
- Other open source projects