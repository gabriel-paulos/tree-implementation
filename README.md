# Visual Semantic Search 

##Prerequisites:

Install the dependencies of the project by running:

```
python requirements
```

##Usage:

Use an library of images and run the vocabulary tree using the command line.

Below are the options:

*``` database, -d ```: The directory path for database images, the default folder is ``` ./data/books ```
*``` test, -t ```: The directory for the query image, the default folder is ``` ./data/test ```
*``` method, -m ```: Method to get keypoint feature, the default is SIFT 
*``` branches, -k ```: The branch factor for vocabulary tree, the default is 10
*``` levels, -l ```: The depth for the vocabulary tree, the default is 10


Tree Implementation 

The following is a scalable vocabulary tree implementation for book covers.

There are two main components: the data folder which stores the images for both the database used by the tree (in the books folder) and the query image in
the test folder.
