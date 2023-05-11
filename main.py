from inputEditor import *
from homography import *
from vocabTree import *
from featureDescriptor import *

import click
import time
import matplotlib.pyplot as plt

METHOD = {
    "s":"SIFT",
    "m":"MSER"
}

@click.command()
@click.option("-d", "--database", type=click.Path(exists=True), show_default=True,
              default='./data/books', help="The directory path for database images")
@click.option("-t", "--test", type=click.Path(exists=True), show_default=True,
              default='./data/test', help="The directory for the query image")
@click.option("-m", "--method", show_default=True, default="s",
              type=click.Choice(METHOD.keys()), help="Method to get keypoint features")
@click.option("-k", "--branches", type=int, show_default=True,
              default= 10, help="The branch factor for vocabulary tree")
@click.option("-l", "--level", nargs=1, show_default= True,
              type= int, default=10, help="The depth for the vocabulary tree")
def main(database, test, method, branches, level):

    '''
    Build a vocabulary tree to match an image (from COVER_PATH) to the query (from TEST_PATH), then get a homography
    '''
    
    cover_path = database
    test_path = test
    k = branches
    l = level
    method = METHOD[method]
    
    getInput = inputEditor()

    print("Getting all images from folder...")

    print(str(cover_path))

    imageDescriptors, images, num_imgs, num_descs = getInput.imageDescriptors(cover_path)

    #print(f"Time taken to preprocess database images: {(end - start)} for {num_imgs} images that correspond to {num_descs} descriptors")

    print('Loading the images from {}, use {} for features'.format(cover_path, method))
    
    print("Building the vocab tree...")
    
    db = buildVocabTree(imageDescriptors, images, num_imgs, num_descs, k, l)

    test = test_path + '/img01.jpg'
    test = cv.imread(test)
    test_kp,test_des = get_test(test_path)
    
    best_matches = db.query(test_des)
    
    print(f"The best matches are given by: {best_matches}")

    #start = time.time()
    best_img, cv_hom, mask, keypoints_matched = getHomography(test_kp,test_des,best_matches, method= method)
    #end = time.time()

    #print(f"Time taken to get homography for vocabulary tree of branch_number {k} and level {l} and number of leaf nodes {len(db.leafs)}: {end - start}")

    print(f'Best img {best_img}, homography {cv_hom} and inliers: {mask.ravel().tolist().count(1)}')

    h,w = cv.imread(best_img, cv.IMREAD_GRAYSCALE).shape
    
    # saving all points in pts
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    
    # applying perspective algorithm
   # dst = cv.perspectiveTransform(pts, cv_hom)

    #homography = cv.polylines(test, [np.int32(dst)], True, (255, 0, 0), 3)
  
    # showing the final output 
    # with homography
    #cv.imshow("Homography", homography)
    #cv.waitKey()

    plt.imshow(test)
    for x in keypoints_matched:
        plt.plot(x[0].pt[0],x[0].pt[1],'bo')

    plt.show()
    
    selected_img = cv.imread(best_img)
    plt.imshow(selected_img)
    for x in keypoints_matched:
        plt.plot(x[1].pt[0],x[1].pt[1],'bo')

    plt.show()
    return 

def get_test(test_path):
    ind = inputEditor()

    test_kp,test_des = ind.queryDescriptors(test_path)

    return test_kp,test_des

def buildVocabTree(desc,img,num_imgs,num_descs, k, l):
    
    db = VocabTree(k,l)

    db.setImageDescriptors(desc,img,num_imgs,num_descs)

    print('Building Vocabulary Tree, with {} clusters, {} levels'.format(k, l))
    db.kmeans_hierarchy(None,0,0, db.data)

    print('Building Histgram for each images...')
    db.histogram()

    print('Building BoW for each images...')
    db.BagofWords()

    #print(f"db.leafs {db.BoW}")

    print('Building tf-idf for each leaf node...')
    db.build_tf_idf()

    print("Done building tree")

    return db

if __name__ == "__main__":
    main()
