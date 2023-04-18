import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

from featureDescriptor import *

class VocabTree():
    
    class KNode():
        def __init__(self, level, parent, current_branch, id):
            self.clusters = [] #its cluster centers
            self.descriptors = [] #list of the descriptors it holds (only for leaf nodes)
            self.k = current_branch
            self.id = id
            self.l = level
            self.parent = parent
            self.leaf = False
            self.leaf_id = None
            self.children = [] # KNodes that succeed it

    def __init__(self, k, depth):

        self.branches = k #number of clusters
        self.l = depth #max depth
        #self.tree = {} #tree --> {KNode: []}
        self.data = [] #data --> [[descriptor, image_path]]
        self.images = [] #image --> [img_path]
        self.num_imgs = None
        self.num_descs = None 
        self.histo = {}
        self.BoW = None
        self.root = None #root of the tree, where we then can traverse down from
        self.nodes= [] #all of the visual words in the tree
        self.tf_idf = None # get the inverted word index list([visual_word_index, list(image_path)])
        self.leafs = [] #all leaf nodes stored here
        self.visual_word_id_count = 0 #total number of nodes

    def setImageDescriptors(self, descriptors, images, num_images, num_descs):

        self.data = descriptors
        self.images = images
        self.num_descs = num_descs
        self.num_imgs = num_images

    def kmeans_hierarchy(self, root, k, level, data):
        
        '''
        Build the kmeans hierarchy tree recursively as detailed in the nister paper
        '''

        #print(f"Sorting the kmeans at current depth {level} and in cluster {k}")

        if root == None:
            #print("R is none")
            root = self.KNode(level,None,0,self.visual_word_id_count)
            self.root = root
            self.nodes.append(root)
            self.visual_word_id_count+=1
            self.kmeans_hierarchy(root, 1, level+1, data)
            return
        
        if level >= self.l or len(data) < self.branches:
            root.leaf = True
            root.leaf_id = len(self.leafs)+1
            self.leafs.append(root)
            root.descriptors = np.copy(data)

            #print(f"Reached leaf node {root.leaf_id}")
            return 

        #print("Initializing Kmeans")

        descriptors = [pair[0] for pair in data]
        
        kmeans = KMeans(n_clusters=self.branches, random_state=0, n_init="auto")

        kmeans_labels = kmeans.fit_predict(descriptors)
        kmeans_clusters = kmeans.fit(descriptors)
        
        clusters = {i:self.KNode(level,root.id,i,self.visual_word_id_count+i+1) for i in range(self.branches)}

        root.children = list(clusters.values())
        root.clusters = kmeans_clusters.cluster_centers_

        self.nodes+= root.children
        self.visual_word_id_count += self.branches+1

        for i in range(self.branches):
            #print(f'cluster {i} has {len(data[kmeans_labels == i])}')
            self.kmeans_hierarchy(clusters[i],i,level+1, data[kmeans_labels == i])

    def histogram(self):     

        '''
        Dictionary of images with all of the nodes it passes through
        '''
        if self.histo == {}:
            self.histo = {i:[] for i in self.images}

        for r in self.leafs:
            for desc in r.descriptors:

                self.histo[desc[1]].append(r.leaf_id)

    def BagofWords(self):
        
        '''
        Dictionary of all the words with a vector representation of all of the children nodes
        * Note that in the BoW the zeroth number is always 0 as first leaf node id is 1
        '''
        
        self.BoW = {i:np.zeros(len(self.leafs)+1) for i in self.images}

        for k,v in self.histo.items():
            counter = Counter(v)
  
            #look at slide 14 
            for c,s in counter.items():
                self.BoW[k][c] = s

    def build_tf_idf(self):

        self.tf_idf = {r.leaf_id:[] for r in self.leafs}
        
        #print("Initialize tf_idf of tree")

        for r in self.leafs:
            self.tf_idf[r.leaf_id] = Counter(s[1] for s in r.descriptors)
        
        #print("Done tf_idf of tree")

    def word_occurrence(self):

        '''
        Get the total word occurrence in the database
        Used for tf_idf_lecture function
        '''

        self.word_occurrence_total = np.zeros(len(self.leafs)+1)

        for x in self.BoW:
            self.word_occurrence_total= self.word_occurrence_total+x

    def queryPath(self, root, descriptor):

        if len(root.children) == 0 or root.leaf == True:
            
            return root
        descriptor = np.array(descriptor)
        children = root.clusters
        scores = np.linalg.norm(children - descriptor, axis=1)

        nextNode = np.argmin(scores)

        return self.queryPath(root.children[nextNode],descriptor)

    def tf_idf_paper(self, leaf_nodes):
       
        '''
        Using d_i = w_i * m_i from nister paper
        where m_i is the number of descriptors for a database image that goes through leaf node i 
        w_i = log(N/N_i) where N is the number of images in the database and 
        N_i is the number of images with min 1 descriptor vector that goes through leaf node i
        N is the total number of images in the database
        '''
        n = len(self.images)

        d = np.zeros(len(self.leafs)+1)

        for i in range(1,len(leaf_nodes)):
            n_i = len(self.tf_idf[i])
            w_i = np.log(n/n_i)

            d[i]  = w_i * leaf_nodes[i]

        return d

    def tf_idf_lecture(self,histo):

        '''
        Using ti =nid/nd * log(N/n_i) from lecture 14 slide 19

        where n_id is the number of occurrences of word i in image d
        nd is the total number of words in image d
        n_i is the number of occurrences of word i in the whole database
        N is the number of documents in the whole database
        '''
      
        n_d = len(histo[histo > 0])
        n = len(self.images)

        for s in range(1, len(self.leafs)):
            n_i = self.word_occurrence_total[s]
            n_id = histo[s]

            histo[s] = (n_id/n_d) * np.log(n/n_i)

    def getDBImages(self, leaf_nodes):
        
        '''
        Get all of the images in the tree that share leaf nodes with the query image
        '''

        dbImages = dict()

        print("Getting database images with matching leaf nodes...")

        for visual_words in range(1,len(self.tf_idf)):

            if leaf_nodes[visual_words] > 0:
                lis = list(self.tf_idf[visual_words].keys())
                for x in lis:
                    if x not in dbImages:
                        dbImages[x] = 1
                    
        print('Successfully collected database images with matching leaf nodes')

        return list((dbImages))

    def compute_similarity(self, query_img, db_img):

        '''
        Compute the similarity score between list of candidate images and the query image using 
        equation (5) from nister paper, using L1-norm
        '''

        similarity_scores = []

        print("Getting similarity scores between candidate database images and query image...")

        for i in db_img:
            db_vector = self.tf_idf_paper(self.BoW[i])
            score = 2 + np.sum(np.abs(query_img - db_vector) - np.abs(query_img) - np.abs(db_vector))
            similarity_scores.append(score)

        print("Successfully collected similarity scores between candidate database images and query image")

        return similarity_scores

    def query(self, des):

        '''
        Find the closest image in the database to the query image
        '''
        nodes_passed  = np.zeros(len(self.leafs)+1)

        for d in des:
        
            leaf = self.queryPath(self.root, d)
            nodes_passed[leaf.leaf_id] +=1

        query_vector = self.tf_idf_paper(nodes_passed)  

        #normalized_histo_lecture = self.tf_idf_lecture(histogram)
        
        candidate_images = self.getDBImages(nodes_passed)
        
        scores = self.compute_similarity(query_vector,candidate_images)
        #print(np.array(candidate_images)[np.argpartition(scores,10)[:10]])
        best_matches = np.array(candidate_images)[np.argsort(scores)[:10]]     

        return best_matches