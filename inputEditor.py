from featureDescriptor import *

import os

class inputEditor():

    def __init__(self):
        self.descriptors = []
        self.images = []

    def imageDescriptors(self, dir_path):
        
        descriptify = FeatureDescriptor()
        i = 0
        for root, _, files in os.walk(dir_path):
            for f in files:

                img_path = os.path.join(root,f)

                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f'img path {img_path} is not a png or jpg file!')
                    return

                _, descriptors = descriptify.extract_descriptors(img_path)
                self.descriptors +=[[des, img_path] for des in descriptors]
                self.images += [img_path]

        num_imgs = len(self.images)
        num_descs = len(self.descriptors)
        self.descriptors = np.array(self.descriptors, dtype=object)

        return self.descriptors, self.images, num_imgs, num_descs

    def queryDescriptors(self, dir_path):

        descriptify = FeatureDescriptor()

        des = []
        kps = []
        for root, _, files in os.walk(dir_path):
            for f in files:

                img_path = os.path.join(root,f)

                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f'img path {img_path} is not a png or jpg file!')
                    return

                keypoints, descriptors = descriptify.extract_descriptors(img_path)

                des= descriptors
                kps = keypoints
        return kps,des