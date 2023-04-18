import cv2 as cv
import numpy as np

def getHomography(query_kp,query_des, img_path_lst, method="SIFT"):
  
  max_inliers = -np.inf
  best_imgpath = None
  best_H = None
  best_mask = None
  keypoints_matched_best = None

  for img_path in img_path_lst:
    print(img_path)
    matched, keypoints_matched = match_descriptors(query_kp,query_des,img_path, method=method)
    cv_hom, mask = RANSAC(keypoints_matched)
    
    counts = mask.ravel().tolist().count(1)
    print(f'RANSAC for image: {img_path} with inliers = {counts}')

    if counts > max_inliers:
      max_inliers = counts
      best_imgpath = img_path
      best_H = cv_hom
      best_mask = mask
      keypoints_matched_best = keypoints_matched

  return best_imgpath, best_H, best_mask, keypoints_matched_best


def match_descriptors(test_kp,test_des,db_img,thres=0.8,method='SIFT'): 

  db_img = cv.imread(db_img)  
  db_img = cv.cvtColor(db_img, cv.COLOR_BGR2GRAY)

  #sift
  if method == 'SIFT':
    sift1 = cv.SIFT_create()
    sift2 = cv.SIFT_create()

    keypoints_2, descriptors_2 = sift2.detectAndCompute(db_img,None)

  elif method == 'MSER':
    mser = cv.MSER_create()
    sift = cv.SIFT_create()
    
    keypoints_2 = mser.detect(db_img)   
    descriptors_2 = sift.compute(db_img, keypoints_2) 

  closest = []
  matched = []

  print(np.shape(test_des)[0]) 
  for x in range(np.shape(test_des)[0]):
   # print(x)
    dst = np.linalg.norm(test_des[x] - descriptors_2, ord=2, axis=1)
    a = np.argpartition(dst,1)
    closed = dst[a]
    ratio = closed[0]/closed[1]
    
    if ratio < thres:
      closest.append(closed[0])
      matched.append([a[0],x])

  '''
  implot = plt.imshow(test)

  for x in matched:
    plt.plot(test_kp[x[1]].pt[0],test_kp[x[1]].pt[1],'bo')

  plt.show()

  #img = io.imread(image2)
  implot = plt.imshow(db_img)
  
  for x in matched:
    plt.plot(keypoints_2[x[0]].pt[0],keypoints_2[x[0]].pt[1],'bo')
  
  plt.show()
'''
  keypoints_matched =[]
  for x in matched:
    keypoints_matched.append([test_kp[x[1]],keypoints_2[x[0]]])
  
  return matched, keypoints_matched

def RANSAC(matched):
  '''
  best_hom = np.zeros((3,3))
  max_inliers = 0
  
  for x in range(iterations):
    lis = random.choices(matched,k=4)
    match1 = np.float32([column[1].pt for column in lis])
    match2 = np.float32([col[0].pt for col in lis])
    his, mask = cv.findHomography(np.asarray(match1),np.asarray(match2),0,30)
  
    if his is None:
      return None, None, 0
    
    inlier=0
    for trans in matched:
      p = np.asarray([trans[1].pt[0],trans[1].pt[1],1])
      p_prime = his@p
      p_real = np.asarray([trans[0].pt[0],trans[0].pt[1],1])

      if np.linalg.norm(p_prime-p_real,ord=2) < 30:
        inlier+=1

    if inlier > max_inliers:
      best_hom = his
      max_inliers = inlier
'''
  match1 = np.asarray([column[1].pt for column in matched])
  match2 = np.asarray([col[0].pt for col in matched])

  M, mask = cv.findHomography(match1, match2, cv.RANSAC,30)
  return  M, mask

