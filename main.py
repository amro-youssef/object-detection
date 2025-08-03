import argparse
import pandas as pd
import os
import matplotlib, cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import re
from itertools import combinations
import random
import csv 

### Coordinates from csv file for bounding box correspond to the top (x) left (y) corner and the bottom (x) right (y) corner ###
### Coordinate system starts from top left corner so same as numpy array indexing ###
def testTask3(iconFolderName, testFolderName):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    import time
    start_time = time.time()

    # Load images
    icon_files = []
    icon_images = []
    for filename in sorted(os.listdir(iconFolderName + "/png")):
        icon_img = cv2.imread(iconFolderName + "/png/" + filename)
        icon_files.append(filename)
        
        # Rescale the icon so that it's easier to template match from a smaller size
        rescaled_icon_img = cv2.resize(icon_img, (64, 64)) 
        grey_icon_img = cv2.cvtColor(rescaled_icon_img, cv2.COLOR_BGR2GRAY)

        icon_images.append(grey_icon_img)
        
    icon_labels = []
    for filename in icon_files:
        icon_labels.append(extract_name(filename))

    color_images = []
    test_images = []

    for filename in sorted(os.listdir(testFolderName + "/images")):
        test_img = cv2.imread(testFolderName + "/images/" + filename)
        grey_test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        color_images.append(test_img)
        test_images.append(grey_test_img)
            
    test_annotations = []
    for filename in sorted(os.listdir(testFolderName + "/annotations")):
        test_annotations.append(filename)
                
    print("Finished loading images.")
    
    test_keypoints = []
    test_descriptors = []
    
    # Find SIFT points of test images
    sift = cv2.SIFT_create(edgeThreshold=10, nOctaveLayers=4, sigma=1.8, contrastThreshold=0.01)
    for image in test_images:
        kp, des = sift.detectAndCompute(image, None)
        
        img = image
        img=cv2.drawKeypoints(image, kp, img)
        
        test_keypoints.append(kp)
        test_descriptors.append(des)
      
    icon_keypoints = []
    icon_descriptors = []
    
    # Find SIFT points of icons  
    for image in icon_images:
        kp, des = sift.detectAndCompute(image, None)
        
        img = image
        img=cv2.drawKeypoints(image, kp, img)
        
        icon_keypoints.append(kp)
        icon_descriptors.append(des)

    print("Finished finding sift points.")
    
    total_correct_predictions = 0
    total_iou = 0
    total_incorrect_predictions = 0
    total_predictions_missed = 0
    total_number_icons = 0
    
    for test_image_index in range(len(test_images)):
        filtered_matches = []
        
        bounding_boxes = []
        icon_indices = []
        
        final_image = color_images[test_image_index].copy()
        
        num_correct_predictions_for_this_image = 0
        
        for icon_index in range(len(icon_images)):
            # Find SIFT point matches between icon and training image
            matches = brute_force_match_k(icon_descriptors[icon_index], test_descriptors[test_image_index])
            
            if len(matches) <= 5:
                continue

            filtered_matches.append(matches)
                            
            # all possible ways we can pick 4 points
            combs = list(combinations(matches, 4))            
            warped_img = None
            
            random.shuffle(combs)
            
            max_inliers = 0
            best_matrix = None
            
            # apply RANSAC
            for comb in combs[:10000]:
                match1 = comb[0]
                match2 = comb[1]
                match3 = comb[2]
                match4 = comb[3]
                                    
                icon_kp1 = icon_keypoints[icon_index][match1[0]]
                test_kp1 = test_keypoints[test_image_index][match1[1]]
                
                icon_kp2 = icon_keypoints[icon_index][match2[0]]
                test_kp2 = test_keypoints[test_image_index][match2[1]]
                
                icon_kp3 = icon_keypoints[icon_index][match3[0]]
                test_kp3 = test_keypoints[test_image_index][match3[1]]
                
                icon_kp4 = icon_keypoints[icon_index][match4[0]]
                test_kp4 = test_keypoints[test_image_index][match4[1]]
                
                A = construct_linear_system_matrix(icon_kp1, test_kp1, icon_kp2, test_kp2, icon_kp3, test_kp3, icon_kp4, test_kp4)

                # Solve using SVD. Taken from lecture slides
                U, D, Vt = np.linalg.svd(A)
                H = Vt[-1, :].reshape(3, 3)
                # Normalize the solution to ensure H[2, 2] = 1
                H = H / H[2, 2] # H is our transformation matrix
                
                inliers = 0
                # count number of inliers
                for match in matches:
                    icon_kp = icon_keypoints[icon_index][match[0]]
                    test_kp = test_keypoints[test_image_index][match[1]]
                    
                    # apply transformation to image sift points
                    point = [icon_kp.pt[0], icon_kp.pt[1], 1]
                    transformed_point = H @ point
                    
                    # maybe need to normalise to make sure last element equals one?
                    transformed_point[0] = transformed_point[0] / transformed_point[2]
                    transformed_point[1] = transformed_point[1] / transformed_point[2]
                    
                    # compare transformed point with corresponding sift point of test image
                    distance = np.sqrt((transformed_point[0] - test_kp.pt[0])**2 + (transformed_point[1] - test_kp.pt[1])**2)
                    
                    if distance < 2:
                        inliers += 1
                        
                if inliers > max_inliers:
                    best_matrix = H
                    max_inliers = inliers
            
            try:
                if best_matrix == None:
                    continue
            except:
                pass
                
            start_bounding_box = np.array([[0,0,1], [64, 0 ,1], [0, 64,1], [64,64,1]]).T
            transformed_bounding_box = (best_matrix @ start_bounding_box).T

            # normalise each row so that the last element equals one
            transformed_bounding_box[1] = transformed_bounding_box[1] / transformed_bounding_box[1][2]
            transformed_bounding_box[2] = transformed_bounding_box[2] / transformed_bounding_box[2][2]
            transformed_bounding_box[3] = transformed_bounding_box[3] / transformed_bounding_box[3][2]
            
            x_coords = transformed_bounding_box[:, 0]
            y_coords = transformed_bounding_box[:, 1]

            
            # NOTE: Uncomment to show matches between icon and image
            # match_list = []
            # icon_kp_list = []
            # test_kp_list = []

            # for i, match in enumerate(matches):
            #     icon_kp = icon_keypoints[icon_index][match[0]]
            #     test_kp = test_keypoints[test_image_index][match[1]]

            #     icon_kp_list.append(icon_kp)
            #     test_kp_list.append(test_kp)

            #     icon_descriptor = icon_descriptors[icon_index][match[0]]
            #     test_descriptor = test_descriptors[test_image_index][match[1]]

            #     # convert to DMatch object so that we can visualise the matches
            #     m = cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=euclidean_distance(icon_descriptor, test_descriptor))
            #     match_list.append(m)

            # matched_img = cv2.drawMatches(icon_images[icon_index], icon_kp_list, test_images[test_image_index], test_kp_list, match_list, None)

            # # Show the result
            # plt.imshow(matched_img)
            # plt.axis('off')
            # plt.show()                
            
            # remove invalid boxes
            if np.any(transformed_bounding_box < -50) or np.any(transformed_bounding_box > 562):
                continue
                            
            if np.isclose(x_coords[0], x_coords[1]):
                continue                
            
            min_x = int(round((min(x_coords))))
            min_y = int(round((min(y_coords))))
            
            max_x = int(round((max(x_coords))))
            max_y = int(round((max(y_coords))))
            
            # remove very small bounding boxes
            if (max_x - min_x) < 15 or (max_y - min_y) < 15:
                continue
            
            df = pd.read_csv(testFolderName + "/annotations/" + test_annotations[test_image_index])
            
            max_iou = 0            
            num_icons = 0
            
            # find IOU of bounding box found compared to the closest bounding box from the ground truth
            for index, row in df.iterrows():
                classname = row['classname']
                min_x_actual = int(row['top'])
                min_y_actual = int(row['left'])
                max_x_actual = int(row['bottom'])
                max_y_actual = int(row['right'])
                
                num_icons += 1
                                
                iou = find_intersection_over_union([min_x, min_y, max_x, max_y], [min_x_actual, min_y_actual, max_x_actual, max_y_actual])
                if iou > max_iou:
                    max_iou = iou
            
            if max_iou >= 0.85:
                total_correct_predictions += 1
                num_correct_predictions_for_this_image += 1
                total_iou += max_iou
            else:
                total_incorrect_predictions += 1
            
            bounding_boxes.append([min_x, min_y, max_x, max_y, icon_labels[icon_index]])
            icon_indices.append(icon_index)
                
            # Draw bounding box on image
            # final_image = cv2.rectangle(final_image, (min_x, min_y), (max_x, max_y), color=(0, 255, 0), thickness=4)

            # Display the image with bounding box drawn one
            # plt.figure()
            # plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
            # plt.title("Bounding Box Image")
            # plt.show()

        # NOTE: Uncomment to show bounding boxes for each image
        for bounding_box, index in zip(bounding_boxes, icon_indices):
            final_img = cv2.rectangle(final_image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), color=(0, 255, 0), thickness=2)
            label = bounding_box[4]
            
            label_position = (bounding_box[2], bounding_box[3]+10)
            
            if bounding_box[2] > 450:
                label_position = (bounding_box[0], bounding_box[3]+10)
                    
            elif bounding_box[3] > 450:
                label_position = (bounding_box[0], bounding_box[1]-10)
            
            final_img = cv2.putText(final_img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 0, 0), thickness=1)
        
        # Display the image with bounding boxes drawn on
        # plt.figure()
        # plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
        # plt.title("Bounding Box Image")
        # plt.show()
        
        cv2.imwrite(f"output_image_{test_image_index+1}.png", cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))

        df = pd.read_csv(testFolderName + "/annotations/" + test_annotations[test_image_index])

        num_icons = 0
        for index, row in df.iterrows():
            classname = row['classname']
            min_x_actual = int(row['top'])
            min_y_actual = int(row['left'])
            max_x_actual = int(row['bottom'])
            max_y_actual = int(row['right'])
            
            num_icons += 1
            total_number_icons += 1
            
        total_predictions_missed += (num_icons - num_correct_predictions_for_this_image)


    average_iou = total_iou / total_correct_predictions
    print(f"Average IOU of correct predictions: {average_iou}")

    Acc = total_correct_predictions / total_number_icons
    TPR = total_correct_predictions / total_number_icons
    FPR = total_incorrect_predictions / total_number_icons
    FNR = total_predictions_missed / total_number_icons
    print(f"Accuracy: {Acc}")
    print(f"TPR: {TPR}")
    print(f"FPR: {FPR}")
    print(f"FNR: {FNR}")
    print("Runtime: %s seconds" % (time.time() - start_time))
    return (Acc,TPR,FPR,FNR)

def construct_linear_system_matrix(icon_kp1, test_kp1, icon_kp2, test_kp2, icon_kp3, test_kp3, icon_kp4, test_kp4):
    A =  A = np.zeros((8, 9))
    
    A[0,0] = icon_kp1.pt[0]
    A[0,1] = icon_kp1.pt[1]
    A[0,2] = 1
    A[0,3] = 0
    A[0,4] = 0
    A[0,5] = 0
    A[0,6] = -1 * icon_kp1.pt[0] * test_kp1.pt[0]
    A[0,7] = -1 * icon_kp1.pt[1] * test_kp1.pt[0]
    A[0,8] = -1 * test_kp1.pt[0]
    
    A[1,0] = 0
    A[1,1] = 0
    A[1,2] = 0
    A[1,3] = icon_kp1.pt[0]
    A[1,4] = icon_kp1.pt[1]
    A[1,5] = 1
    A[1,6] = -1 * icon_kp1.pt[0] * test_kp1.pt[1]
    A[1,7] = -1 * icon_kp1.pt[1] * test_kp1.pt[1]
    A[1,8] = -1 * test_kp1.pt[1]
    
    A[2,0] = icon_kp2.pt[0]
    A[2,1] = icon_kp2.pt[1]
    A[2,2] = 1
    A[2,3] = 0
    A[2,4] = 0
    A[2,5] = 0
    A[2,6] = -1 * icon_kp2.pt[0] * test_kp2.pt[0]
    A[2,7] = -1 * icon_kp2.pt[1] * test_kp2.pt[0]
    A[2,8] = -1 * test_kp2.pt[0]
    
    A[3,0] = 0
    A[3,1] = 0
    A[3,2] = 0
    A[3,3] = icon_kp2.pt[0]
    A[3,4] = icon_kp2.pt[1]
    A[3,5] = 1
    A[3,6] = -1 * icon_kp2.pt[0] * test_kp2.pt[1]
    A[3,7] = -1 * icon_kp2.pt[1] * test_kp2.pt[1]
    A[3,8] = -1 * test_kp2.pt[1]
    
    A[4,0] = icon_kp3.pt[0]
    A[4,1] = icon_kp3.pt[1]
    A[4,2] = 1
    A[4,3] = 0
    A[4,4] = 0
    A[4,5] = 0
    A[4,6] = -1 * icon_kp3.pt[0] * test_kp3.pt[0]
    A[4,7] = -1 * icon_kp3.pt[1] * test_kp3.pt[0]
    A[4,8] = -1 * test_kp3.pt[0]
    
    A[5,0] = 0
    A[5,1] = 0
    A[5,2] = 0
    A[5,3] = icon_kp3.pt[0]
    A[5,4] = icon_kp3.pt[1]
    A[5,5] = 1
    A[5,6] = -1 * icon_kp3.pt[0] * test_kp3.pt[1]
    A[5,7] = -1 * icon_kp3.pt[1] * test_kp3.pt[1]
    A[5,8] = -1 * test_kp3.pt[1]
    
    A[6,0] = icon_kp4.pt[0]
    A[6,1] = icon_kp4.pt[1]
    A[6,2] = 1
    A[6,3] = 0
    A[6,4] = 0
    A[6,5] = 0
    A[6,6] = -1 * icon_kp4.pt[0] * test_kp4.pt[0]
    A[6,7] = -1 * icon_kp4.pt[1] * test_kp4.pt[0]
    A[6,8] = -1 * test_kp4.pt[0]
    
    A[7,0] = 0
    A[7,1] = 0
    A[7,2] = 0
    A[7,3] = icon_kp4.pt[0]
    A[7,4] = icon_kp4.pt[1]
    A[7,5] = 1
    A[7,6] = -1 * icon_kp4.pt[0] * test_kp4.pt[1]
    A[7,7] = -1 * icon_kp4.pt[1] * test_kp4.pt[1]
    A[7,8] = -1 * test_kp4.pt[1]
    
    return A
    

def extract_name(filename):
    match = re.match(r"\d+-(.+)\.png", filename)
    if match:
        return match.group(1)
    return None

def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        print("Sizes of vectors do not match up")
        return -1
    
    a = 0
    for i in range(len(vector1)):
        a += math.pow((vector1[i]-vector2[i]),2)
        
    return math.sqrt(a)

def find_intersection_over_union(prediction_box, ground_truth_box):
    '''
    Boxes should be in format [x_min, y_min, x_max, y_max]
    '''
    
    x_left = max(prediction_box[0], ground_truth_box[0])
    y_top = max(prediction_box[1], ground_truth_box[1])
    
    x_right = min(prediction_box[2], ground_truth_box[2])
    y_bottom = min(prediction_box[3], ground_truth_box[3])
    
    if x_right < x_left or y_top > y_bottom:
        return 0
    
    overlap_area = (x_right - x_left) * (y_bottom - y_top)
        
    prediction_box_area = abs((prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1]))
    ground_truth_box_area = abs((ground_truth_box[2] - ground_truth_box[0]) * (ground_truth_box[3] - ground_truth_box[1]))
    
    union_area = prediction_box_area + ground_truth_box_area - overlap_area
    intersection_over_union = overlap_area / union_area
    
    return intersection_over_union
    

# find brute force matches between the descriptors of each image, returning only ones that fall above the lowe's threshold (i.e. only returning matches which are 'useful')
def brute_force_match_k(des_list_1, des_list_2, ratio=0.8):
    '''
    :param des_list_1 list of descriptors for the first image
    :param des_list_2 list of descriptors for the second image
    :return: matches - array of shape [3, num_matches_found].
    '''
    
    matches = []
    
    for i in range(len(des_list_1)):
        best_match_des1_index = 0
        best_match_des2_index = 0
        best_match_value = 99999
        
        second_best_match_value = 99999
        
        des1 = des_list_1[i]
        for j in range(len(des_list_2)):
            des2 = des_list_2[j]
            
            distance = euclidean_distance(des1, des2)
            # find the two SIFT points with the lowest distance (greatest similarity)
            if best_match_value > distance:
                # set previous best match as second best match
                second_best_match_value = best_match_value
                
                # update best match
                best_match_value = distance
                best_match_des1_index = i
                best_match_des2_index = j
                
            # find second lowest distance
            elif second_best_match_value > distance:
                second_best_match_value = distance
                
        if best_match_value < ratio * second_best_match_value:
            matches.append([best_match_des1_index, best_match_des2_index, best_match_value])
                         
    return matches

if __name__ == "__main__":

    # parsing the command line path to directories and invoking the test scripts for each task
    parser = argparse.ArgumentParser("Data Parser")
    # parser.add_argument("--Task1Dataset", help="Provide a folder that contains the Task 1 Dataset.", type=str, required=False)
    parser.add_argument("--IconDataset", help="Provide a folder that contains the Icon Dataset for Task2 and Task3.", type=str, required=False)
    # parser.add_argument("--Task2Dataset", help="Provide a folder that contains the Task 2 test Dataset.", type=str, required=False)
    parser.add_argument("--Task3Dataset", help="Provide a folder that contains the Task 3 test Dataset.", type=str, required=False)
    args = parser.parse_args()
    # if(args.Task1Dataset!=None):
    #     # This dataset has a list of png files and a txt file that has annotations of filenames and angle
    #     testTask1(args.Task1Dataset)
    # if(args.IconDataset!=None and args.Task2Dataset!=None):
    #     # The Icon dataset has a directory that contains the icon image for each file
    #     # The Task2 dataset directory has two directories, an annotation directory that contains the annotation and a png directory with list of images 
    #     testTask2(args.IconDataset,args.Task2Dataset)
    if(args.IconDataset!=None and args.Task3Dataset!=None):
        # The Icon dataset directory contains an icon image for each file
        # The Task3 dataset has two directories, an annotation directory that contains the annotation and a png directory with list of images 
        testTask3(args.IconDataset,args.Task3Dataset)
