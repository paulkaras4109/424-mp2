import cv2
import time
import os
import numpy as np
import json
import sys


def visualize_history_file(history, Text_colors=(255,255,255)):
    """Visualize scheduling history from dictionary.

    Draw the scheduling order of bounding boxes in the image_out_path.
    Blue for box that meet deadline and red for box that missed.

    Args:
        history: A dictionary of scheduling history read from json file. 
    """
    for order in history:
        entry = history[order]
        if os.path.exists(entry["image_out_path"]):
            image = cv2.imread(entry["image_out_path"])
        else:
            image = cv2.imread(entry["image_path"])
        image_h, image_w, _ = image.shape

        bbox_color = (0,0,255) if (entry["missed"]) else (255,0,0)
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        coor = entry["coord"]
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)
        order_text = "order: " + str(order)
        # get text size
        (text_width, text_height), baseline = cv2.getTextSize(order_text, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                fontScale, thickness=bbox_thick)
        # put filled text rectangle
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

        # put text above rectangle
        cv2.putText(image, order_text, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
        
        i = entry["image_out_path"].rfind('/')
        output_directory = entry["image_out_path"][:i]
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
                
        cv2.imwrite(entry["image_out_path"], image)


def get_group_avg_response_time(history):
    """Calculate average response time for each depth group.

    Use the scheduling history to calculate the average response time for 
    each depth group. Each group is composed of objects that are in a 10m
    range, such as 0-10m, 10-20m, etc..

    Args:
        history: A dictionary of scheduling history read from json file. 
    
    Returns:
        A list of response time for each depth group. 
        For example,
        [25.293, 31.901, 9.244, 8.324, 3.987, 1.0, 0, 1.0, 1.0, 0]
    """

    res_time = [0] * 10
    group_cnt = [0] * 10
    result = []

    for key in history:
        entry = history[key]
        group_id = int(entry["depth"] / 10)
        res_time[group_id] += entry["response_time"]
        group_cnt[group_id] += 1
    
    for i in range(10):
        if group_cnt[i] != 0:
            result.append(float("{:.3f}".format(res_time[i] / group_cnt[i])))
        else:
            result.append(0)

    return result


def get_group_worst_response_time(history):
    """Calculate worst response time for each depth group.

    Use the scheduling history to calculate the average response time for 
    each depth group. Each group is composed of objects that are in a 10m
    range, such as 0-10m, 10-20m, etc..

    Args:
        history: A dictionary of scheduling history read from json file. 
    
    Returns:
        A list of response time for each depth group. 
        For example,
        [25.293, 31.901, 9.244, 8.324, 3.987, 1.0, 0, 1.0, 1.0, 0]
    """

    res_time = [0] * 10

    for key in history:
        entry = history[key]
        group_id = int(entry["depth"] / 10)
        if entry["response_time"] > res_time[group_id]:
            res_time[group_id] = entry["response_time"]

    return res_time


def extract_png_files(input_path):
    '''Find all png files within the given directory, sorted numerically.'''
    input_files = []
    file_names = os.listdir(input_path)

    for file in file_names:
        if ".png" in file:
            input_files.append(os.path.join(input_path, file))
    input_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return input_files


def read_json_file(filename):
    '''Return a dictionary read from json file.'''
    with open(filename) as json_file:
        data = json.load(json_file)
        return data


def get_cluster_box_info(frame, cluster_boxes):
    """Find cluster box information for the given frame.

    Get the cluster box information for the input frame from a dictionary.

    Args:
        frame: The image frame to be searched.
        cluster_boxes: a dictionary containing bounding box data.

    Returns:
        A list with the related bounding box data, including coordinates, depth, etc..
        For example, 
        [
            [644, 655, 729, 720, 64.44659992346784, ...],
            [571, 667, 759, 813, 29.452592092432084, ...],
            [1322, 764, 1920, 1214, 9.531812389460798, ...]
        ]
    """
    image_path = frame.path
    i = image_path.rfind('/')
    image_name = image_path[i+1:]

    if image_name in cluster_boxes:
        cluster_box_raw = cluster_boxes[image_name]
        cluster_box = []
        for entry in cluster_box_raw:
            tmp = []
            tmp.append(int(entry[0]))
            tmp.append(int(entry[1]))
            tmp.append(int(entry[2]))
            tmp.append(int(entry[3]))
            tmp.append(float(entry[4]))
            tmp.append(int(entry[5]))
            cluster_box.append(tmp)
        return cluster_box
    else:
        sys.exit("Error: no cluster box info for image {:s}".format(image_path))


def get_bbox_info(frame, cluster_boxes):
    """Find bounding box information for the given frame.

    Get the bounding box information for the input frame from a dictionary.

    Args:
        frame: The image frame to be searched.
        cluster_boxes: a dictionary containing bounding box data.

    Returns:
        A list with the related bounding box data, including coordinates, depth, etc..
        For example, 
        [
            [644, 655, 729, 720, 64.44659992346784, ...],
            [571, 667, 759, 813, 29.452592092432084, ...],
            [1322, 764, 1920, 1214, 9.531812389460798, ...]
        ]
    """
    image_path = frame.path
    i = image_path.rfind('/')
    image_name = image_path[i+1:]

    if image_name in cluster_boxes:
        cluster_box = cluster_boxes[image_name]
        return cluster_box
    else:
        sys.exit("Error: no cluster box info for image {:s}".format(image_path))


def list_to_str(l):
    """Function convert a coordinate list to string for printing"""
    return '(' + str(l[0]) + ',' + str(l[1]) + '), (' + str(l[2]) + ',' + str(l[3]) + ')'


def line_intersection(a0, a1, b0, b1):
    """Get intersection for a line.
    """
    if a0 >= b0 and a1 <= b1: # Contained
        intersection = [a0, a1]
    elif a0 < b0 and a1 > b1: # Contains
        intersection = [b0, b1]
    elif a0 < b0 and a1 > b0: # Intersects right
        intersection = [b0, a1]
    elif a1 > b1 and a0 < b1: # Intersects left
        intersection = [a0, b1]
    else: # No intersection (either side)
        intersection = 0

    return intersection


def intersection(box1, box2):
    """Find intersection of the two boxes
    """
    inter_x = line_intersection(box1[0], box1[2], box2[0], box2[2])
    inter_y = line_intersection(box1[1], box1[3], box2[1], box2[3])

    if inter_x and inter_y:
        return [inter_x[0], inter_y[0], inter_x[1], inter_y[1]]
    else:
        return 0


def set_image_pixel_value(pixels, box, value):
    """Increase the box area of pixels with specified value
    """
    for i in range(box[1]-1, box[3]-1):
        pixels[i][box[0]-1:box[2]-1] = value
    

def get_statistics_per_image(image, ground_truth, cluster_box_info):
    """Get coverage and accuracy for a single frame.
    """
    if image in ground_truth and image in cluster_box_info:
        true_boxes = ground_truth[image]
        cluster_boxes = cluster_box_info[image]
        coverage = [0] * len(true_boxes)
        cluster_statistic = [0] * len(cluster_boxes)
        pixel_array = np.zeros((len(true_boxes),1280,1920))

        i, j = 0, 0
        for entry in true_boxes:
            true_box = [entry[0], entry[1], entry[2], entry[3]]
            j = 0
            for entry2 in cluster_boxes:
                box = [int(entry2[0]), int(entry2[1]), int(entry2[2]), int(entry2[3])]
                overlap = intersection(true_box, box)
                if len(entry2) <= 5:
                    # add the sixth field
                    entry2.append(0)
                if overlap and abs(entry[4] - entry2[4]) < 10:
                    # update statistics
                    cluster_statistic[j] = 1
                    entry2[5] = 1
                    set_image_pixel_value(pixel_array[i], overlap, 1)
                j += 1
            # calculate coverage for this bounding box
            coverage[i] =  np.count_nonzero(pixel_array[i] == 1) / \
                    ((true_box[2] - true_box[0]) * (true_box[3] - true_box[1]))
            i += 1
        accuracy = sum(cluster_statistic) / len(cluster_statistic)
        return [coverage, accuracy]
    else:
        return 0


def get_statistics(ground_truth, cluster_box_info):
    """Get average coverage for bounding boxes and accuracy for cluster boxes.

    Process the ground truth bounding boxes and cluster box information to get 
    the average coverage for bounding boxes and accuracy for cluster boxes.
    This function also adds a sixth field to scheduled_boxes.json indicating 
    whether the box has some overlap with ground truth bounding boxes.

    Args:
        ground_truth: dictionary of Waymo ground truth bounding box.
        cluster_box_info: dictionary of Waymo ground truth bounding box.
    """
    avg_coverage = []
    avg_accuracy = []
    for image in ground_truth:
        result = get_statistics_per_image(image, ground_truth, cluster_box_info)
        if result:
            avg_coverage.extend(result[0])
            avg_accuracy.append(result[1])
    coverage = sum(avg_coverage) / len(avg_coverage)
    accuracy = sum(avg_accuracy) / len(avg_accuracy)

    with open('scheduled_boxes.json', 'w') as outfile:
        json.dump(cluster_box_info, outfile, ensure_ascii=False, indent=4)

    print("average coverage: %.3f" % (coverage))
    print("average accuracy: %.3f" % (accuracy))


def visualize_boxes(image_folder, ground_truth, cluster_box_info, Text_colors=(255,255,255)):
    """Visualize scheduling history from dictionary.

    Draw the scheduling order of bounding boxes in the image_out_path.
    Blue for box that meet deadline and red for box that missed.

    Args:
        history: A dictionary of scheduling history read from json file. 
    """
    for image_name in cluster_box_info:
        cluster_boxes = cluster_box_info[image_name]
        true_boxes = ground_truth[image_name]

        # get image information
        image_path = image_folder + image_name
        i = image_path.rfind('/')
        image_out_path = image_path[:i+1] + "out/" + image_path[i+1:]
        if os.path.exists(image_out_path):
            image = cv2.imread(image_out_path)
        else:
            image = cv2.imread(image_path)
        image_h, image_w, _ = image.shape
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: 
            bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        
        # draw cluster boxes
        for box in cluster_boxes:
            if len(box) < 6:
                print("scheduled_boxes.json has not been processed by get_statistics().")
                return -1
            
            bbox_color = (0,0,255) if (box[-1] == 0) else (255,0,0)
            (x1, y1), (x2, y2) = (box[0], box[1]), (box[2], box[3])

            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        # draw true bounding boxes
        for true_box in true_boxes:
            bbox_color = (0,255,0)
            (x1, y1), (x2, y2) = (true_box[0], true_box[1]), (true_box[2], true_box[3])

            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        i = image_out_path.rfind('/')
        output_directory = image_out_path[:i]
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
                
        cv2.imwrite(image_out_path, image)
