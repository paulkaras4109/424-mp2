from scheduling.misc import *
from scheduling.TaskEntity import *
import numpy as np


# read the input cluster box data from file
box_info = read_json_file('../dataset/depth_clustering_detection_flat.json')

def box_area(cluster):
    l = abs(cluster[2] - cluster[0])
    w = abs(cluster[3] - cluster[1])
    return l * w


def process_frame(frame):
    """Process frame for scheduling.

    Process a image frame to obtain cluster boxes and corresponding scheduling parameters
    for scheduling. 

    Student's code here.

    Args:
        param1: The image frame to be processed. 

    Returns:
        A list of task_batches with each task_batch containing some tasks.
    """
    
    small_box_dim = 50
    med_box_dim = 150
    large_box_dim = 250
    small_task_set = []
    med_task_set = []
    large_task_set = []

    cluster_boxes_data = get_cluster_box_info(frame, box_info)

    task_batches = []

    known_boxes = []
    avg_box = []
    
    #student's code here
    for cluster in cluster_boxes_data:
        tmp_coord = []
        tmp_coord.append(cluster[0])
        tmp_coord.append(cluster[1])
        tmp_coord.append(cluster[2])
        tmp_coord.append(cluster[3])
        task_addressed = False
        for box in known_boxes:
            if (((cluster[1] >= box[1]) and (cluster[1] <= box[3])) or ((cluster[3] >= box[1]) and (cluster[3] <= box[3]))) and (((cluster[0] >= box[0]) and (cluster[0] <= box[2])) or ((cluster[2] >= box[0]) and (cluster[2] <= box[0]))):
                tmp_box = []
                tmp_box.append(min(cluster[0], box[0]))
                tmp_box.append(min(cluster[1], box[1]))
                tmp_box.append(max(cluster[2], box[2]))
                tmp_box.append(max(cluster[3], box[3]))
                tmp_box.append(cluster[4])
                box = tmp_box
                task_addressed = True
                break
        if not(task_addressed):
            tmp_coord.append(cluster[4])
            known_boxes.append(tmp_coord)

    for box in known_boxes:
        #print(box)
        dim = 0
        l = abs(box[2] - box[0])
        w = abs(box[3] - box[1])
        dim = max(l,w)
        if(dim <= small_box_dim and box[1] + small_box_dim < 1920 and box[0] + small_box_dim < 1280):
            #print("small")
            box[3] = box[1] + small_box_dim
            box[2] = box[0] + small_box_dim
            task_small = TaskEntity(frame.path, coord = box[0:4], depth = box[4])
            small_task_set.append(task_small)
        elif(dim <= med_box_dim and box[1] + med_box_dim < 1920 and box[0] + med_box_dim < 1280):
            #print("med")
            box[3] = box[1] + med_box_dim
            box[2] = box[0] + med_box_dim
            task_med = TaskEntity(frame.path, coord = box[0:4], depth = box[4])
            med_task_set.append(task_med)
        elif(dim <= large_box_dim and box[1] + large_box_dim < 1920 and box[0] + large_box_dim < 1280):
            #print("large")
            box[3] = box[1] + large_box_dim
            box[2] = box[0] + large_box_dim
            task_large = TaskEntity(frame.path, coord = box[0:4], depth = box[4])
            large_task_set.append(task_large)
        else:
            #print("none")
            task = TaskEntity(frame.path, coord = box[0:4], depth = box[4])
            task_batch = TaskBatch([task], task.img_width, task.img_height, priority = 1) 
            task_batches.append(task_batch)

    small_task_batch = TaskBatch(small_task_set, small_box_dim, small_box_dim, priority = 4)
    med_task_batch = TaskBatch(med_task_set, med_box_dim, med_box_dim, priority = 3)
    large_task_batch = TaskBatch(large_task_set, large_box_dim, large_box_dim, priority = 2)

    task_batches.append(small_task_batch)
    task_batches.append(med_task_batch)
    task_batches.append(large_task_batch)
    #print(len(task_batches))
    return task_batches
