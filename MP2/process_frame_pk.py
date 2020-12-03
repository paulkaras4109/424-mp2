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
    
    cluster_boxes_data = get_cluster_box_info(frame, box_info)

    task_batches = []

    known_boxes = []
    avg_box = []
    
    #student's code here
    for cluster in cluster_boxes_data:
        avg_box.append(box_area(cluster))
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

    avg_area = np.mean(avg_box)
    sd_area = np.std(avg_box)

    for box in known_boxes:
        if (box_area(box) <= 3*sd_area + avg_area) and (box_area(box) >= avg_area - (3*sd_area)):
            task = TaskEntity(frame.path, coord = box[0:4], depth = box[4])
            task_batch = TaskBatch([task], task.img_width, task.img_height, priority = task.depth)
            task_batches.append(task_batch)

        
    #print(len(task_batches))
    return task_batches