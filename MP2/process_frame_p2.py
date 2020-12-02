from scheduling.misc import *
from scheduling.TaskEntity import *
import itertools

# read the input cluster box data from file
box_info = read_json_file('../dataset/depth_clustering_detection_flat.json')


"""
Reduce the number of boxes by removing all boxes that are completely contained by another box.
"""

def remove_nested_boxes(cluster_a, cluster_b):
    if cluster_b[0] >= cluster_a[0] and cluster_b[1] >= cluster_a[1] and cluster_b[2] <= cluster_a[2] and cluster_b[3] <= cluster_a[3]:
        return True
    else:
        return False
    

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

    for cluster_a in cluster_boxes_data:
        for cluster_b in cluster_boxes_data:
            if remove_nested_boxes(cluster_a, cluster_b):
                cluster_boxes_data.remove(cluster_b)

    task_batches = []

    # student's code here
    for cluster in cluster_boxes_data:
        tmp_coord = []
        tmp_coord.append(cluster[0])
        tmp_coord.append(cluster[1])
        tmp_coord.append(cluster[2])
        tmp_coord.append(cluster[3])
        task = TaskEntity(frame.path, coord = tmp_coord, depth = cluster[4])
        task_batch = TaskBatch([task], task.img_width, task.img_height, priority = task.depth) 

        task_batches.append(task_batch)


    return task_batches

    