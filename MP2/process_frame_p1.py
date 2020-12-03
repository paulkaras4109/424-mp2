from scheduling.misc import *
from scheduling.TaskEntity import *


# read the input cluster box data from file
box_info = read_json_file('../dataset/depth_clustering_detection_flat.json')


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
