from scheduling.misc import *

class Image:
    """ Image class.

    A object with image data and original image path. 
    """
    def __init__(self, path):
        self.path = path
        # setting to 0 to save some memory space
        self.image = 0
        

class TaskBatch:
    """TaskBatch consists of a list of TaskEntity.

    Each TaskBatch is created with a list of TaskEntity. All task should have same
    width and height. All the tasks in the batch will be executed at the same time.

    Attributes:
        tasks: a list of TaskEntity.
        batch_size: the size of TaskEntity list.
        img_width: width of the image. For full frame this is set as 1920, from Waymo dataset.
        img_height: height of the image. For full frame this is set as 1280, from Waymo dataset.
        priority: priority assigned to this task. A lower number means higher priority.
        remain_time: this filed is filled by the scheduler and used by the scheduler to 
                see if this task has finished execution.
        enqueue_time: the time instance that this task is added to the scheduler run queue.
                This field is filled by the scheduler.

    """
    def __init__(self, tasks, img_height, img_width, priority = 0):
        self.tasks = tasks
        self.batch_size = len(tasks)
        self.enqueue_time = 0
        self.remain_time = 0
        self.img_height = img_height
        self.img_width = img_width
        self.priority = priority

    def set_enqueue_time(self, time):
        """Set enqueue_time for tasks in the batch."""
        self.enqueue_time = time
        for task in self.tasks:
            task.enqueue_time = time
    
    def set_exec_time(self, time):
        """Set exec_time for tasks in the batch."""
        for task in self.tasks:
            task.exec_time = time
    
    def set_remain_time(self, time):
        """Set remain_time for tasks in the batch."""
        for task in self.tasks:
            task.remain_time = time

    def set_response_time(self, time):
        """Set response_time for tasks in the batch."""
        for task in self.tasks:
            task.response_time = time
    
    def set_task_order(self, order):
        """Set scheduling order for tasks in the batch."""
        for task in self.tasks:
            task.order = order

    """
    The following functions are used to implement comparison between TaskBatch.
    """
    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        else:
            return self.enqueue_time < other.enqueue_time
    
    def __eq__(self, other):
        return self.priority == other.priority

    def __gt__(self, other):
        if self.priority != other.priority:
            return self.priority > other.priority
        else:
            return self.enqueue_time > other.enqueue_time
    

class TaskEntity:
    """Task with image data and scheduling parameters. 

    Each task is associated with image data, original image path and cluster box coordinates. 
    Scheduling parameters priority, deadline, etc. is also included.

    Attributes:
        image_path: the path to the image.
        coord: coordinates of the related bounding box coordinates in the original image frame. 
                For full frame this field is not required.
        img_width: width of the image. For full frame this is set as 1920, from Waymo dataset.
        img_height: height of the image. For full frame this is set as 1280, from Waymo dataset.
        image_out_path: output path to store classification result and scheduling order visualization.
        priority: priority assigned to this task. A lower number means higher priority.
        depth: distance of the related bounding box in the image frame to the vehicle. 
        bbox_id: an id assigned to the bounding box in the image frame. 
        order: the order this task is executed by the scheduler. 
                This field is filled by the scheduler.
        exec_time: exec_time of the task. This field is filled by the scheduler.
        remain_time: this filed is filled by the scheduler and used by the scheduler to 
                see if this task has finished execution.
        deadline: a deadline assigned to this task. Can help in visualizing scheduling order.
                Default is set as 100, same as frame period.
        enqueue_time: the time instance that this task is added to the scheduler run queue.
                This field is filled by the scheduler.
        response_time: the response time of this task. This field is filled by the scheduler.
        missed: whether this task has missed deadline, i.e. response time > deadline.
                This field is filled by the scheduler.
    """
    def __init__(self, image_path, priority = 0, depth = 0, image_out_path = "", coord = 0, 
                bbox_id = 0):
        
        self.image_path = image_path
        self.coord = coord
        self.priority = priority
        self.depth = depth
        self.bbox_id = bbox_id
        self.set_image_out_path(image_path)
        self.order = 0
        self.exec_time = 0
        self.remain_time = 0
        self.enqueue_time = 0
        self.response_time = 0
        self.missed = 0
        
        if coord:
            self.img_width = coord[2] - coord[0]
            self.img_height = coord[3] - coord[1]
        else:
            # default image size from Waymo
            self.img_width = 1920
            self.img_height = 1280

        self.set_deadline(depth)
    
    def set_image_path(self, image_path):
        """Set image path as image file name."""
        i = image_path.rfind('/')
        self.image_path = image_path[i+1:]

    def set_image_out_path(self, image_path):
        """Set the image output path."""
        i = image_path.rfind('/')
        self.image_out_path = image_path[:i+1] + "out/" + image_path[i+1:]

    def set_deadline(self, depth):
        """Set task deadline based on depth."""
        dl_table = [30, 50, 60, 70, 80, 100, 100, 100, 100, 100]
        self.deadline = dl_table[int(depth/10)]

    def print(self):
        """Return a string showing important task information for printing."""
        return '{:<32s}{:>25s}{:>8.3f}{:>10d}{:>15d}{:>12d}{:>12.3f}{:>10d}{:>10d}'.format(
                self.image_path, list_to_str(self.coord), self.depth, self.priority, self.enqueue_time,
                self.exec_time, self.response_time, self.deadline, self.missed)

    """
    The following functions are used to implement comparison between TaskEntity.
    """
    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        else:
            return self.enqueue_time < other.enqueue_time
    
    def __eq__(self, other):
        return self.priority == other.priority

    def __gt__(self, other):
        if self.priority != other.priority:
            return self.priority > other.priority
        else:
            return self.enqueue_time > other.enqueue_time
