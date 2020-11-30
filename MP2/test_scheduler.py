import time
from scheduling.Scheduler import *


scheduler = Scheduler(num_frames = 100)
scheduler.run()


cluster_box_info = read_json_file('scheduled_boxes.json')
ground_truth = read_json_file('../dataset/waymo_ground_truth_flat.json')
history = read_json_file("scheduling_history.json")

# calculate group worst response time from history file
group_response_time = get_group_worst_response_time(history)
print(group_response_time)

get_statistics(ground_truth, cluster_box_info)

# # visualize cluster boxes and ground truth boxes
# visualize_boxes('../dataset/', ground_truth, cluster_box_info)

