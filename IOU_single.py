# 101 10 196 190
gt_box = [114,75,415,341] #(21)
#gt_box = [109,2,406,298] #61175
# gt_box = [106 7 202 192]   # swap

b_box = [101, 10, 196, 190]



width0 = gt_box[2] - gt_box[0]
height0 = gt_box[3] - gt_box[1]
width1 = b_box[2] - b_box[0]
height1 = b_box[3] - b_box[1]
max_x = max(gt_box[2], b_box[2])
min_x = min(gt_box[0], b_box[0])
width = width0 + width1 - (max_x - min_x)
max_y = max(gt_box[3], b_box[3])
min_y = min(gt_box[1], b_box[1])
height = height0 + height1 - (max_y - min_y)

interArea = width * height
boxAArea = width0 * height0
boxBArea = width1 * height1
iou = interArea / (boxAArea + boxBArea - interArea)
print(iou)


