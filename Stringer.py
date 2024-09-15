import json
import math
import numpy as np
import cv2

LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
MIN_FRAME_LENGTH = 10

confidences_dict = {}
with open('confidences.json', 'r') as f:
    confidences_dict = json.load(f)


# Find fingerspelling separations
confidences = np.ndarray((24, len(confidences_dict)+1))

for i, conf in confidences_dict.items():
    letters = np.array(list(conf.values()))
    confidences[:, int(i)] = letters

# Each row is a letter (A-Z excluding J and Z) and each column is the frame
# We can do this by taking the derivative of the confidence values

# Derivative of confidence values
derivative = np.diff(confidences, axis=1)

# Find the frames where the derivative is high (i.e. the confidence values suddenly change)
seperations = np.where(np.abs(derivative) > 0.7)
frame_seps = np.unique(seperations[1])

# Save the frame separations to a json file
with open('frame_seps.json', 'w') as f:
    json.dump(frame_seps.tolist(), f)



# Slice up the confidence matrix using frame_seps
slices = []
start = 0
for i in frame_seps:
    slices.append(confidences[:, start:i])
    start = i

# Add the last slice
slices.append(confidences[:, start:])

# Collapse each slice into a single 1d array and get the letter
word = []
for i, slice in enumerate(slices):
    if slice.shape[1] < MIN_FRAME_LENGTH:
        continue # Skip if the slice is too short
    slices[i] = np.sum(slice, axis=1) / slice.shape[1]
    word.append((LETTERS[np.argmax(slices[i])], np.max(slices[i])))

print(word)




