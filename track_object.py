import cv2
from cells import generate_cell_localizations
from bats import generate_bat_image_list, fetch_bat_localizations
from tracker import Tracker
import random
import time

bat_localizations_root_path = './Localization'
cell_localizations_root_path = './Cell-Loc'
segmentations_root_path = './Segmentation'

def generate_random_colors(count):
	return  [(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)) for i in range(count)]

cell_images, cell_localizations = generate_cell_localizations()

bat_images = generate_bat_image_list()
bat_localizations = fetch_bat_localizations(bat_localizations_root_path)

def track_object(images, localizations):
	tracker = Tracker(160, 30, 5, 100)

	track_colors = generate_random_colors(9)

	pause = False

	# Infinite loop to process video frames
	for i in range(len(images)):
		frame = images[i]

		# obtain centeroids of the objects in the frame
		centers = localizations[i]
		if (len(centers) > 0):
			# Track object with the Kalman Filter
			tracker.Update(centers)

			# Draw tracklines of identified objects
			for j in range(len(tracker.tracks)):
				if (len(tracker.tracks[j].trace) > 1):
					for k in range(len(tracker.tracks[j].trace)-1):
						# Draw trace line
						x1 = tracker.tracks[j].trace[k][0][0]
						y1 = tracker.tracks[j].trace[k][1][0]
						x2 = tracker.tracks[j].trace[k+1][0][0]
						y2 = tracker.tracks[j].trace[k+1][1][0]
						clr = tracker.tracks[j].track_id % 9
						cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
								 track_colors[clr], 2)

			# Display the resulting tracking frame
			cv2.imshow('Tracking', frame)

		# reduce frame rate
		cv2.waitKey(5)
		# time.sleep(0.1)

	cv2.destroyAllWindows()


if __name__ == "__main__":
	track_object(cell_images, cell_localizations)
	track_object(bat_images, bat_localizations)