import math
import numpy as np 
import cv2

bats_root_path = './CS585-BatImages/Gray'

def generate_bat_image_list():
	image_list = []
	for i in range(750, 901):
		img_path = f"{bats_root_path}/CS585Bats-Gray_frame000000{i}.ppm"
		image_list.append(cv2.imread(img_path))
	print(f"{len(image_list)} bat image frames processed")
	return image_list


def fetch_bat_localizations(localizations_root_path):
	localizations = []
	for i in range(750, 901):
		localization_path = f"{localizations_root_path}/CS585Bats-Localization_frame000000{i}.txt"
		loaded_file = np.loadtxt(localization_path, delimiter=',')
		centers = []
		for i in range(len(loaded_file)):
			[x,y] = loaded_file[i]
			b = np.array([[x], [y]])
			centers.append(np.round(b))
		localizations.append(centers)
		# localizations[i] = np.loadtxt(file_list[i], delimiter=',')
	# return np.array(localizations), min_value
	return localizations

# def fetch_bat_localizations(localizations_root_path):
# 	localizations = []
# 	for i in range(750, 901):
# 		localization_path = f"{localizations_root_path}/CS585Bats-Localization_frame000000{i}.txt"
# 		loaded_file = np.loadtxt(localization_path, delimiter=',')
# 		frame_locs = []
# 		for j in range(len(loaded_file)):
# 			[x,y] = loaded_file[j]
# 			center = np.array([x, y])
# 			frame_locs.append(center)
# 		localizations.append(frame_locs)
# 		# localizations[i] = np.loadtxt(file_list[i], delimiter=',')
# 	# return np.array(localizations), min_value
# 	return localizations