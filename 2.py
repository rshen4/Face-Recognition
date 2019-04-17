import numpy as np
import cv2
from matplotlib import pyplot as plt
import random 

def match_features(descriptor1,descriptor2,keypoints1,keypoints2):
	list_kp1 = []
	list_kp2 = []
	matches = []
	difference = []
	len1 = len(keypoints1)
	len2 = len(keypoints2)

	for i in range(len1):
		orb_des1 = list(descriptor1[i])
		Error_min_1 = 1000.0
		Error_min_2 = 1000.0
		best_fit = -1
		for j in range(len2):
			orb_des2 = list(descriptor2[j])
			difference = list(map(lambda x: x[0]-x[1], zip(orb_des2, orb_des1)))
			distance = np.sqrt(sum([x**2 for x in difference]))
			if distance < Error_min_1:
				Error_min_2 = Error_min_1
				Error_min_1 = distance
				best_fit = j
			elif distance < Error_min_2:
				Error_min_2 = distance
		ratio = Error_min_1/Error_min_2
		if ratio < 0.85:
			matches.append([i,best_fit])
			(x1,y1) = keypoints1[i].pt
			(x2,y2) = keypoints2[best_fit].pt
			list_kp1.append((int(round(x1)),int(round(y1))))
			list_kp2.append((int(round(x2)),int(round(y2))))
	print(matches)
	return matches,list_kp1,list_kp2

def compute_fundamental(matches,list_kp1,list_kp2):
	fun_matrix = np.zeros((3,3))
	R = 5000
	inliers_max = 0
	list_inliers = []
	list_outliers = []

	for i in range(R):
		list_inliers_loop = []
		list_outliers_loop = []
		sample = [random.randint(0, len(matches)-1) for i in range(8)]
		if len(sample)!=len(set(sample)):
			continue
		Matrix = np.zeros((8,9),dtype=float)
		
		for i in range(len(sample)):
			index = sample[i]
			(u_l,v_l) = list_kp1[index]
			(u_r,v_r) = list_kp2[index]
			Matrix[i,:] = [u_l*u_r, u_l*v_r, u_l, v_l*u_r, v_l*v_r, v_l, u_r, v_r, 1]
			
		Matrix_cal = np.dot(Matrix.T,Matrix)
		eigenvalues, eigenvectors = np.linalg.eig(Matrix_cal)
		eigenvalues_index = eigenvalues.argsort()
		eigenvalues_smallest = eigenvalues_index[0]
		eigenvectors_related = eigenvectors[:,eigenvalues_smallest]
		Trans_matrix = eigenvectors_related.reshape((3,3))

		inliers = 0
		l_position = []
		r_position = []
		for j in range(len(matches)):
			(u_l_test,v_l_test) = list_kp1[j]
			(u_r_test,v_r_test) = list_kp2[j]
			l_position = [u_l_test,v_l_test,1]
			r_position = [u_r_test,v_r_test,1]
			epipo = np.dot(l_position,Trans_matrix)
			a = epipo[0]
			b = epipo[1]
			c = epipo[2]
			y_distance = abs((-c-a*u_r_test)/b-v_r_test)
			distance = b/(np.sqrt(a**2+b**2))*y_distance
			distance = abs(distance)

			threshold = 10  
			if distance < threshold:
				inliers +=1
				list_inliers_loop.append(j)
			else:
				list_outliers_loop.append(j)
		if inliers > inliers_max:
			inliers_max = inliers
			fun_matrix = Trans_matrix
			list_inliers = list_inliers_loop
			list_outliers = list_outliers_loop

	return [fun_matrix,list_inliers,list_outliers]

def epipolar_line(fundamental_matrix,testpoint):
	
	epipolar_coe = []
	for element in testpoint:
		test_position = [element[0], element[1], 1]
		epipo = np.dot(test_position,fundamental_matrix)
		a = epipo[0]
		b = epipo[1]
		c = epipo[2]
		epipolar_coe.append([a,b,c])

	return epipolar_coe

image1 = cv2.imread('hopkins1.jpg')
image2 = cv2.imread('hopkins2.jpg') 

image_combine = np.concatenate((image1,image2),axis=1)
orb = cv2.ORB_create(nfeatures=600)
kp1, des1 = orb.detectAndCompute(image1,None)
kp2, des2 = orb.detectAndCompute(image2,None)

matches,list_kp1,list_kp2 = match_features(des1,des2,kp1,kp2)

cols_image_1 = image1.shape[1]
for index in range(len(matches)):
	x1 = list_kp1[index][0]
	y1 = list_kp1[index][1]
	x2 = list_kp2[index][0] + cols_image_1
	y2 = list_kp2[index][1]
	cv2.circle(image_combine, (x1,y1), 3, (0,255,0),-1)
	cv2.circle(image_combine, (x2,y2), 3, (0,255,0),-1)
	cv2.line(image_combine,(x1,y1),(x2,y2),(0,255,0),1)
cv2.imwrite("feature_matching.jpg",image_combine)


output_fundamental = compute_fundamental(matches,list_kp1,list_kp2)
cols_image_1 = image1.shape[1]
for index in output_fundamental[1]:
	x1 = list_kp1[index][0]
	y1 = list_kp1[index][1]
	x2 = list_kp2[index][0] + cols_image_1
	y2 = list_kp2[index][1]
	cv2.circle(image_combine, (x1,y1), 3, (0,255,0),-1)
	cv2.circle(image_combine, (x2,y2), 3, (0,255,0),-1)
	cv2.line(image_combine,(x1,y1),(x2,y2),(0,0,255),1)
for index in output_fundamental[2]:
	x1 = list_kp1[index][0]
	y1 = list_kp1[index][1]
	x2 = list_kp2[index][0] + cols_image_1
	y2 = list_kp2[index][1]
	cv2.circle(image_combine, (x1,y1), 3, (0,255,0),-1)
	cv2.circle(image_combine, (x2,y2), 3, (0,255,0),-1)
	cv2.line(image_combine,(x1,y1),(x2,y2),(0,255,0),1)
print(output_fundamental[0])
print(len(output_fundamental[1]))

image_epipo = np.concatenate((image1,image2),axis=1)
fundamental_matr = output_fundamental[0]
test_point = list_kp1[:9]

coe = epipolar_line(fundamental_matr,test_point)
for element in test_point:
	x_test_point = element[0]
	y_test_point = element[1]
	cv2.circle(image_epipo, (x_test_point,y_test_point), 3, (0,255,0),-1)

for element in coe:
	x_epipo_1 = int(round(-element[2]/element[0] + cols_image_1))
	y_epipo_1 = 0
	x_epipo_2 = cols_image_1
	y_epipo_2 = int(round(-element[2]/element[1]))
	cv2.line(image_epipo,(x_epipo_1,y_epipo_1),(x_epipo_2,y_epipo_2),(0,0,255),1)
cv2.imwrite("ransanc.jpg",image_combine)
cv2.imwrite("epipolarline.jpg",image_epipo)

