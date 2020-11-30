import numpy as np
import sys
from random import shuffle
import csv
from play import Play

defense_positions = {'SS', 'DE', 'ILB', 'FS', 'CB', 'DT', 'OLB', 'NT', 'MLB'}

def read_csv(file_name):
	data = []
	with open(file_name) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
			else:
				line_count += 1
				data += [row]
	return data
def mean_stand(arr):
	arr = np.array(arr)
	mean = np.mean(arr)
	std = np.std(arr)
	return mean, std

def norm(real, mean, std):
	return (real-mean)/std

def normalize(data):
	labels = []
	xpos = []
	ypos = []
	speed = []
	acc = []
	disTrav = []
	orient = []
	direction = []
	weight = []
	defe = []
	home = []
	for i in range(len(data)):
		no_orient = False
		try:
			orient.append(float(data[i][8]))  # orientation
		except ValueError:
			no_orient = True

		if no_orient:
			continue

		no_direction = False
		try:
			direction.append(float(data[i][9]))  # direction
		except ValueError:
			no_direction = True

		if no_direction:
			continue

		labels.append(data[i][31])
		defe.append(data[i][36])
		home.append(data[i][2])
		xpos.append(float(data[i][3]))
		ypos.append(float(data[i][4]))  # y position
		speed.append(float(data[i][5]))  # speed (yards per second)
		acc.append(float(data[i][6]))  # acceleration (yards per second^2)
		disTrav.append(float(data[i][7]))  # distance traveled since snap
		weight.append(float(data[i][33]))  # weight

	xposIn = mean_stand(xpos)
	yposIn = mean_stand(ypos)
	speedIn = mean_stand(speed)
	accIn = mean_stand(acc)
	disTravIn = mean_stand(disTrav)
	orientIn = mean_stand(orient)
	directIn = mean_stand(direction)
	weightIn = mean_stand(weight)


	plays = []
	play_count = 0
	cur_play = []
	for i in range(len(labels)):
		row = []
		row += [labels[i]] #label 0
		row += [defe[i]] #defence 1
		row += [home[i]] #home away 2
		row += [norm(xpos[i], xposIn[0], xposIn[1])] #xpos 3
		row += [norm(ypos[i], yposIn[0], yposIn[1])] # 4
		row += [norm(speed[i], speedIn[0], speedIn[1])] # 5
		row += [norm(acc[i], accIn[0], accIn[1])] # 6
		row += [norm(disTrav[i], disTravIn[0], disTravIn[1])] #xpos 7
		row += [norm(orient[i], orientIn[0], orientIn[1])] #xpos 8
		row += [norm(direction[i], directIn[0], directIn[1])] #xpos 9
		row += [norm(weight[i], weightIn[0], weightIn[1])] #xpos 10

		if ((i+1) % 22) == 0:
			play_count += 1
			cur_play += [row]
			plays += [cur_play]
			cur_play = []
		else:
			cur_play += [row]
	return plays

def get_plays(data):
	plays = []
	ball_carriers = []
	for row in data:
		nodes = []
		edges = []
		label = row[0][0]
		offense = []
		defense = []
		num_nodes = 0
		for i in range(len(row)):
			node = []

			if row[i][1] in defense_positions:
				defense.append(num_nodes)
			else:
				offense.append(num_nodes)

			if row[i][2] == "away":
				node.append(0)
			else:
				node.append(1)
			node.append(float(row[i][3])) # x position
			node.append(float(row[i][4])) # y position
			node.append(float(row[i][5])) # speed (yards per second)
			node.append(float(row[i][6])) # acceleration (yards per second^2)
			node.append(float(row[i][7])) # distance traveled since snap
			try :
				node.append(float(row[i][8])) # orientation
			except ValueError:
				node.append(0)
			try:
				node.append(float(row[i][9])) # direction
			except ValueError:
				node.append(0)
			node.append(float(row[i][10])) # weight
			num_nodes += 1
			nodes += [node]

		for o in offense:
			for d in defense:
				edges.append([o, d])
		label = np.array(label).astype(np.long)
		plays += [Play(np.array(nodes, dtype=np.float32), edges, label)]

	return plays, np.array(ball_carriers)

def get_data(file_name):
	np.set_printoptions(threshold=sys.maxsize)
	test_fraction = 0.1
	plays, ball_carriers = get_plays(normalize(read_csv(file_name)))
	# np.random.shuffle(plays)
	test_length = int(np.floor(len(plays) * test_fraction))
	test = plays[:test_length]
	train = plays[test_length:]
	test_ball_carriers = ball_carriers[:test_length]
	train_ball_carriers = ball_carriers[test_length:]
	return train, test

def get_convolution_data(file_name):
	## return train, train_labels, test, test_labels
	np.set_printoptions(threshold=sys.maxsize)
	test_fraction = 0.1
	plays, ball_carriers = get_plays(normalize(read_csv(file_name)))

	test_length = int(np.floor(len(plays) * test_fraction))
	test = plays[:test_length]
	train = plays[test_length:]
	train, train_labels = convolution_play_converter(train)
	test, test_labels = convolution_play_converter(test)


	return train, train_labels, test, test_labels

def convolution_play_converter(plays):
	"""
	This function converts play data into matrices for use in
	the convolution model
	:param plays:
	:return:
	"""

	result = []
	labels = []
	for i in range(0,len(plays)):
		offense = []
		defense = []
		play = plays[i]
		play_features = play.nodes
		labels.append(play.label)

		for j in range(0, len(play_features)):
			if (j < 11):

				defense.append(play_features[j])
			else:

				offense.append(play_features[j])
		defen = np.float32(defense)
		off = np.float32(offense)

		result.append((defen, off))

	inputs = np.float32(result)
	labels = np.float32(labels)
	return inputs,labels

