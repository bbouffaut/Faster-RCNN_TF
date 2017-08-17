images_index = []
with open('./data/VOCdevkit2007/VOC2007/ImageSets/Main/person_train.txt') as f:
	for line in f.readlines():
		line_content = [ n for n in line.strip().split(' ') if n ]
		if line_content[1] > 0:
			images_index.append(line_content[0])

print images_index
