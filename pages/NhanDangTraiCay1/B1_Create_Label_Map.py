labels = [{'name':'Buoi', 'id':1}, {'name':'Cam', 'id':2}, {'name':'Coc', 'id':3}, {'name':'Khe', 'id':4}, {'name':'Mit', 'id':5}]

ANNOTATION_PATH = 'Tensorflow/workspace/annotations'

with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
