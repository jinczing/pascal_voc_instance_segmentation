import pickle
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, default='./test.pkl', help='input path')
    parser.add_argument('--out-path', type=str, default='./output.json', help='output path')
    opt = parser.parse_args()

    convert(opt.in_path, opt.out_path)


def convert(in_path='test.pkl', out_path='output.json'):
	with open(in_path, 'rb') as f:
	    data = pickle.load(f)


	instances = []
	with open('./test.json') as f:
	  test_ann = json.load(f)
	for i, d in enumerate(data):
	  for j, bbs in enumerate(d[0]):
	    if len(bbs) == 0:
	      continue
	    for k, bb in enumerate(bbs):
	      conf = float(bb[4])
	      cate = j
	      id = test_ann['images'][i]['id']
	      seg = d[1][j][k]['counts'].decode()
	      size = d[1][j][k]['size']
	      dic = {'image_id':id, 'score':conf, 'category_id':j+1, 'segmentation':{'size':size, 'counts':seg}}
	      instances.append(dic)

	with open(out_path, 'w') as f:
	  json.dump(instances, f)