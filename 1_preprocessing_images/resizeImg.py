import os, sys
from PIL import Image

#size = (200, 200)


target_size =128
fill_color =(0,0,0,0)

def make_square(im, w, h):
	size_ = max(target_size, w, h)
	new_im = Image.new('RGBX', (size_, size_), fill_color)
	new_im.paste(im, (abs(size_ - w) / 2, abs(size_ - h) / 2))
	return new_im


def check_size(im):
	w,h = im.size
	if (w!=h or w!= target_size):
		print "NOT SUPPOSE TO HAPPEND"



def transform(input_path, out_dir):
	try:
		outfile = os.path.join(out_dir, os.path.basename(input_path))
		#print "outfile : " + outfile
		im = Image.open(input_path)
		im = im.convert('RGB')
		w,h = im.size

		if (w < target_size or h < target_size): #if square and smaller than target size
			im = im.resize((w*2,h*2),Image.ANTIALIAS) #double the image first
			w,h = im.size
		if (w!=h):
			im = make_square(im, w, h)

		im = im.resize((target_size,target_size),Image.ANTIALIAS)
		check_size(im)
		im.save(outfile, "JPEG")
	except IOError as e:
		print (e)
		print "cannot create thumbnail for '%s'" % input_path


def pathiterator(dir, output_dir):
	for f in os.listdir(dir):
		img_full_path = os.path.join(dir, f)
		transform(img_full_path, output_dir)


def resize(in_dir,out_dir):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	pathiterator(in_dir, out_dir)
