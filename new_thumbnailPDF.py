# Authors Alexey Titov and Shir Bentabou
# Version 1.0
# Date 04.2019
# USAGE:
# python new_thumbnailPDF.py

# import the necessary packages  
import os
import tempfile
from pdf2image import convert_from_path
 
files = [f for f in os.listdir('.') if os.path.isfile(f)]
cnt_files = len(files) - 1
files = filter(lambda f: f.endswith(('.pdf','.PDF')), files)
i = 0
for filename in files:  
	with tempfile.TemporaryDirectory() as path:
	     images_from_path = convert_from_path(filename, output_folder=path, last_page=1, first_page =0)
 
	base_filename  =  os.path.splitext(os.path.basename(filename))[0] + '.jpg'     
 
	save_dir = 'IMAGE'
 
	for page in images_from_path:
	    page.save(os.path.join(save_dir, base_filename), 'JPEG')
	i += 1	
	# show an update every 500 images
	if i > 0 and i % 500 == 0:
		print("[INFO] processed {}/{}".format(i, cnt_files))
