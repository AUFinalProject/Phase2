# Authors Alexey Titov and Shir Bentabou
# Version 1.0
# Date 04.2019
# USAGE:
# python randomWhiteList.py Number
  
# libraries
import os
import sys
import random
import shutil

# Function to choice random files 
def main():
	i = int(sys.argv[1])
	choiceFiles = []
    	files = [f for f in os.listdir('.') if os.path.isfile(f)]
    	files = filter(lambda f: f.endswith(('.pdf','.PDF')), files)
	while( i != 0):
    		choice = random.choice(files)
		if choice not in choiceFiles:
			i -= 1
			choiceFiles.append(choice)
	for item in choiceFiles:
		shutil.copyfile(item, "CHOICE/{}".format(os.path.basename(item)))
  
# Driver Code 
if __name__ == '__main__': 
      
    	# Calling main() function 
    	main() 
