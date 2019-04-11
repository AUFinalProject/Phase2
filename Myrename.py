# Authors Alexey Titov and Shir Bentabou
# Version 1.0
# Date 04.2019
# USAGE:
# python Myrename.py StartIndex Kind 
  
# importing os module 
import os
import sys
  
# Function to rename multiple files 
def main():
    i = int(sys.argv[1])
    kind = sys.argv[2]
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    files = filter(lambda f: f.endswith(('.pdf','.PDF')), files)
    for filename in files: 
        dst =kind+"." + str(i) + ".pdf"
        src = filename 
        dst = dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 

