# Phase2
1.K-NN for indicate image:

  * `pip3 install -U scikit-learn`     installing scikit-learn		
  * `pip3 install imutils`             installing imutils
  * `pip install opencv-python`        installing cv2
  * Run command `python3 newKNN.py -d DATA_FOLDER`
  <br/>
2.Save the thumbnail picture using - pdf2image:

  * `pip3 install pdf2image`
  * `pip3 install temp`
  * `pip3 install tempfile`
  * Run command `python3 new_thumbnailPDF.py`
  <br/>
3.Randomly select a group of PDF files:
  
  * Run command `python randomWhiteList.py Number`
  <br/>
4.Replace a file names of PDF to Kind.Number.pdf:
  
  * Run command `python Myrename.py StartIndex Kind`
  <br/>
5.Extract /Root/Lang attribute and choose language for tessarcat:
  
  * `pip3 install PyPDF2`
  * Run command `python3 detectLang2.py`
  <br/>
6.Detect if image is blur:
  
  * Run command `python detectBlur.py`
  <br/>
7.elbowMetod find optimale k:
  
  * Run command `python3 elbowMetod -d DATA_FOLDER`
  <br/>
8.KMC for clustering image:

  * `pip3 install seaborn`
  * Run command `python KMC.py -d DATA_FOLDER`
  <br/>
