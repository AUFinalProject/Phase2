# USAGE 
# pip install pyPdf
# python detectLang.py

# libraries
from pyPdf import PdfFileReader
import os



# dictionary for translate PDF language to tessaract language
lan_lst = {
	"en-us" : "eng",	"en" : "eng",	"en-za" : "eng",	"en-gb" : "eng",	"en-in" : "eng",
	"es-co" : "spa",	"es" : "spa",	"de-de" : "deu",	"fr-fr" : "fra",	"fr-ca" : "fra"
}
# dictionary for /Root/Lang 1 - except; 2 - a file have not /Root/Lang; 3 - /Root/Lang = ''; 4 - language
ans_list = dict()


# dir of folder and filter for pdf files
files = [f for f in os.listdir('trainPDF') if os.path.isfile(os.path.join('trainPDF', f))]
files = list(filter(lambda f: f.endswith(('.pdf','.PDF')), files))

f = open("Langs.txt", "w")

for filepdf in files:
	try:
		name = 'IMAGES/'+filepdf.replace('pdf','jpg')
		pdfFile = PdfFileReader(file('trainPDF/'+filepdf, 'rb'))
		catalog = pdfFile.trailer['/Root'].getObject()
		if catalog.has_key("/Lang"):
			value = 4 
			lang = catalog['/Lang'].getObject()
			if (lang == ''):
				value = 3
				f.write(filepdf+" "+lang+" value = "+str(value)+"\n")
				ans_list.update( {name : [value,'None']} )
                        else:
				lang = lang.lower()
				language = lan_lst.get(lang)
				f.write(filepdf+" "+lang+" => "+language+" value = "+str(value)+"\n")
				ans_list.update( {name : [value,language]} )
		else:
			value = 2
			f.write(filepdf+" value = "+str(value)+"\n")
			ans_list.update( {name : [value,'None']} )
	except:
		value = 1
		f.write(filepdf+' except ; value = '+str(value)+"\n")
		ans_list.update( {name : [value,'None']} )

f.close()
print(ans_list)
