import os

for filename in os.listdir('./cropped/'):
	arquivo = './cropped/' + filename
	print(arquivo)
	os.system('/home/amc/.local/bin/pdf-crop-margins -v -s -u ' + arquivo)
