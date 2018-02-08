solic:
	pdflatex solicitation_template.tex
	evince solicitation_template.pdf

all:
	python process_companies.py


clear:
	rm pdfs/*.aux
	rm pdfs/*.tex
	rm pdfs/*.tex~
	rm pdfs/*.pdf
	rm pdfs/*.log

# print:
#	IFS=$(echo -en"\n\b")
#	for f in `ls pdfs/*.pdf`; do
#	  echo $f
#       done 

print:
	pdfunite `ls pdfs/*.pdf` pdfs/all.pdf
