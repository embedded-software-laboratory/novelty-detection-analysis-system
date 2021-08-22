@echo off
del *.aux
del *.bbl
del *.blg
rem del RWTH-Fak1.sty
pdflatex main.tex
makeindex -s main.ist -t main.alg -o main.acr main.acn
makeindex -s main.ist -t main.slg -o main.syi main.syg
texify --pdf --tex-option="-synctex=1" main.tex