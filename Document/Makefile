SRC=template
all: $(SRC).tex
	latex $(SRC);
	latex $(SRC);
	dvipdf $(SRC);
clean:
	rm -f $(SRC).aux $(SRC).dvi $(SRC).lof $(SRC).lot $(SRC).toc $(SRC).log
