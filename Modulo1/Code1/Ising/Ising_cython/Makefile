# Simple makefile to quickly access handy build commands for Cython extension
# code generation.  Note that the actual code to produce the extension lives in
# the setup.py file, this Makefile is just meant as a command
# convenience/reminder while doing development.
help:
	@echo "Available tasks:"
	@echo "compile  -> Build the Cython extension module."
	@echo "annotate -> Create annotated HTML from the .pyx sources"
	@echo "all      -> Call compile and annotate"
	@echo "clean    -> Remove *.so *.c *.o *.html build"
	@echo "test     -> Run the test.py"

all: compile annotate

compile:
	python setup.py build_ext --inplace
annotate:
	cython -a *.pyx
	@echo "Annotated HTML of the code"
test:
	python test.py
# Phony targets for cleanup and similar uses
.PHONY: clean
clean:
	rm -rf *~ *.so *.c *.o *.html build

# Suffix rules
%.c : %.pyx
	cython $<
