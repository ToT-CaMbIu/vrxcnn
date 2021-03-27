exe_name = main

SRCS = main.cc utils.cc

all: $(SRCS)
	gcc -o $(exe_name) -std=c++11 $^ -framework OpenCL
h5: $(SRCS)
	h5c++ -o $(exe_name) $^ -lm -framework OpenCL

clean:
	rm -rf $(exe_name) *.o .depend 