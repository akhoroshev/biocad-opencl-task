CXX=g++
CXXFLAGS=-std=c++17 -I.
DEPS = config.h utils.h
OBJ = main.o
LIBS=-lOpenCL

%.o: %.c $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

biocad_opencl_task: $(OBJ)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f biocad_opencl_task main.o