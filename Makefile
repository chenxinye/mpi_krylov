CXX = mpicxx
CXXFLAGS = -O3 -std=c++17 -Wall

OBJS = main.o matrix.o preconditioner.o cg.o bicg.o gmres.o

all: solver

solver: $(OBJS)
	$(CXX) $(CXXFLAGS) -o solver $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.o solver
