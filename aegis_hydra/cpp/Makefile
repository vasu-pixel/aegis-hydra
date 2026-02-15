
CC = g++
CFLAGS = -O3 -march=native -fopenmp -fPIC -shared -std=c++17
TARGET = libising.so
SRC = ising_kernel.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC)

clean:
	rm -f $(TARGET)
