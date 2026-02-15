
CC = g++
CFLAGS = -O3 -march=native -fopenmp -fPIC -shared -std=c++17
TARGET = libising.so
TARGET_ENGINE = libising_engine.so
SRC = ising_kernel.cpp
SRC_ENGINE = ising_engine.cpp

all: $(TARGET) $(TARGET_ENGINE)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC)

$(TARGET_ENGINE): $(SRC_ENGINE)
	$(CC) $(CFLAGS) -o $@ $(SRC_ENGINE)

clean:
	rm -f $(TARGET) $(TARGET_ENGINE)
