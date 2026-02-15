
CC = g++
CFLAGS = -O3 -march=native -fopenmp -fPIC -shared -std=c++17
TARGET = libising.so
TARGET_ENGINE = libising_engine.so
TARGET_DAEMON = aegis_daemon
SRC = ising_kernel.cpp
SRC_ENGINE = ising_engine.cpp
SRC_DAEMON = main.cpp

all: $(TARGET) $(TARGET_ENGINE) $(TARGET_DAEMON)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC)

$(TARGET_ENGINE): $(SRC_ENGINE)
	$(CC) $(CFLAGS) -o $@ $(SRC_ENGINE)

$(TARGET_DAEMON): $(SRC_DAEMON)
	$(CC) -O3 -march=native -fopenmp -std=c++17 -o $@ $(SRC_DAEMON)

clean:
	rm -f $(TARGET) $(TARGET_ENGINE) $(TARGET_DAEMON)
