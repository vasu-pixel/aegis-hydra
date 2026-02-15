
CC = g++
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
	# macOS specific flags (Homebrew libomp)
	# Check if brew libomp exists, else fallback
	OMP_FLAGS = -Xpreprocessor -fopenmp -lomp
	# Default paths for Homebrew
	CFLAGS = -O3 -march=native $(OMP_FLAGS) -fPIC -shared -std=c++17 -I/usr/local/opt/libomp/include -L/usr/local/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib
	LFLAGS = -O3 -march=native $(OMP_FLAGS) -std=c++17 -I/usr/local/opt/libomp/include -L/usr/local/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib
else
	CFLAGS = -O3 -march=native -fopenmp -fPIC -shared -std=c++17
	LFLAGS = -O3 -march=native -fopenmp -std=c++17
endif

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
	$(CC) $(LFLAGS) -o $@ $(SRC_DAEMON)

clean:
	rm -f $(TARGET) $(TARGET_ENGINE) $(TARGET_DAEMON)
