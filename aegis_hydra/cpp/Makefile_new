# Makefile for Critical Flow HFT Strategy
# Replaces Ising model with OFI + Hawkes approach

CXX = g++
CXXFLAGS = -O3 -march=native -std=c++17 -Wall -Wextra
LDFLAGS = -lm

# Targets
DAEMON = critical_flow_daemon

# Source files
SRCS = critical_flow_daemon.cpp order_book.cpp
OBJS = $(SRCS:.cpp=.o)
HEADERS = critical_flow_strategy.h hawkes_estimator.h mlofi_calculator.h order_book.h

# Default target
all: $(DAEMON)

# Build daemon
$(DAEMON): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✅ Critical Flow Daemon built successfully"

# Compile object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(DAEMON) $(OBJS)
	@echo "🧹 Cleaned build artifacts"

# Archive old Ising code
archive:
	mkdir -p ../archive/ising_model
	mv main.cpp ising_kernel.cpp ising_engine.cpp risk_guard.h ../archive/ising_model/ 2>/dev/null || true
	@echo "📦 Archived old Ising model code"

.PHONY: all clean archive
