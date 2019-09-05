CXX = g++
CXXFLAGS = -std=c++11 -g
TARGET = mwd
all: $(TARGET)

$(TARGET): $(TARGET).cc
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(TARGET) *.o
