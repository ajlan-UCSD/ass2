CXX = aarch64-linux-android-clang++
CXXFLAGS = -O1 -Iinclude -I/data/data/com.termux/files/usr/include/opencv4
LDFLAGS = -L/data/data/com.termux/files/usr/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs

BUILD_DIR = build
SRC_DIR = src
INCLUDE_DIR = include

SOURCES = $(wildcard $(SRC_DIR)/*.cxx)
OBJECTS = $(patsubst $(SRC_DIR)/%.cxx,$(BUILD_DIR)/%.o,$(SOURCES))
TARGET = vectest

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cxx
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

