CXX = g++
NVCC = nvcc

# Directories
SRC_DIR = image_kernel
BUILD_DIR = build

# Files
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
CU_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))

# Compiler and linker flags
CPPFLAGS = -I/usr/include/opencv4 -I/opt/cuda/include -I./image_kernel
LDFLAGS = -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
NVCCFLAGS = -arch=sm_75 -I./image_kernel

# Executable name
EXECUTABLE = $(BUILD_DIR)/image_converter

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR) *.o
