SOURCE_DIR = src
BUILD_DIR = build
EXEC_FILE = NN-classifier

CPU_SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cpp')
GPU_SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cu')

build: FORCE
	mkdir -p ${BUILD_DIR}
	nvcc ${CPU_SOURCE_FILES} ${GPU_SOURCE_FILES} -lineinfo -o ${BUILD_DIR}/${EXEC_FILE}

run:
	./${BUILD_DIR}/${EXEC_FILE}

clean:
	rm -rf ${BUILD_DIR}

FORCE: