CC := g++
CFLAGS := -std=c++11 -I/usr/include -I/usr/local/include/opencv4
INCLUDEPATHS := -I${HOME}/tensorflow -I${HOME}/EAI/yolo_with_pycam
LDFLAGS := -pthread -ltensorflow-lite -lflatbuffers -l:libedgetpu.so.1.0 -ldl -lraspicam -lraspicam_cv -lopencv_dnn -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_imgcodecs

LDPATH := -L${HOME}/tensorflow/tensorflow/lite/tools/make/gen/bbb_armv7l/lib\
	  -L${HOME}/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build\
	  -L/usr/local/lib/arm-linux-gnueabihf

SRCS := main.cc
OBJS := $(SRCS:.cc=.o)
EXEC := main

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -g -o $@ $^ $(LDPATH) $(LDFLAGS)  `pkg-config --cflags --libs opencv4`

%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -c $< -g -o $@ `pkg-config --cflags --libs opencv4`  

clean : 
	rm -f $(OBJS) $(EXEC)
