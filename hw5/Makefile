CC := g++
CFLAGS := -std=c++11 -I/usr/include
INCLUDEPATHS := -I${HOME}/tensorflow
LDFLAGS := -pthread -ltensorflow-lite -lflatbuffers -ldl
LDPATH := -L${HOME}/tensorflow/tensorflow/lite/tools/make/gen/bbb_armv7l/lib\
	  -L${HOME}/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build

SRCS := main.cpp
OBJS := $(SRCS:.cpp=.o)
EXEC := main

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -o $@ $^ $(LDPATH) $(LDFLAGS) `pkg-config --cflags --libs opencv4`

%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -c $< -o $@ `pkg-config --cflags --libs opencv4`

clean : 
	rm -f $(OBJS) $(EXEC)
