#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>

#ifndef TRIG1
#define TRIG1 29 //gpio 21 pin 40
#endif
#ifndef ECHO1
#define ECHO1 28 //gpio 20 pin 38
#endif
#ifndef TRIG2
#define TRIG2 3
#endif
#ifndef ECHO2
#define ECHO2 2
#endif
float get_distance(bool);
bool detect();
