#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>

#ifndef TRIG
#define TRIG 29 //gpio 21 pin 40
#endif
#ifndef ECHO
#define ECHO 28 //gpio 20 pin 38
#endif

float get_distance();
