#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef BUZZER
#define BUZZER 4 // gpio 23 pin 16
#endif

void pin_setting();
void warning_on();
void warning_off();