#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include "headers/gpio.h"

float get_distance()
{
	unsigned long TX_time = 0;
	unsigned long RX_time = 0;
	float distance = 0;
	unsigned long timeout = 50000000; // 0.5 sec ~ 171 m 50*10^6 us
	unsigned long Wait_time=micros();

	pinMode(TRIG, OUTPUT); //gpio 21 pin 40 using trigger
	pinMode(ECHO, INPUT); //gpio 20 pin 38 using Echo ultra sound

	// Ensure trigger is low.
	digitalWrite(TRIG, LOW);
	delay(50); //mili sec

	// Trigger tx sound.
	digitalWrite(TRIG, HIGH);
	delayMicroseconds(10);
	digitalWrite(TRIG, LOW);

	// Wait for tx response, or timeout.
	while ((digitalRead(ECHO) == LOW && (micros()-Wait_time) < timeout)) {
		if(digitalRead(ECHO) == HIGH) break;
	}

	// Cancel on timeout.
	if ((micros()-Wait_time) > timeout) {
		printf("0 Out of range.micros =%d wait-time=%d \n",micros(),Wait_time);
	return -1;
	}

	TX_time = micros(); //since call wiringPiSetup, the number of microseconds
	// Wait for rx sound response, or timeout.
	while ((digitalRead(ECHO) == HIGH && (micros()-Wait_time)) < timeout) {
		if(digitalRead(ECHO) == LOW) break;
	}

	// Cancel on timeout.
	if ((micros()-Wait_time) > timeout) {
		printf("1.Out of range.\n");
		return -1;
	}

	RX_time = micros();

	// Calculate rx-tx duration to change distance.
	distance = (float) (RX_time - TX_time) * 0.017; //( 340m/2) *100cm/10^6 us
	//printf("Range %.2f cm.\n", distance);
	return distance;
}
