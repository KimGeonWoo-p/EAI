#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include "headers/gpio.h"

float get_distance(bool flag)
{
	int trig, echo;

	if (flag)
	{	
		trig = TRIG1;
		echo = ECHO1;
	} else
	{
		trig = TRIG2;
		echo = ECHO2;
	}

	unsigned long TX_time = 0;
	unsigned long RX_time = 0;
	float distance = 0;
	unsigned long timeout = 50000000; // 0.5 sec ~ 171 m 50*10^6 us
	unsigned long Wait_time=micros();

	pinMode(trig, OUTPUT); //gpio 21 pin 40 using trigger
	pinMode(echo, INPUT); //gpio 20 pin 38 using Echo ultra sound

	// Ensure trigger is low.
	digitalWrite(trig, LOW);
	delay(50); //mili sec

	// Trigger tx sound.
	digitalWrite(trig, HIGH);
	delayMicroseconds(10);
	digitalWrite(trig, LOW);

	// Wait for tx response, or timeout.
	while ((digitalRead(echo) == LOW && (micros()-Wait_time) < timeout)) {
		if(digitalRead(echo) == HIGH) break;
	}

	// Cancel on timeout.
	if ((micros()-Wait_time) > timeout) {
		printf("0 Out of range.micros =%d wait-time=%d \n",micros(),Wait_time);
	return -1;
	}

	TX_time = micros(); //since call wiringPiSetup, the number of microseconds
	// Wait for rx sound response, or timeout.
	while ((digitalRead(echo) == HIGH && (micros()-Wait_time)) < timeout) {
		if(digitalRead(echo) == LOW) break;
	}

	// Cancel on timeout.
	if ((micros()-Wait_time) > timeout) {
		printf("1.Out of range.\n");
		return -1;
	}

	RX_time = micros();

	// Calculate rx-tx duration to change distance.
	distance = (float) (RX_time - TX_time) * 0.017; //( 340m/2) *100cm/10^6 us
	//printf("%d Range %.2f cm.\n", flag, distance);
	return distance;
}

bool detect()
{
	float dist1, dist2;
	dist1 = dist2 = -1;

	//dist1 = get_distance(0);
	dist2 = get_distance(1);

	printf("dist1 : %f, dist2 : %f\n", dist1, dist2);

	if (dist1 >= 0 && dist1 < 20)
		return true;
	if (dist2 >= 0 && dist2 < 20)
		return true;

	return false;
}

