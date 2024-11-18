#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>

#define PA 2
#define PB 4
#define PC 1
#define PD 16
#define PE 15
#define PF 8
#define PG 9
#define PDP 0

char nums[10] = {0xc0, 0xf9, 0xa4, 0xb0, 0x99, 0x92, 0x82, 0xf8, 0x80, 0x90};

char pins[8] = {PA, PB, PC, PD, PE, PF, PG, PDP};

void clear_pin()
{
	int i;
	for (i = 0; i < 8; i++)
		digitalWrite(pins[i], 1);
}

void set_pin(int n)
{
	int i;
	for (i = 0; i < 8; i++)
		digitalWrite(pins[i], (nums[n] >> i)&0x1);
}

void init_pin()
{
	int i;
	for (i = 0; i < 8; i++)
		pinMode(pins[i], OUTPUT);
}

int main(int argc, char** argvs)
{
	if (!argc)
		return -1;

	int i = (int) argvs[i][0];
	wiringPiSetup();
	init_pin();

	clear_pin();
	set_pin(i);

	return 0;
}
