#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>
#include "headers/warning.h"

void pin_setting() 
{
    pinMode(BUZZER, OUTPUT);
    digitalWrite(BUZZER, LOW);
}

void warning_on() 
{
    digitalWrite(BUZZER, HIGH);     
}

void warning_off()
{
    digitalWrite(BUZZER, LOW); 
}
