#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//this c program generates 50 data entries for the csv, that have a 1:1 pass fail ratio in order to be
//easily seperable by the perceptron model
int main() {
    //seed the random number generator
    srand(time(NULL));

    int countpassed = 0;
    int countfailed = 0;
	int passed = 1;
	int failed = 1;
    
    while(1){
    if(!countfailed && !countpassed) break;
       int num1 = rand() % 101;
       int num2 = rand() % 101;
       int num3 = rand() % 101;
	if(num1+num2+num3 >= 180 && passed){
		printf("%d,%d,%d,pass\n", num1, num2, num3);
		countpassed++;
		if(countpassed == 25) passed = 0;
}
	else if(failed){
		printf("%d,%d,%d,fail\n", num1, num2, num3);
		countfailed++;
		if(countfailed==25)failed = 0;
	}
}
    return 0;
}
