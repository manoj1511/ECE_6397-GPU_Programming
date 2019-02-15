/* blueSquare.c */ 

#include<stdio.h>

int main(void) {
int i;
// this is the header of the ppm file

 printf("P6\n");
 printf("#Hello\n#How are you\n");
 printf("20 20 255\n"); // width = 400, height = 400

// image data: loops over all the pixels and sets them all to bright yellow 
for (i = 0; i < 20 * 20; i++) {
printf("%c%c%c", 0, 0, 255); // r = 0, g = 0, b = 255 
}
return 0; }
