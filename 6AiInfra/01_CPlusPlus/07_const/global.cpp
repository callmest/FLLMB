#include "global.h"
#include <iostream>

const int bufSize2 = 1024;

void printBufSize(){
    std::cout << "The address of the bufSize in global.cpp: " << &bufSize << std::endl;
    std::cout << "The address of the bufSize2 in global.cpp: " << &bufSize2 << std::endl;
}