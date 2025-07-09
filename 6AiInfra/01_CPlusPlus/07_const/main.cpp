#include <iostream>
#include "global.h"

int main()
{
    // basic usage
    const int a = 10;

    // define from a var
    int i1 = 10;
    const int i2 = i1;
    int i3 = i2;
    std::cout << "The address of the i1: " << &i1 << std::endl;
    std::cout << "The address of the i2: " << &i2 << std::endl;
    std::cout << "The address of the i3: " << &i3 << std::endl;
    // The address of the i1: 0x7ffd098ed388
    // The address of the i2: 0x7ffd098ed38c, cannot modified
    // The address of the i3: 0x7ffe98e36cb0

    // the printBufSize function is defined in global.cpp
    // the bufSize is defined in the global.h, so the compiler will generate diff vars for diff cpps
    // the bufSize2 is externed in the global.h, but defined in the global.cpp, it means there is only one var
    printBufSize();
    // print main cpp this file, bufsize
    std::cout << "The address of the bufSize in main.cpp: " << &bufSize << std::endl;
    std::cout << "The address of the bufSize2 in main.cpp: " << &bufSize2 << std::endl;
    // The address of the bufSize in global.cpp: 0x60c5ac65b0b8
    // The address of the bufSize2 in global.cpp: 0x60c5ac65b0bc
    // The address of the bufSize in main.cpp: 0x60c5ac65b008
    // The address of the bufSize2 in main.cpp: 0x60c5ac65b0bc

    // const reference
    const int i = 10;
    const int &r = i;
    int b = 10;
    const int& r2 = b;
    // you can modified b using b
    b = 20;
    // you can not modified b with r2
    // [x] r2 = 20;
    // bind the express value
    const int& r3 = r2 * 2;
    // bind diff types, double -> int
    double dval = 3.14;
    const int& r4 = dval;
    std::cout << "The r4 value: " << r4 << std::endl;
    // The r4 value: 3
    // multiple ref
    int c = 1024;
    int& r5 = c;
    const int& r6 = c;
    r5 = 2048;
    std::cout << "The c value: " << c << std::endl;
    std::cout << "The r5 value: " << r5 << std::endl;
    std::cout << "The r6 value: " << r6 << std::endl;
    // The c value: 2048
    // The r5 value: 2048
    // The r6 value: 2048

    //pointer to const
    const double pi = 3.14;
    const double pi2 = 2.44;
    const double* ptr = &pi;
    // pointed value can not be modified, but the pointed var can be changed
    ptr = &pi2;
    // [x] *ptr = 20
    // [x] int* ptr2 = ptr

    //const pointer
    int errNum = 0;
    int* const p = &errNum;
    // pointed value can be modified, but the pointed var can not be changed
    *p = 10;
    // int errNum2 = 20
    // [x] p = &errNum2

    //constexpr
    constexpr int sz = getSizeConst();
    std::cout << "sz value: " << sz << std::endl;
    int errNum2 = 20;
    // [x] constexpr int* p1 = &errNum2;
    constexpr int* p1 = nullptr; // equals to int* const p1 = nullptr;
}