#include <iostream>

int main() {
    // declare a int pointer
    int * ptr;

    // declare two int pointers at the same time
    int * ip1, * ip2;

    // double var and pointer
    double db, * db2;

    // define a var and define a pointer point to it
    int var = 10;
    int * ptr2 = &var;
    // ptr2 is a pointer and define a pointer of a pointer to point to it
    int * * ptr2_address = &ptr2;

    std::cout << "ptr value: " << ptr2 << std::endl;
    std::cout << "ptr address: " << &ptr2 << std::endl;
    std::cout << "ptr2_address value: " << ptr2_address << std::endl;
    std::cout << "ptr2_address address: " << &ptr2_address << std::endl;
    // ptr value: 0x7ffe6f30757c
    // ptr address: 0x7ffe6f307580
    // ptr2_address value: 0x7ffe6f307580
    // ptr2_address address: 0x7ffe6f307588


    // define a double var
    double dvar = 3.14;
    double *dptr = &dvar;
    // dptr2 and dptr both point to dvar, but they are two diff vars
    double *dptr2 = dptr;
    std::cout << "dptr value: " << dptr << std::endl;
    std::cout << "dptr address: " << &dptr << std::endl;
    std::cout << "dptr2 value: " << dptr2 << std::endl;
    std::cout << "dptr2 address: " << &dptr2 << std::endl;
    // dptr value: 0x7ffe6f307590
    // dptr address: 0x7ffe6f307598
    // dptr2 value: 0x7ffe6f307590
    // dptr2 address: 0x7ffe6f3075a0

    // get the value of a pointer points to
    int ivar = 42;
    int* iptr = &ivar;
    std::cout << "The value of iptr points to: " << *iptr << std::endl;
    // modify the value
    *iptr = 43;
    std::cout << "Mofified the value of ivar: " << ivar << std::endl;
    std::cout << "Mofified the value of iptr points to: " << *iptr << std::endl;
    // The value of iptr points to: 42
    // Mofified the value of ivar: 43
    // Mofified the value of iptr points to: 43

    // & and *
    int ivar2 = 52;
    // ref
    int &r = ivar2;
    // pointer
    int* p;
    // get address
    p = &ivar2;
    // deref
    *p = 43;
    // union
    int &r2 = *p;
    std::cout << "ivar2 value: " << ivar2 << std::endl;
    std::cout << "r value: " << r << std::endl;
    std::cout << "r2 value: " << r2 << std::endl;
    std::cout << "ivar2 address: " << p << std::endl;
    std::cout << "r address: " << &r << std::endl;
    std::cout << "r2 address: " << &r2 << std::endl;
    // ivar2 value: 43
    // r value: 43
    // r2 value: 43
    // ivar2 address: 0x7fff79bfa58c
    // r address: 0x7fff79bfa58c
    // r2 address: 0x7fff79bfa58c

    // define an empty pointer, recomm
    int* nptr = nullptr;

    // you can give a pointer to another
    int ivar3 = 66;
    int* iptr3 = &ivar3;
    int* iptr4 = nullptr;
    iptr4 = iptr3;
    std::cout << "ivar3 address: " << &ivar3 << std::endl;
    std::cout << "iptr3 value: " << iptr3 << std::endl;
    std::cout << "iptr4 value: " << iptr4 << std::endl;
    // ivar3 address: 0x7ffeda4cfd14
    // iptr3 value: 0x7ffeda4cfd14
    // iptr4 value: 0x7ffeda4cfd14

    // how to identify a nullptr: 0 means false, and 1 means true
    int* emptrptr = nullptr;
    if (! emptrptr)
    {
        std::cout << "emptyptr is null" << std::endl;
    }

    // the size of diff pointer, depends on platform, 4 bytes for 32, 8 bytes for 64
    std::cout << "size of int pointer: " << sizeof(iptr3) << std::endl;
    std::cout << "size of double pointer: " << sizeof(dptr) << std::endl;
    // size of int pointer: 8
    // size of double pointer: 8

    // void pointer
    double obj = 3.14, * pobj = &obj;
    void* pv;
    pv = pobj;
    std::cout << "The value of pv: " << (*(double*)pv) << std::endl;

    // the ref of a pointer
    int init = 100;
    int* pinit = nullptr;
    int* &rpinit = pinit;
    pinit = &init;
    std::cout << "init address: " << &init << std::endl;
    std::cout << "pinit value: " << pinit << std::endl;
    std::cout << "rpinit value: " << rpinit << std::endl;
    *rpinit = 101;
    std::cout << "init value: " << init << std::endl;
    std::cout << "pinit pointed value: " << *pinit << std::endl;
    std::cout << "rpinit pointed value: " << *rpinit << std::endl;
    // init address: 0x7fffe44b1f84
    // pinit value: 0x7fffe44b1f84
    // rpinit value: 0x7fffe44b1f84
    // init value: 101
    // pinit pointed value: 101
    // rpinit pointed value: 101

    // pointer for array
    int arr[5] = {1, 2, 3, 4, 5};
    // define a pointer to arr, arr is first element address, therefore no "&"
    int* aptr = arr;
    int first_element = *aptr;
    std::cout << "The address of the aptr: " << aptr << std::endl;
    std::cout << "The first element of the arr: " << first_element << std::endl;
    ++aptr;
    std::cout << "The new address of the aptr: " << aptr << std::endl;
    std::cout << "The second element of the arr: " << *aptr << std::endl;
    // The address of the aptr: 0x7ffe8c5dfa90
    // The first element of the arr: 1
    // The new address of the aptr: 0x7ffe8c5dfa94
    // The second element of the arr: 2
    // or, aptr + 1 because it is an int type
    aptr = aptr + 1;
    std::cout << "The new address of the aptr: " << aptr << std::endl;
    std::cout << "The third element of the arr: " << *aptr << std::endl;
    // The new address of the aptr: 0x7ffc9233e6f8
    // he third element of the arr: 3
    return 0;
}