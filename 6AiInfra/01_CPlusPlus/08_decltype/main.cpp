#include <iostream>

//basic usage
typedef double wages;
//p is a double*
typedef wages base, *p;

//c++11
using int64_t = long long;

// const pointer
typedef char* pstring;

int main(){

    wages wage = 10;
    base wage2 = wage;
    p wage3 = &wage;

    int64_t money = 1000000;

    //const pointer -> char* const str = "pikaqiu"
    const pstring str = nullptr;

    const pstring* str2 = nullptr;
    // equals to
    char* const* str3 = nullptr;

    //auto
    int age1 = 10;
    int age2 = 20;
    auto age3 = age1 + age2;
    return 0;

    auto i = 0, *p = &i;

    //decltype to the up const will be ignored
    int i = 0;
    // ci actually is another var, has its own address
    const int ci = i, &cr = ci;
    // b: int
    auto b = ci;
    // d: int*
    auto d = &i;
    //e: const int*
    auto e = &ci;
    //f: const int
    const auto f = ci;
    //g: const int&
    auto &g = ci;
    //k: int, l: int&
    auto k = ci, &l = i;
    //m: const int&, ptr3: const int*
    auto &m = ci, *p = &ci;

    //decltype
    {
        const int ci = 0, &cj = ci;
        //x: const int
        decltype(ci) x = 0;
        //y: const int &
        decltype(ci) y = x;

        int i = 42, *p = &i, &r = i;
        //b: int
        decltype(r+0) b;
        //*p:int&, c: int&
        decltype(*p) c = i;
        //e: int&
        decltype(r) e = i;
        //f: int&, (i) is a expression
        decltype((i)) f = i;
    }
}