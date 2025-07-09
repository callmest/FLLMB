#include <iostream>

// 返回的临时字符串是一个右值
std::string foo() {
    return std::string("Hello, World!");
}

int main() {
    // 定义变量a
    int a = 100;
    // 定义引用b并且指向a, b就是a的别名
    int &b = a;

    //输出a和b的地址
    std::cout << "a的地址为：" << &a << std::endl;
    std::cout << "b的地址为: " << &b << std::endl;
    // 输出a和b的值
    std::cout << "a的值:" << a << std::endl;
    std::cout << "b的值: " << b << std::endl;
//
    // a和b是指向同一个变量。所以我们修改a的值，b的值也会变
    a = 200;
    std::cout << "修改a的值后，a和b的值分别为:\n" << a << std::endl;
    std::cout << b << std::endl;
//
    // 修改b的值，a的值也会变
    b = 300;
    std::cout << "修改b的值后，a和b的值分别为:\n" << a << std::endl;
    std::cout << b << std::endl;
//
    // 定义c，存储a的值
    int c = a;
    std::cout << "c的值:" << c << std::endl;
    // 修改c的值
    c = 400;
    std::cout << "修改c的值后，c为:" << c << std::endl;
    std::cout << "修改c的值后，a和b的值分别为:\n" << a << std::endl;
    std::cout << b << std::endl;
//
    // 尝试将引用重新绑定到另一个对象是非法的（编译时错误）
    int c2 = 300;
    // 表示修改b的值为c2的值
    b = c2;
    std::cout << "修改a的值后，a和b的值分别为:\n" << a << std::endl;
    std::cout << b << std::endl;
    //输出a和b的地址
    std::cout << "a的地址为：" << &a << std::endl;
    std::cout << "b的地址为: " << &b << std::endl;
    std::cout << "c2的地址为：" << &c2 << std::endl;

    //左值， 可以被修改的值
    //右值，不可以被修改的值，比如函数返回值，字面量等

    //绑定到左值的引用叫做左值引用
    int m = 10;
    int & m_r = m;
    //绑定到右值的引用叫做右值引用
    int && m_rr = 10;
    //不能用左值引用绑定右值
    //int & m_r1 = 10;

    // c是整数字面量20的右值引用（但这种情况不常见，通常用于函数参数或返回值）
    int&& cr = 20;

    // d是foo()返回的临时字符串的右值引用
    std::string&& d = foo();
    std::cout << "d的值:" << d << std::endl;
    return 0;
}
