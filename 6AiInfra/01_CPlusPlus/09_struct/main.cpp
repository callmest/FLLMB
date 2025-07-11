#include <iostream>
#include <string>

struct Student
{
    // member variable
    int id;

    std::string name;

    float grade;

    // constructor
    Student(int studentID, std::string studentName, float studentGrade):
        id(studentID), name(studentName), grade(studentGrade) {}

    //default constructor
    Student():id(0), name(""), grade(0.0f){}
};

struct Point
{
    // default accession: public
    int x;
    int y;
};

// struct as function params
// 1. pass by value
void printPoint(Point p){
    std::cout << "Point: ( "<< p.x << "," << p.y << ")" << std::endl;
};

// 2. pass by ref, change param p directly
void movePoint(Point& p, int x,  int y){
    p.x += x;
    p.y += y;
};

// 3. pass by pointer, change param p directly
void movePoint_Pointer(Point* p, int x, int y){
    p->x += x;
    p->y += y;
};

// 4. return struct, p will not be changed, the new_p is another Point Var.
Point movePoint_return(Point* p, int x, int y){
    Point new_p = *p;
    new_p.x += x;
    new_p.y += y;
    return new_p;
};

// 5. return struct, change param p directly
Point movePoint_inplace(Point* p, int x, int y){
    (*p).x += x;
    (*p).y += y;
    return *p;
};

// 5. 
class Rectangle {
    //can set accession
    private:
        int width;
        int height;
    
    public:
        void set(int w, int h){
            width = w;
            height = h;
        }
    
        int area() const {
            return width * height;
        }
};

// struct and struct
struct Address
{
    std::string city;
    std::string street;
    int houseNumber;

    void printInfo() const {
        std::cout << "city: " << city 
                  << ", street: " << street 
                  << ", houseNumber: " << houseNumber << std::endl;        
    }
};

struct Person
{
    std::string name;
    int age;
    Address address;
};

int main(){
    // initialize Student
    Student zhangsan(1004, "zhangsan", 80.05f);

    // struct and class
    Rectangle rt;
    // [x] rt.width
    rt.set(5,6);
    rt.area();

    // struct and struct
    Person person;
    person.name = "Eve";
    person.age = 30;
    person.address.city = "New York";
    person.address.street = "5th Avenue";
    person.address.houseNumber = 101;

    std::cout << person.name << " lives at " 
              << person.address.houseNumber << " " 
              << person.address.street << ", " 
              << person.address.city << std::endl;

    // struct arr
    Student st_arr[] = {
        {10, "Bob", 80.0f},
        {10, "Alice", 77.0f}
    };
    for (int i =0; i < 2; i++){
        std::cout << "ID: "  << st_arr[i].id << std::endl;
        std::cout << "Name: "  << st_arr[i].name << std::endl;
        std::cout << "Grade: "  << st_arr[i].grade << std::endl;
    }

    // pointer to struct
    Person* pperson = &person;
    std::cout << "person name: "  << pperson->name << std::endl;
    std::cout << "person city: "  << pperson->address.city << std::endl;

    // struct function
    person.address.printInfo();

    // struct as function params
    // 1. pass by value
    Point p = {20, 10};
    // 20, 10
    printPoint(p);
    
    movePoint(p, 10, -10);
    // 30, 0
    printPoint(p);

    movePoint_Pointer(&p, 10, -10);
    // 40, -10
    printPoint(p);

    auto p1 = movePoint_return(&p, 10, -10);
    //40, -10
    printPoint(p);

    movePoint_inplace(&p, 10, -10);
    //50, -20
    printPoint(p);

    // Point: ( 20,10)
    // Point: ( 30,0)
    // Point: ( 40,-10)
    // Point: ( 40,-10)
    // Point: ( 50,-20)
    
    return 0;
}