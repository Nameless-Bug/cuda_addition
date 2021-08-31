
// #include <opencv2/opencv.hpp>
// #include <stdio.h>
// #include <vector>
// #include <memory>

// using namespace cv;
// using namespace std;

// class A{
// public:
//     A(int a, int b){
//         printf("1, 创建新实例 %d %d,   %p\n", a, b, this);
//         this->data = new int[2];
//         this->data[0] = a;
//         this->data[1] = b;
//     }
//     A(const A& other){
//         printf("2. copy构造.  this = %p, other = %p\n", this, &other);
//         // copy构造，咱们定义为深copy
//         this->data = new int[2];
//         memcpy(this->data, other.data, 2 * sizeof(int));
//     }
//     A(A&& other){
//         printf("3. 移动构造.  this = %p, other = %p\n", this, &other);
//         // 移动构造，我确定要把other的东西移动到当前实例中
//         this->data = other.data;
//         other.data = nullptr;
//     }
//     virtual ~A(){
//         printf("4. 析构 %p, %p\n", this, this->data);
//         if(this->data){
//             delete[] this->data;
//             this->data = nullptr;
//         }
//     }
// private:
//     int* data = nullptr;
// };


// ///  Variadic template
// //重载的递归终止函数
// void printX() {
//     printf("End.\n");
// }

// template<typename T, typename...Types>
// void printX(const T& firstArg, const Types&...args) {
// 	cout << firstArg << endl;
// 	printX(args...);
// }
 
// int main(){
//     printX(7.5, "hello", 1.2f, 42);
// 	return 0;

//     //vector<int> a{1, 2, 3};
//     //vector<int> b(std::move(a));
//     //printf("a.size = %d, b.size = %d\n", a.size(), b.size());

//     //static_cast<A&&>(a)   c++的类型转换
//     // (A&&)a               c语言的类型转换
//     // a = b;
//     //A a(1, 2);
//     //A b(std::move(a));
//     //A b((A&&)a);

//     // 第一代opencv出现时，是C语言接口
//     // 刚开始的图像是IplImage，矩阵是CvMat
//     // cols, rows, BCHW channel()   batch
//     // Mat ; 以二维角度审视opencv
//     //  如果是表达三维的图像，那么他是一个二维的矩阵（无非是元素是一个Scalar，是1个值或者多个值（小于等于4）)
//     //Mat;

//     // // 类型 CV_8UC1
//     // #define CV_8U   0   unsigned char
//     // #define CV_8S   1   signed char
//     // #define CV_16U  2   unsigned short
//     // #define CV_16S  3   signed short
//     // #define CV_32S  4   signed int
//     // #define CV_32F  5   float
//     // #define CV_64F  6   double
//     // #define CV_16F  7   half

//     // 通道的概念
//     // CV_8UC3   8U类型，3通道
//     // CV_8UC(n) 8U类型，n通道
//     // CV_8U = CV_8UC1
//     // CV_8UC3  ->   mat.at<T>(row_index, col_index);    
//     //     sizeof(T) == 1byte(8bit) * 3(channel)
//     //     Vec3b     =  b  char
//     // Vec3b;  =  unsigned char   3 channel
//     // Vec3b = Vec<uchar, 3>
//     // Vec<short, 3>
//     // Vec<uchar, 3>* ptr = mat.ptr<Vec<uchar, 3>>(0);
//     //
//     //  
//     //

//     Mat matrix(2, 2, CV_16UC3);

//     // 给类型取别名
//     typedef Vec<unsigned short, 3> Kx3SU;
    
//     matrix.at<Kx3SU>(0, 0) = Kx3SU(1, 2, 3);
//     matrix.at<Kx3SU>(1, 1) = Kx3SU(3, 2, 1);

//     Kx3SU* ptr = matrix.ptr<Kx3SU>(0);
//     for(int r = 0; r < matrix.rows; ++r){
//         for(int c = 0; c < matrix.cols; ++c){
//             //auto& pixel = matrix.at<Kx3SU>(r, c);
//             auto& pixel = *ptr++;
//             printf("pixel[%d, %d] = %d, %d, %d\n", r, c, pixel[0], pixel[1], pixel[2]);
//         }
//     }

//     float data[] = {
//         1, 2, 3,
//         4, 5, 6,
//         7, 8, 9
//     };
//     Mat m(3, 3, CV_32F, data);
//     auto o = m.clone();

//     auto c = o * 2 - 3;
//     // 懒计算，延迟计算
//     // 懒加载

//     data[4] = 512;
//     printf("%f\n", o.at<float>(1, 1));

//     o.at<float>(1, 1) = 333;
//     printf("%f\n", data[4]);

//     // 如果捕获列表中指定捕获某个变量，或者&捕获所有变量
//     // 如果变量过了作用域被析构。然后再调用lambda函数，会得到随机值（如果是对象，则很可能崩溃）
//     // 匿名函数 -》C++
//     // 匿名函数 -> Python
//     // 闭包 -> Python
//     // auto func = [o](){
//     //     auto value = o.at<Vec3b>(0);
//     //     return 123;
//     // };


//     return 0;
// }