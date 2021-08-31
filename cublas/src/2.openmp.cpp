

// #include <stdio.h>
// #include <thread>
// #include <iostream>
// #include <omp.h>

// using namespace std;

// int main(){

//     volatile int count = 0;
//     // 虚拟线程/物理线程，超线程技术
//     cout << "omp_get_max_threads: " << omp_get_max_threads() << endl;

//     // 物理处理单位，物理线程
//     cout << "omp_get_num_procs: " << omp_get_num_procs() << endl;

//     #pragma omp parallel for //num_threads(2)
//     for(int i =0 ; i < 1000; ++i){
    
//         printf("%d, omp_get_thread_num = %d, omp_get_num_threads = %d\n", i, omp_get_thread_num(), omp_get_num_threads());
//         this_thread::sleep_for(chrono::milliseconds(1000));

//         // 锁
//         // lock
//         #pragma omp critical
//         {
//             // 这里执行的代码，同一时刻只有一个线程进入
//             count += i;
//         }
//         // unlock
//     }   
//     printf("count = %d\n", count);
//     ///
//     return 0;
// }