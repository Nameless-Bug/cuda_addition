

cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_srcs))
cpp_objs := $(subst src/,objs/,$(cpp_objs))

# 由于cpp可能与cu同名，但是不同文件
# 比如开心做的
# 因此，对于cuda的程序，把cu改成cuo
cu_srcs := $(shell find src -name "*.cu")
cu_objs := $(patsubst %.cu,%.cuo,$(cu_srcs))
cu_objs := $(subst src/,objs/,$(cu_objs))


# 定义名称参数
workspace := workspace
binary := pro
sbinary := sb.so

# 这里定义头文件库文件和链接目标没有加-I -L -l，后面用foreach一次性增加 
include_paths := /usr/local/cuda-10.1/include \
				 /datav/lean/opencv4.5.1/include/opencv4 \
				 src

# 这里需要清楚的认识链接的库到底链接是谁，这个非常重要
# 要求链接对象一定是预期的
library_paths := /usr/local/cuda-10.1/lib64 \
				 /datav/lean/opencv4.5.1/lib \
				 /datav/lean/tensorRT6.0.1.5_cuda10.1_cudnn7.6.0.3/lib

link_librarys := cudart cublas opencv_core opencv_imgcodecs opencv_imgproc gomp

# 定义编译选项,  -w屏蔽警告
cpp_compile_flags := -m64 -fPIC -g -O0 -std=c++11 -w -fopenmp
cu_compile_flags  := -m64 -Xcompiler -fPIC -g -O0 -std=c++11 -w -Xcompiler -fopenmp

# 2种类型
# 1. 字符串
# 2. 字符串数组
# 空格隔开就是数组
#
# 对头文件、库文件、目标统一增加-I -L -l
# foreach var,list,cmd
#     var  = item
#     list = link_librarys
#     cmd  = -Wl,-rpath=$(item)
#
# output = []
# for item in link_librarys:
#     output.append(f"-Wl,-rpath={item}")
# rpaths = output
#
# -L  指定链接目标时查找的目录
# -l  指定链接的目标名称，符合libname.so， -lname 规则
# -I  指定编译时头文件查找目录
rpaths        := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))


# 合并选项
cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        := $(rpaths) $(library_paths) $(link_librarys)



# 定义cpp的编译方式
# $@   生成项
# $<   依赖项第一个
# $^   依赖项所有
# $?  $+
objs/%.o : src/%.cpp
	@mkdir -p $(dir $@)
	@echo Compile $<
	@g++ -c $< -o $@ $(cpp_compile_flags)


# 定义cuda文件的编译方式
objs/%.cuo : src/%.cu
	@mkdir -p $(dir $@)
	@echo Compile $<
	@nvcc -c $< -o $@ $(cu_compile_flags)


# 定义workspace/pro文件的编译
$(workspace)/$(binary) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $@
	@g++ $^ -o $@ $(link_flags)

# 定义链接workspace/sb.so
$(workspace)/$(sbinary) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $@
	@g++ -shared $^ -o $@ $(link_flags)

# 定义pro快捷编译指令，这里只发生编译，不执行
pro : $(workspace)/$(binary)
#@strip $<

sb  : $(workspace)/$(sbinary)

# 定义编译并执行的指令，并且执行目录切换到workspace下
run : pro
	@cd $(workspace) && ./$(binary)

debug :
	@echo $(cpp_objs)
	@echo $(cu_objs)

clean :
	@rm -rf objs $(workspace)/$(binary)

# 指定伪标签，作为指令
.PHONY : clean debug run pro