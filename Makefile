
cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_srcs))
cpp_objs := $(subst src/,objs/,$(cpp_objs))

cu_srcs := $(shell find src -name "*.cu")
cu_objs := $(patsubst %.cu,%.cuo,$(cu_srcs))
cu_objs := $(subst src/,objs/,$(cu_objs))

lean := /datav/lean
include_paths := src \
			$(lean)/opencv4.2.0/include/opencv4 \
			/usr/local/cuda/include 

library_paths := $(lean)/opencv4.2.0/lib \
			/usr/local/cuda/lib64

link_librarys := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs \
			cuda curand cublas cudart \
			stdc++  dl

run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

cpp_compile_flags := -std=c++11 -fPIC -m64 -g -fopenmp -w -O0
cu_compile_flags  := -std=c++11 -m64 -Xcompiler -fPIC -g -w -gencode=arch=compute_75,code=sm_75 -O0
link_flags        := -pthread -fopenmp

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags 		  += $(library_paths) $(link_librarys) $(run_paths)

pro : workspace/pro

# $@ 生成项
# $< 依赖项第一个
# $^ 依赖项所有
workspace/pro : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@g++ $^ -o $@ $(link_flags)

objs/%.o : src/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@g++ -c $< -o $@ $(cpp_compile_flags)

objs/%.cuo : src/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@nvcc -c $< -o $@ $(cu_compile_flags)

run : workspace/pro
	@cd workspace && ./pro

debug :
	@echo $(includes)

clean :
	@rm -rf objs workspace/pro