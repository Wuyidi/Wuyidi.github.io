---
layout:     post
title:      "How to build a Cmake-Based project"
subtitle:   "在linux下使用Cmake构建应用程序"
date:       2019-03-07
author:     "Yidi"
header-img: "img/post-bg-2015.jpg"
tags:
    - cmake
---

### 介绍

CMake 允许开发者编写一种平台无关的 CMakeList.txt 文件来定制整个编译流程，然后再根据目标用户的平台进一步生成所需的本地化 Makefile 文件。

在 Linux 平台下使用 CMake 生成 Makefile 并编译的流程如下：

1. 编写CMakeList.txt文件
2. 执行 `cmake [options] <path-to-source>  ` 生成 Makefile （通过CMAKE_BUILD_TYPE选择生成release或者debug版的Makefile）
3. 使用make命令进行编译

本文的部分内容翻译自[cmake tutorial](https://cmake.org/cmake-tutorial/) 结合AMI项目进行的总结

### 步骤（一）

对于大部分项目来说，一个最基本的 CMakeList.txt 必须包含以下三个要素。

1. 规定了Cmake的最低使用版本
2. 定义项目的名称
3. 通过 `add_executable()` 生成可执行文件所需要的依赖关系

```cmake
# CMake minimum version >= 3.9.0
cmake_minimum_required(VERSION 3.9)

# Project info
project (ami_gateway)
project (manager)
project (security_adapter)
project (server)
# add the executable
```

### 步骤（二）

1. 使用 `add_compile_options` 命令添加 `--std=c++11` 选项，是针对所有编译器（包括c&c++），而set命令设置 `CMAKE_C_FLAGS` 或 `CMAKE_CXX_FLAGS` 变量则是分别只针对c和c++编译器。

   ```cmake
   # 判断编译器类型,如果是gcc编译器,则在编译选项中加入c++11支持
   if(CMAKE_COMPILER_IS_GNUCXX)
       set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
       message(STATUS "optional:-std=c++11")   
   endif(CMAKE_COMPILER_IS_GNUCXX)
   ```

2. 设置root目录
3. 设置可执行文件输出目录

```cmake
# Adds options to the compilation of source files
add_compile_options("--std=c++11")
# Set the root directory
set(HFT_ROOT ${PROJECT_SOURCE_DIR})
set(AMI_ROOT ${PROJECT_SOURCE_DIR}/../install_package)
# Set runtime output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${HFT_ROOT}/build/debug/bin)
```

### 步骤（三）

添加检索目录：

1.  `include_directory()` 添加头文件目录
2. `add_subdirectory()` 添加需要编译的子目录

`${PROJECT_SOURCE_DIR}` 和 `${PROJECT_BINARY_DIR}` 是CMake预先定义的两个变量，代表顶层 CMakelists.txt 所在目录，后者是执行cmake的目录，就是build目录

```cmake
# Add the binary tree directory to the search path for include files
# to the source code
include_directories(${HFT_ROOT}/src
					${HFT_ROOT}/include
					${HFT_ROOT}/message
					${AMI_ROOT}/include
					${AMI_ROOT}/include/3rd
					${AMI_ROOT}/include/3rd/protobuf/
					${AMI_ROOT}/include/abf/)
					
add_subdirectory(${HFT_ROOT}/message)
```

### 步骤（四）

1. 通过 `add_executable()` 生成可执行文件所需要的依赖关系

```cmake
add_executable(ami_gateway ${HFT_ROOT}/src/ami_gateway.cpp)
add_executable(manager ${HFT_ROOT}/src/manager.cpp)
add_executable(security_adapter ${HFT_ROOT}/src/security_adapter.cpp)
add_executable(server ${HFT_ROOT}/src/server.cpp)
```

### 步骤（五）

1. 设置连接连接库文件的路径
2. 设置需要连接的库文件

```cmake
set(LD_PATH ${AMI_ROOT}/lib64/)
set(PROJECT_LIBS ${LD_PATH}/libabf.so
	${LD_PATH}/libami.so
	${LD_PATH}/libboost_date_time.so.1.62.0
	${LD_PATH}/libboost_locale.so.1.62.0
	${LD_PATH}/libboost_log.so.1.62.0
	${LD_PATH}/libboost_log_setup.so.1.62.0
	${LD_PATH}/libboost_program_options.so.1.62.0
	${LD_PATH}/libboost_regex.so.1.62.0
	${LD_PATH}/libboost_thread.so.1.62.0
	${LD_PATH}/libprotobuf.so.12)
	
target_link_libraries(ami_gateway ${PROJECT_LIBS} biz_message)
target_link_libraries(manager ${PROJECT_LIBS} biz_message)
target_link_libraries(security_adapter ${PROJECT_LIBS} biz_message)
target_link_libraries(server ${PROJECT_LIBS} biz_message)
```



### 步骤（六）

1. 选择 Debug 或者 Release 模式

   通过命令行的方式: `cmake -DCMAKE_BUILD_TYPE` 或者 `set(CMAKE_BUILD_TYPE DEBUG)`

