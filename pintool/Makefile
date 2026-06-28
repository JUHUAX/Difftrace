# PIN 根目录
PIN_ROOT ?= /root/semvec/pintool/pin-3.28-98749-g6643ecee5-gcc-linux

# 包含 PIN 官方配置
ifdef PIN_ROOT
    CONFIG_ROOT := $(PIN_ROOT)/source/tools/Config
    TOOLS_ROOT := $(PIN_ROOT)/source/tools
else
    $(error PIN_ROOT is not specified)
endif

include $(CONFIG_ROOT)/makefile.config
include makefile.rules
include $(TOOLS_ROOT)/Config/makefile.default.rules

# 过滤掉不兼容的编译标志
CXXFLAGS := $(filter-out -Werror -Wno-dangling-pointer, $(CXXFLAGS))
TOOL_CXXFLAGS := $(filter-out -Werror -Wno-dangling-pointer, $(TOOL_CXXFLAGS))

# 添加 -no-pie 选项解决编译器版本高导致的重定位错误
CXXFLAGS += -no-pie
LDFLAGS += -no-pie

# ============== 自定义源文件和对象文件 ==============
SOURCES = config.cpp logger.cpp instrumentation.cpp nethook.cpp moduleinfo.cpp taintset.cpp
OBJECTS = $(SOURCES:%.cpp=$(OBJDIR)%$(OBJ_SUFFIX))

# ============== 自定义链接规则 ==============
# 覆盖默认的工具链接规则，包含所有对象文件
$(OBJDIR)pintool$(PINTOOL_SUFFIX): $(OBJDIR)pintool$(OBJ_SUFFIX) $(OBJECTS)
	$(LINKER) $(TOOL_LDFLAGS) $(LINK_EXE)$@ $^ $(TOOL_LPATHS) $(TOOL_LIBS)

all: $(OBJDIR)pintool$(PINTOOL_SUFFIX)

