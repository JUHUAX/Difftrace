#include "config.h"

namespace config {

// ============== 基础配置 ==============
const char* LOG_FILE = "/root/semvec/pintool_new/taint_record.log";
const bool ENABLE_LOGGING = true;
const bool FLUSH_IMMEDIATELY = true;

// ============== 污点追踪配置 ==============
const size_t MAX_TAINT_SIZE = 8192;
const bool USE_GLOBAL_TAINT_INDEX = true;
const bool TRACK_MEMORY_TAINT = true;
const bool TRACK_REGISTER_TAINT = true;

// ============== 插桩控制 ==============
const bool RECORD_FUNCTION = true;
const bool RECORD_BASICBLOCK = true;
const bool RECORD_INSTRUCTION = true;
const bool RECORD_ONLY_TAINTED_INSTRUCTIONS = true;
const bool RECORD_ONLY_TAINTED_BASICBLOCKS = true;
const bool RECORD_LOOPS = true;
const bool DEDUPLICATE_LOOPS = true;

// ============== Function Hook 配置 ==============
const char* RECV_FUNCTIONS[] = {
    "recv",
    "recvmsg",
    "recvfrom",
    "read",
    nullptr
};

// ============== 函数记录过滤配置 ==============
// 默认关闭白名单，便于适配不同协议程序。
// 若只想聚焦 opendnp3 语义函数，可改为 true。
const bool ENABLE_FUNCTION_WHITELIST = false;

const char* FUNCTION_WHITELIST_PREFIXES[] = {
    "opendnp3::",
    nullptr
};

// ============== 过滤配置 ==============
const char* BLACKLIST_FUNCTIONS[] = {
    "malloc",
    "free",
    "calloc",
    "realloc",
    nullptr
};

// 系统库黑名单（不插桩这些模块）
const char* MODULE_BLACKLIST[] = {
    "ld-linux-x86-64.so",
    "libc.so",
    "libc-",
    "libm.so",
    "libm-",
    "libpthread.so",
    "libpthread-",
    "libdl.so",
    "libdl-",
    "librt.so",
    "librt-",
    "libresolv.so",
    "libresolv-",
    "libstdc++.so",
    "libgcc_s.so",
    "libgcc_s-",
    "ld-",
    "linux-vdso.so",
    nullptr
};

const bool ENABLE_MODULE_FILTERING = true;

// ============== 调试开关 ==============
const bool DEBUG_MODE = false;
const bool VERBOSE_MODE = false;

}  // namespace config
