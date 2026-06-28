#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace config {

// ============== 基础配置 ==============
extern const char* LOG_FILE;
extern const bool ENABLE_LOGGING;
extern const bool FLUSH_IMMEDIATELY;

// ============== 污点追踪配置 ==============
// 最大污点字节数
extern const size_t MAX_TAINT_SIZE;

// 污点索引配置（recv 调用编号）
extern const bool USE_GLOBAL_TAINT_INDEX;

// 是否追踪内存污点
extern const bool TRACK_MEMORY_TAINT;

// 是否追踪寄存器污点
extern const bool TRACK_REGISTER_TAINT;

// ============== 插桩控制 ==============
// 是否记录 Function（函数入口/出口）
extern const bool RECORD_FUNCTION;

// 是否记录 BasicBlock
extern const bool RECORD_BASICBLOCK;

// 是否记录 Instruction
extern const bool RECORD_INSTRUCTION;

// 是否只在指令有污点时记录
extern const bool RECORD_ONLY_TAINTED_INSTRUCTIONS;

// 是否只输出被污染的基本块（BBL内有被记录的指令才输出）
extern const bool RECORD_ONLY_TAINTED_BASICBLOCKS;

// 是否记录循环（LOOP）
extern const bool RECORD_LOOPS;

// 记录循环时去重
extern const bool DEDUPLICATE_LOOPS;

// ============== Function Hook 配置 ==============
// 要 hook 的网络接收函数优先级列表
extern const char* RECV_FUNCTIONS[];

// ============== 函数记录过滤配置 ==============
// 是否启用函数白名单（仅记录匹配白名单前缀的函数）
extern const bool ENABLE_FUNCTION_WHITELIST;

// 函数白名单前缀列表（如 "opendnp3::"）
extern const char* FUNCTION_WHITELIST_PREFIXES[];

// ============== 过滤配置 ==============
// 黑名单函数名（不进行污点追踪）
extern const char* BLACKLIST_FUNCTIONS[];

// 黑名单模块（系统库，不进行插桩）
extern const char* MODULE_BLACKLIST[];

// 是否启用模块过滤
extern const bool ENABLE_MODULE_FILTERING;

// ============== 调试开关 ==============
extern const bool DEBUG_MODE;
extern const bool VERBOSE_MODE;

}  // namespace config

#endif  // CONFIG_H
