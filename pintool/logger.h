#ifndef LOGGER_H
#define LOGGER_H

#include "address.h"
#include "taintset.h"
#include "config.h"
#include "pin.H"
#include <cstdio>
#include <cstdarg>
#include <string>
#include <map>

class Logger {
private:
    FILE* file;
    PIN_LOCK fileLock;
    bool initialized;

    Logger();
    ~Logger();

    // 禁止复制
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

public:
    // 单例模式获取实例
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    // 初始化日志文件
    bool init(const char* filepath);

    // 关闭日志文件
    void close();

    // ============== 记录函数 ==============
    // 记录函数入口
    void logFunctionEnter(THREADID tid, const std::string& funcName,
                         const Address& addr);

    // 记录函数出口
    void logFunctionExit(THREADID tid, const std::string& funcName);

    // ============== 记录基本块 ==============
    // 记录基本块
    void logBasicBlock(THREADID tid, const Address& addr, uint32_t size);

    // ============== 记录指令 ==============
    // 分支状态枚举：0=无, 1=TAKEN, 2=NOT_TAKEN
    enum BranchState { BRANCH_NONE = 0, BRANCH_TAKEN = 1, BRANCH_NOT_TAKEN = 2 };

    // 循环状态枚举：0=无, 1=LOOP, 2=REPEAT
    enum LoopType { LOOP_NONE = 0, LOOP_TYPE = 1, REPEAT_TYPE = 2 };
    
    // 记录指令执行（完整版本，支持分离污点）
    void logInstruction(THREADID tid, const Address& addr, const std::string& disasm,
                       const TaintSet& taint,
                       const std::map<std::string, uint64_t>& values,
                       BranchState branchState = BRANCH_NONE,
                       LoopType loopType = LOOP_NONE,
                       const std::string& formattedValues = "",
                       const TaintSet& dstTaint = TaintSet(),
                       const TaintSet& srcTaint = TaintSet());

    // 记录指令（简化版本，仅地址、反汇编、污点）
    void logInstructionSimple(THREADID tid, const Address& addr, const std::string& disasm,
                             const TaintSet& taint);

    // ============== 记录污点注入 ==============
    // 记录污点注入点
    void logTaintInject(THREADID tid, const Address& addr, size_t byteCount);

    // ============== 记录循环 ==============
    // 记录循环执行
    void logLoop(THREADID tid, const Address& addr, uint32_t size);

    // ============== 记录系统函数 ==============
    // 记录系统函数调用（如 memcpy）
    void logSystemFunction(THREADID tid, const std::string& funcName,
                          const std::vector<std::string>& taintedArgs);

    // ============== 内部辅助函数 ==============
private:
    // 线程安全的写入操作
    void writeLine(const std::string& line);

    // 格式化输出（变长参数）
    void writeFormatted(const char* fmt, va_list args);
};

#endif  // LOGGER_H
