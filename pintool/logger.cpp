#include "logger.h"
#include <cstring>
#include <ctime>

static std::string makeThreadPrefix(THREADID tid) {
    return "THREADID\t" + std::to_string(static_cast<unsigned long>(tid)) + "\t";
}

Logger::Logger() : file(nullptr), initialized(false) {
    PIN_InitLock(&fileLock);
}

Logger::~Logger() {
    close();
    // PIN_LOCK doesn't require explicit cleanup
}

bool Logger::init(const char* filepath) {
    PIN_GetLock(&fileLock, PIN_ThreadId());

    if (initialized) {
        PIN_ReleaseLock(&fileLock);
        return true;
    }

    file = fopen(filepath, "w");
    if (!file) {
        fprintf(stderr, "[ERROR] Failed to open log file: %s\n", filepath);
        PIN_ReleaseLock(&fileLock);
        return false;
    }

    initialized = true;
    fprintf(stderr, "[INFO] Logger initialized: %s\n", filepath);

    PIN_ReleaseLock(&fileLock);
    return true;
}

void Logger::close() {
    PIN_GetLock(&fileLock, PIN_ThreadId());

    if (file) {
        fflush(file);
        fclose(file);
        file = nullptr;
    }

    initialized = false;

    PIN_ReleaseLock(&fileLock);
}

void Logger::logFunctionEnter(THREADID tid, const std::string& funcName,
                             const Address& addr) {
    std::string line = makeThreadPrefix(tid) + "Function\tenter\t" + funcName + "\t" + addr.toString();
    writeLine(line);
}

void Logger::logFunctionExit(THREADID tid, const std::string& funcName) {
    std::string line = makeThreadPrefix(tid) + "Function\texit\t" + funcName;
    writeLine(line);
}

void Logger::logBasicBlock(THREADID tid, const Address& addr, uint32_t size) {
    char buf[256];
    snprintf(buf, sizeof(buf), "THREADID\t%lu\tBasicBlock\t%s\tsize=%u",
             static_cast<unsigned long>(tid),
             addr.toString().c_str(), size);
    writeLine(buf);
}

void Logger::logInstruction(THREADID tid, const Address& addr, const std::string& disasm,
                            const TaintSet& taint,
                            const std::map<std::string, uint64_t>& values,
                            BranchState branchState, LoopType loopType,
                            const std::string& formattedValues,
                            const TaintSet& dstTaint, const TaintSet& srcTaint) {
    std::string line = makeThreadPrefix(tid) + "Instruction\t" + addr.toString() + ": " + disasm;

    // 问题2修复：如果提供了分离的污点，使用分号分隔格式
    if (!dstTaint.empty() || !srcTaint.empty()) {
        // 使用分号分隔格式：DST;SRC
        std::string taintStr;
        if (!dstTaint.empty()) {
            taintStr = dstTaint.toString();
        }
        if (!srcTaint.empty()) {
            if (!taintStr.empty()) {
                taintStr += ";";
            }
            taintStr += srcTaint.toString();
        }
        line += "\t" + taintStr;
    } else if (!taint.empty()) {
        // 使用默认格式（逗号分隔）
        line += "\t" + taint.toString();
    } else {
        line += "\t-";
    }

    // 添加值信息
    if (!formattedValues.empty()) {
        // 使用预格式化的值字段（已符合规范）
        line += "\t" + formattedValues;
    } else if (!values.empty()) {
        // 兼容旧的 values map 格式（如果没有预格式化）
        line += "\t";
        bool first = true;
        for (const auto& kv : values) {
            if (kv.first == "_formatted") {
                // 跳过标记
                continue;
            }
            if (!first) {
                line += ";";
            }
            char valBuf[64];
            snprintf(valBuf, sizeof(valBuf), "%s=0x%lx", kv.first.c_str(), kv.second);
            line += valBuf;
            first = false;
        }
    }

    // 添加分支标记
    if (branchState == BRANCH_TAKEN) {
        line += "\tTAKEN";
    } else if (branchState == BRANCH_NOT_TAKEN) {
        line += "\tNOT_TAKEN";
    }

    // 添加循环标记
    if (loopType == LOOP_TYPE) {
        line += "\tLOOP";
    } else if (loopType == REPEAT_TYPE) {
        line += "\tREPEAT";
    }

    writeLine(line);
}

void Logger::logInstructionSimple(THREADID tid, const Address& addr, const std::string& disasm,
                                 const TaintSet& taint) {
    std::string line = makeThreadPrefix(tid) + "Instruction\t" + addr.toString() + ": " + disasm;

    if (!taint.empty()) {
        line += "\t" + taint.toString();
    } else {
        line += "\t-";
    }

    writeLine(line);
}

void Logger::logTaintInject(THREADID tid, const Address& addr, size_t byteCount) {
    char buf[256];
    snprintf(buf, sizeof(buf), "THREADID\t%lu\tTaint\t%s\t%zu",
             static_cast<unsigned long>(tid),
             addr.toString().c_str(), byteCount);
    writeLine(buf);
}

void Logger::logLoop(THREADID tid, const Address& addr, uint32_t size) {
    char buf[256];
    snprintf(buf, sizeof(buf), "THREADID\t%lu\tLOOP\t%s\t%u",
             static_cast<unsigned long>(tid),
             addr.toString().c_str(), size);
    writeLine(buf);
}

void Logger::logSystemFunction(THREADID tid, const std::string& funcName,
                               const std::vector<std::string>& taintedArgs) {
    std::string line = makeThreadPrefix(tid) + "Function\tsys\t" + funcName + "(tainted_args=";

    for (size_t i = 0; i < taintedArgs.size(); i++) {
        if (i > 0) {
            line += ",";
        }
        line += taintedArgs[i];
    }

    line += ")";
    writeLine(line);
}

void Logger::writeLine(const std::string& line) {
    if (!config::ENABLE_LOGGING || !initialized || !file) {
        return;
    }

    PIN_GetLock(&fileLock, PIN_ThreadId());

    fprintf(file, "%s\n", line.c_str());
    if (config::FLUSH_IMMEDIATELY) {
        fflush(file);
    }

    PIN_ReleaseLock(&fileLock);
}

void Logger::writeFormatted(const char* fmt, va_list args) {
    if (!config::ENABLE_LOGGING || !initialized || !file) {
        return;
    }

    PIN_GetLock(&fileLock, PIN_ThreadId());

    vfprintf(file, fmt, args);
    fprintf(file, "\n");
    if (config::FLUSH_IMMEDIATELY) {
        fflush(file);
    }

    PIN_ReleaseLock(&fileLock);
}
