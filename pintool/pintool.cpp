#include "pin.H"
#include "config.h"
#include "logger.h"
#include "address.h"
#include "taintset.h"
#include "taintstate.h"
#include "propagation.h"
#include "loopdetector.h"
#include "nethook.h"
#include "instrumentation.h"
#include "moduleinfo.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

// ============== PIN 工具元数据 ==============

KNOB<std::string> KnobOutputFile(
    KNOB_MODE_WRITEONCE,
    "pintool",
    "o",
    config::LOG_FILE,
    "taint record output file");

static std::string gLogFile;

INT32 Usage() {
    fprintf(stderr,
            "Taint Analysis PinTool v1.0\n"
            "Usage: pin -t ./pintool.so [-o taint_record.log] -- <target_program> [args...]\n"
            "\nConfiguration:\n"
            "  Default log file: %s\n"
            "  Override log file: -o <path>\n"
            "  Taint tracking: enabled\n"
            "  Loop detection: %s\n"
            "  BasicBlock recording: %s\n",
            config::LOG_FILE, config::RECORD_LOOPS ? "yes" : "no",
            config::RECORD_BASICBLOCK ? "yes" : "no");
    return -1;
}

// ============== 初始化函数 ==============

VOID onImageLoad(IMG img, VOID* v) {
    if (!IMG_Valid(img)) {
        return;
    }

    std::string imgName = IMG_Name(img);
    if (config::DEBUG_MODE) {
        fprintf(stderr, "[IMG] Loaded: %s\n", imgName.c_str());
    }

    // 添加模块信息到缓存
    ModuleInfoManager::addModule(img);

    // 进行镜像级插桩
    instrumentImage(img, v);
}

VOID onTrace(TRACE trace, VOID* v) {
    // 进行追踪级插桩
    instrumentTrace(trace, v);
}

VOID onInstruction(INS ins, VOID* v) {
    // 进行指令级插桩
    instrumentInstruction(ins, v);
}

VOID onThreadStart(THREADID tid, CONTEXT* ctxt, INT32 flags, VOID* v) {
    InstrumentationManager::onThreadStart(tid, ctxt, flags, v);
}

VOID onThreadEnd(THREADID tid, const CONTEXT* ctxt, INT32 code, VOID* v) {
    InstrumentationManager::onThreadEnd(tid, ctxt, code, v);
}

VOID onExit(INT32 code, VOID* v) {
    fprintf(stderr, "[PIN] Tool exiting with code %d\n", code);
    fprintf(stderr, "[LOG] Taint record written to: %s\n", gLogFile.c_str());

    // 关闭日志
    Logger::getInstance().close();

    // 清理资源
    ModuleInfoManager::fini();
    InstrumentationManager::fini();
    NetHook::fini();
}

// ============== PIN_Main（工具入口点） ==============

int main(int argc, char* argv[]) {
    // 初始化 PIN
    if (PIN_Init(argc, argv)) {
        return Usage();
    }

    fprintf(stderr, "=== Taint Analysis PinTool Initializing ===\n");
    gLogFile = KnobOutputFile.Value();
    fprintf(stderr, "[CONFIG] Log file: %s\n", gLogFile.c_str());
    fprintf(stderr, "[CONFIG] Enable logging: %s\n",
            config::ENABLE_LOGGING ? "yes" : "no");
    fprintf(stderr, "[CONFIG] Debug mode: %s\n", config::DEBUG_MODE ? "yes" : "no");

    // 初始化符号表（用于获取真实函数名）
    PIN_InitSymbols();
    fprintf(stderr, "[PIN] Symbol table initialized\n");

    // 初始化日志系统
    if (!Logger::getInstance().init(gLogFile.c_str())) {
        fprintf(stderr, "[ERROR] Failed to initialize logger\n");
        return 1;
    }

    // 初始化模块信息管理器
    ModuleInfoManager::init();

    // 初始化插桩管理器
    InstrumentationManager::init();

    // 初始化网络 Hook
    NetHook::init();

    // 注册插桩回调
    fprintf(stderr, "[PIN] Registering instrumentation callbacks...\n");

    // 镜像加载回调
    IMG_AddInstrumentFunction(onImageLoad, 0);

    // 追踪回调
    TRACE_AddInstrumentFunction(onTrace, 0);

    // 指令回调
    INS_AddInstrumentFunction(onInstruction, 0);

    // 线程回调
    PIN_AddThreadStartFunction(onThreadStart, 0);
    PIN_AddThreadFiniFunction(onThreadEnd, 0);

    // syscall 回调（read/recv 系列兜底）
    PIN_AddSyscallEntryFunction(onSyscallEntry, 0);
    PIN_AddSyscallExitFunction(onSyscallExit, 0);

    // 退出回调
    PIN_AddFiniFunction(onExit, 0);

    fprintf(stderr, "[PIN] Starting monitored execution...\n\n");

    // 启动被 PIN 的程序
    PIN_StartProgram();

    return 0;
}
