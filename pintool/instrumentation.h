#ifndef INSTRUMENTATION_H
#define INSTRUMENTATION_H

#include "pin.H"
#include "address.h"
#include "logger.h"
#include "taintstate.h"
#include "taintset.h"
#include "propagation.h"
#include "loopdetector.h"
#include <map>
#include <string>

// ============== 全局插桩管理器 ==============
class InstrumentationManager {
private:
    // 每线程的污点状态（TLS）
    static std::map<THREADID, TaintState*> threadStates;
    static PIN_LOCK stateLock;

    // 已记录的基本块集合（全局去重）
    static std::set<uint64_t> recordedBBLs;
    static PIN_LOCK bblLock;

    // 已记录的循环集合（全局去重）
    static std::set<uint64_t> recordedLoops;
    static PIN_LOCK loopLock;

    // 每线程当前 BBL 的循环状态（问题3/4修复：支持LOOP和REPEAT区分）
    static std::map<THREADID, bool> currentBBLIsLoop;      // LOOP标记
    static std::map<THREADID, bool> currentBBLIsRepeat;    // REPEAT标记
    static PIN_LOCK loopStateLock;

    // 问题2修复：每线程调用栈深度（用于区分LOOP vs REPEAT）
    static std::map<THREADID, int> threadCallDepth;
    static PIN_LOCK callDepthLock;

public:
    // 循环检测器（全局共享）
    static LoopDetector loopDetector;
    // ============== 初始化与销毁 ==============
    static void init();
    static void fini();

    // ============== TaintState 管理 ==============
    static TaintState* getTaintState(THREADID tid);
    static void releaseTaintState(THREADID tid);

    // ============== 记录基本块 ==============
    static bool shouldRecordBBL(uint64_t bblAddr);
    static void markBBLRecorded(uint64_t bblAddr);

    // ============== 记录循环 ==============
    static bool shouldRecordLoop(uint64_t bblAddr);
    static void markLoopRecorded(uint64_t bblAddr);

    // ============== 循环检测（问题3/4修复：支持LOOP和REPEAT区分） ==============
    static bool isInLoop(THREADID tid, uint64_t bblAddr, uint64_t rsp);
    static void setCurrentBBLLoopState(THREADID tid, bool isLoop, bool isRepeat);
    static bool getCurrentBBLIsLoop(THREADID tid);
    static bool getCurrentBBLIsRepeat(THREADID tid);

    // ============== 调用栈深度管理（问题2修复） ==============
    static int getCallDepth(THREADID tid);
    static void incrementCallDepth(THREADID tid);
    static void decrementCallDepth(THREADID tid);

    // ============== 线程生命周期 ==============
    static void onThreadStart(THREADID tid, CONTEXT* ctxt, INT32 flags,
                             VOID* v);
    static void onThreadEnd(THREADID tid, const CONTEXT* ctxt, INT32 code,
                           VOID* v);
};

// ============== 插桩回调函数 ==============

// 函数级回调
VOID onFunctionEntry(THREADID tid, const std::string* funcName,
                     const Address* addr);
VOID onFunctionExit(THREADID tid, const std::string* funcName,
                    const Address* addr);

// 基本块级回调
VOID onBasicBlockEntry(THREADID tid, uint64_t bblAddr, uint32_t bblSize, CONTEXT* ctxt);

// 指令级回调
VOID onInstructionExecute(THREADID tid, ADDRINT insAddr,
                         const std::string* disasm, const std::string* mnemonic,
                         CONTEXT* ctxt, ADDRINT bblAddr,
                         UINT32 memOpCount, ADDRINT memOp0, ADDRINT memOp1,
                         UINT32 memOp0Read, UINT32 memOp0Write,
                         UINT32 memOp1Read, UINT32 memOp1Write,
                         UINT32 op0IsMem, UINT32 op1IsMem,
                         UINT32 op0IsReg, UINT32 op1IsReg,
                         UINT32 memReadSize, UINT32 memWriteSize,
                         ADDRINT srcReg0, ADDRINT srcReg1, ADDRINT dstReg0, ADDRINT dstReg1,
                         UINT32 branchTaken, UINT32 isJumpInstruction, ADDRINT jumpTargetAddr);

// syscall 级回调（用于网络/文件读取兜底注入）
VOID onSyscallEntry(THREADID tid, CONTEXT* ctxt, SYSCALL_STANDARD std, VOID* v);
VOID onSyscallExit(THREADID tid, CONTEXT* ctxt, SYSCALL_STANDARD std, VOID* v);

// ============== 插桩注册函数 ==============

// 镜像加载时的插桩
VOID instrumentImage(IMG img, VOID* v);

// 追踪插桩（基本块级）
VOID instrumentTrace(TRACE trace, VOID* v);

// 指令插桩
VOID instrumentInstruction(INS ins, VOID* v);

#endif  // INSTRUMENTATION_H
