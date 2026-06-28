#include "instrumentation.h"
#include "moduleinfo.h"
#include "nethook.h"
#include <cxxabi.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <vector>
#include <map>
#include <sstream>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <time.h>
#include <asm/unistd.h>

namespace {
struct SyscallReadState {
    ADDRINT syscallNo = 0;
    ADDRINT fd = 0;
    ADDRINT buffer = 0;
    ADDRINT msgPtr = 0;
    bool active = false;
};

std::map<THREADID, SyscallReadState> g_syscallReadStates;
PIN_LOCK g_syscallReadLock;
std::set<ADDRINT> g_memsetPltTargets;
std::set<ADDRINT> g_bzeroPltTargets;
std::set<ADDRINT> g_memcpyPltTargets;
std::set<ADDRINT> g_memmovePltTargets;
std::set<ADDRINT> g_hookedNetworkRoutineAddrs;
std::set<ADDRINT> g_hookedTimeRoutineAddrs;
std::map<ADDRINT, size_t> g_allocSizes;
PIN_LOCK g_allocLock;
}  // namespace

// 静态成员初始化
std::map<THREADID, TaintState*> InstrumentationManager::threadStates;
PIN_LOCK InstrumentationManager::stateLock;
LoopDetector InstrumentationManager::loopDetector;
std::set<uint64_t> InstrumentationManager::recordedBBLs;
PIN_LOCK InstrumentationManager::bblLock;
std::set<uint64_t> InstrumentationManager::recordedLoops;
PIN_LOCK InstrumentationManager::loopLock;
std::map<THREADID, bool> InstrumentationManager::currentBBLIsLoop;    // 问题3/4修复
std::map<THREADID, bool> InstrumentationManager::currentBBLIsRepeat;  // 问题3/4修复
PIN_LOCK InstrumentationManager::loopStateLock;
std::map<THREADID, int> InstrumentationManager::threadCallDepth;  // 问题2修复
PIN_LOCK InstrumentationManager::callDepthLock;

void InstrumentationManager::init() {
    PIN_InitLock(&stateLock);
    PIN_InitLock(&bblLock);
    PIN_InitLock(&loopLock);
    PIN_InitLock(&loopStateLock);
    PIN_InitLock(&callDepthLock);  // 问题2修复
    PIN_InitLock(&g_syscallReadLock);
    PIN_InitLock(&g_allocLock);
}

void InstrumentationManager::fini() {
    // PIN_LOCK doesn't require explicit cleanup
}

TaintState* InstrumentationManager::getTaintState(THREADID tid) {
    PIN_GetLock(&stateLock, PIN_ThreadId());

    if (threadStates.find(tid) == threadStates.end()) {
        threadStates[tid] = new TaintState();
    }

    TaintState* state = threadStates[tid];
    PIN_ReleaseLock(&stateLock);

    return state;
}

void InstrumentationManager::releaseTaintState(THREADID tid) {
    PIN_GetLock(&stateLock, PIN_ThreadId());

    if (threadStates.find(tid) != threadStates.end()) {
        delete threadStates[tid];
        threadStates.erase(tid);
    }

    PIN_ReleaseLock(&stateLock);
}

bool InstrumentationManager::shouldRecordBBL(uint64_t bblAddr) {
    if (!config::RECORD_BASICBLOCK) {
        return false;
    }

    PIN_GetLock(&bblLock, PIN_ThreadId());

    bool should = recordedBBLs.find(bblAddr) == recordedBBLs.end();

    PIN_ReleaseLock(&bblLock);

    return should;
}

void InstrumentationManager::markBBLRecorded(uint64_t bblAddr) {
    PIN_GetLock(&bblLock, PIN_ThreadId());

    recordedBBLs.insert(bblAddr);

    PIN_ReleaseLock(&bblLock);
}

bool InstrumentationManager::shouldRecordLoop(uint64_t bblAddr) {
    if (!config::RECORD_LOOPS) {
        return false;
    }

    if (!config::DEDUPLICATE_LOOPS) {
        return true;
    }

    PIN_GetLock(&loopLock, PIN_ThreadId());

    bool should = recordedLoops.find(bblAddr) == recordedLoops.end();

    PIN_ReleaseLock(&loopLock);

    return should;
}

void InstrumentationManager::markLoopRecorded(uint64_t bblAddr) {
    PIN_GetLock(&loopLock, PIN_ThreadId());

    recordedLoops.insert(bblAddr);

    PIN_ReleaseLock(&loopLock);
}

bool InstrumentationManager::isInLoop(THREADID tid, uint64_t bblAddr, uint64_t rsp) {
    // 获取当前调用深度
    int callDepth = getCallDepth(tid);
    return loopDetector.isInLoop(tid, bblAddr, rsp, callDepth);
}

void InstrumentationManager::setCurrentBBLLoopState(THREADID tid, bool isLoop, bool isRepeat) {
    PIN_GetLock(&loopStateLock, PIN_ThreadId());
    currentBBLIsLoop[tid] = isLoop;
    currentBBLIsRepeat[tid] = isRepeat;
    PIN_ReleaseLock(&loopStateLock);
}

bool InstrumentationManager::getCurrentBBLIsLoop(THREADID tid) {
    PIN_GetLock(&loopStateLock, PIN_ThreadId());
    bool isLoop = (currentBBLIsLoop.find(tid) != currentBBLIsLoop.end()) 
                  ? currentBBLIsLoop[tid] : false;
    PIN_ReleaseLock(&loopStateLock);
    return isLoop;
}

bool InstrumentationManager::getCurrentBBLIsRepeat(THREADID tid) {
    PIN_GetLock(&loopStateLock, PIN_ThreadId());
    bool isRepeat = (currentBBLIsRepeat.find(tid) != currentBBLIsRepeat.end()) 
                    ? currentBBLIsRepeat[tid] : false;
    PIN_ReleaseLock(&loopStateLock);
    return isRepeat;
}

// 问题2修复：调用栈深度管理方法
int InstrumentationManager::getCallDepth(THREADID tid) {
    PIN_GetLock(&callDepthLock, PIN_ThreadId());
    int depth = (threadCallDepth.find(tid) != threadCallDepth.end()) 
                ? threadCallDepth[tid] : 0;
    PIN_ReleaseLock(&callDepthLock);
    return depth;
}

void InstrumentationManager::incrementCallDepth(THREADID tid) {
    PIN_GetLock(&callDepthLock, PIN_ThreadId());
    threadCallDepth[tid]++;
    PIN_ReleaseLock(&callDepthLock);
}

void InstrumentationManager::decrementCallDepth(THREADID tid) {
    PIN_GetLock(&callDepthLock, PIN_ThreadId());
    if (threadCallDepth.find(tid) != threadCallDepth.end() && threadCallDepth[tid] > 0) {
        threadCallDepth[tid]--;
    }
    PIN_ReleaseLock(&callDepthLock);
}

void InstrumentationManager::onThreadStart(THREADID tid, CONTEXT* ctxt,
                                          INT32 flags, VOID* v) {
    getTaintState(tid);
    if (config::DEBUG_MODE) {
        fprintf(stderr, "[INFO] Thread %u started\n", tid);
    }
}

void InstrumentationManager::onThreadEnd(THREADID tid, const CONTEXT* ctxt,
                                        INT32 code, VOID* v) {
    releaseTaintState(tid);
    loopDetector.clearStack(tid);
    PIN_GetLock(&g_syscallReadLock, PIN_ThreadId());
    g_syscallReadStates.erase(tid);
    PIN_ReleaseLock(&g_syscallReadLock);
    if (config::DEBUG_MODE) {
        fprintf(stderr, "[INFO] Thread %u ended\n", tid);
    }
}

// ============== 回调函数实现 ==============

VOID onFunctionEntry(THREADID tid, const std::string* funcName,
                     const Address* addr) {
    if (!config::ENABLE_LOGGING) {
        return;
    }

    // 问题2修复：函数入口时增加调用栈深度
    InstrumentationManager::incrementCallDepth(tid);

    int callDepth = InstrumentationManager::getCallDepth(tid);
    InstrumentationManager::loopDetector.onFunctionEnter(tid, callDepth);

    if (config::RECORD_FUNCTION) {
        if (!(funcName && *funcName == ".text" && addr && addr->toString().find("[vdso]") != std::string::npos)) {
            Logger::getInstance().logFunctionEnter(tid, *funcName, *addr);
        }
    }

    if (config::DEBUG_MODE) {
        fprintf(stderr, "[FUNC] Enter: %s at %s\n", funcName->c_str(),
                addr->toString().c_str());
    }
}

VOID onFunctionExit(THREADID tid, const std::string* funcName,
                    const Address* addr) {
    if (!config::ENABLE_LOGGING) {
        return;
    }

    TaintState* state = InstrumentationManager::getTaintState(tid);

    int callDepth = InstrumentationManager::getCallDepth(tid);
    InstrumentationManager::loopDetector.onFunctionExit(tid, callDepth);

    // 问题2修复：函数出口时减少调用栈深度
    InstrumentationManager::decrementCallDepth(tid);

    if (state) {
        /*
         * 在函数返回边界清理 caller-saved 寄存器的旧污点，避免前序字段值
         * 在寄存器复用后“搭车”污染内部指针/游标等状态值。
         *
         * 这里保留 RAX：大多数整数/指针返回值经由 RAX 返回，若函数内部确实
         * 生成了新的 tainted 返回值，不应在此被直接抹掉。
         */
        static const REG callerSavedRegsToClear[] = {
            REG_RCX, REG_RDX, REG_RSI, REG_RDI,
            REG_R8,  REG_R9,  REG_R10, REG_R11
        };

        for (REG reg : callerSavedRegsToClear) {
            state->clearRegisterTaint(reg);
        }
    }

    if (config::RECORD_FUNCTION) {
        if (!(funcName && *funcName == ".text" && addr && addr->toString().find("[vdso]") != std::string::npos)) {
            Logger::getInstance().logFunctionExit(tid, *funcName);
        }
    }
}

// ============== 分支执行检测已集成到onInstructionExecute ==============

// ============== 网络函数 Hook 回调 ==============

// recv 系列函数的回调
static VOID recvCallback(THREADID tid, int socket, ADDRINT buffer,
                        size_t length, int flags, ADDRINT retval) {
    ssize_t ret = (ssize_t)retval;
    if (ret > 0) {
        NetHook::injectTaint(tid, (void*)buffer, ret, socket, false);
    }
}

// recvfrom 回调
static VOID recvfromCallback(THREADID tid, int socket, ADDRINT buffer,
                            size_t length, int flags, ADDRINT retval) {
    ssize_t ret = (ssize_t)retval;
    if (ret > 0) {
        NetHook::injectTaint(tid, (void*)buffer, ret, socket, false);
    }
}

// recvmsg 回调
static VOID recvmsgCallback(THREADID tid, int socket, ADDRINT msg_ptr,
                           int flags, ADDRINT retval) {
    ssize_t ret = (ssize_t)retval;
    if (ret > 0 && msg_ptr) {
        struct msghdr* msg = (struct msghdr*)msg_ptr;
        if (msg->msg_iov && msg->msg_iovlen > 0) {
            // 简化处理：只注入第一个 iovec
            void* buffer = msg->msg_iov[0].iov_base;
            size_t len = (ret < (ssize_t)msg->msg_iov[0].iov_len) 
                         ? ret : msg->msg_iov[0].iov_len;
            NetHook::injectTaint(tid, buffer, len, socket, false);
        }
    }
}

VOID onSyscallEntry(THREADID tid, CONTEXT* ctxt, SYSCALL_STANDARD std, VOID* v) {
    ADDRINT syscallNo = PIN_GetSyscallNumber(ctxt, std);

    SyscallReadState local;
    local.syscallNo = syscallNo;
    local.fd = PIN_GetSyscallArgument(ctxt, std, 0);
    local.active = true;

    if (syscallNo == __NR_recvfrom) {
        local.buffer = PIN_GetSyscallArgument(ctxt, std, 1);
    } else if (syscallNo == __NR_recvmsg) {
        local.msgPtr = PIN_GetSyscallArgument(ctxt, std, 1);
    } else {
        return;
    }

    PIN_GetLock(&g_syscallReadLock, PIN_ThreadId());
    g_syscallReadStates[tid] = local;
    PIN_ReleaseLock(&g_syscallReadLock);
}

VOID onSyscallExit(THREADID tid, CONTEXT* ctxt, SYSCALL_STANDARD std, VOID* v) {
    SyscallReadState state;
    bool found = false;
    PIN_GetLock(&g_syscallReadLock, PIN_ThreadId());
    std::map<THREADID, SyscallReadState>::iterator it = g_syscallReadStates.find(tid);
    if (it != g_syscallReadStates.end() && it->second.active) {
        state = it->second;
        g_syscallReadStates.erase(it);
        found = true;
    }
    PIN_ReleaseLock(&g_syscallReadLock);

    if (!found) {
        return;
    }

    ADDRINT retval = PIN_GetSyscallReturn(ctxt, std);
    ssize_t ret = (ssize_t)retval;
    if (ret <= 0) {
        return;
    }

    if (state.syscallNo == __NR_recvfrom && state.buffer) {
        NetHook::injectTaint(tid, (void*)state.buffer, (size_t)ret,
                             (int)state.fd, true);
        return;
    }

    if (state.syscallNo == __NR_recvmsg && state.msgPtr) {
        struct msghdr* msg = (struct msghdr*)state.msgPtr;
        if (msg->msg_iov && msg->msg_iovlen > 0 && msg->msg_iov[0].iov_base) {
            size_t len = (ret < (ssize_t)msg->msg_iov[0].iov_len) ? (size_t)ret : msg->msg_iov[0].iov_len;
            NetHook::injectTaint(tid, msg->msg_iov[0].iov_base, len,
                                 (int)state.fd, true);
        }
    }
}

// ============== 系统函数 Hook（P2）==============

// ============== IPOINT_BEFORE 回调：检查参数污点 ==============

// memcpy/memmove BEFORE: 检查 dst, src, len 是否被污染
static std::string formatTaintRangesForArg(const TaintSet& taint) {
    const std::set<uint64_t>& indices = taint.getIndices();
    if (indices.empty()) {
        return "";
    }

    std::ostringstream oss;
    bool firstRange = true;
    std::set<uint64_t>::const_iterator it = indices.begin();
    while (it != indices.end()) {
        uint64_t start = *it;
        uint64_t end = start;
        ++it;
        while (it != indices.end() && *it == end + 1) {
            end = *it;
            ++it;
        }

        if (!firstRange) {
            oss << ",";
        }
        if (start == end) {
            oss << start;
        } else {
            oss << start << "-" << end;
        }
        firstRange = false;
    }

    return oss.str();
}

static VOID memcpyBeforeCallback(THREADID tid, const char* funcName,
                                  ADDRINT dst, ADDRINT src, ADDRINT len) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;
    
    std::vector<std::string> taintedArgs;
    if (!state->getRegisterTaint(REG_RDI).empty()) taintedArgs.push_back("dst");
    if (!state->getRegisterTaint(REG_RSI).empty()) taintedArgs.push_back("src");
    if (!state->getRegisterTaint(REG_RDX).empty()) taintedArgs.push_back("len");

    TaintSet srcMemoryTaint;
    size_t inspectLen = (len > 1024) ? 1024 : static_cast<size_t>(len);
    for (size_t i = 0; i < inspectLen; ++i) {
        srcMemoryTaint.merge(state->getMemoryTaint(static_cast<uint64_t>(src + i)));
    }

    if (!srcMemoryTaint.empty()) {
        taintedArgs.push_back("srcmem=" + formatTaintRangesForArg(srcMemoryTaint));
    }
    
    if (!taintedArgs.empty()) {
        Logger::getInstance().logSystemFunction(tid, funcName, taintedArgs);
    }
}

static VOID pltCopyLikeBeforeCallback(THREADID tid, ADDRINT src, ADDRINT len) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;

    TaintSet srcMemoryTaint;
    size_t inspectLen = (len > 1024) ? 1024 : static_cast<size_t>(len);
    for (size_t i = 0; i < inspectLen; ++i) {
        srcMemoryTaint.merge(state->getMemoryTaint(static_cast<uint64_t>(src + i)));
    }

    if (!srcMemoryTaint.empty()) {
        std::vector<std::string> taintedArgs;
        taintedArgs.push_back("srcmem=" + formatTaintRangesForArg(srcMemoryTaint));
        Logger::getInstance().logSystemFunction(tid, "memcpy_like", taintedArgs);
    }
}

// memset BEFORE: 检查 s, c, n 是否被污染
static VOID memsetBeforeCallback(THREADID tid, ADDRINT s, ADDRINT c, ADDRINT n) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;

    // memset(ptr, 0, n) 是最常见的结构体/缓冲区清零路径。
    // 这里应优先相信实际参数值 c，而不是当前 RSI 上可能残留的旧寄存器 taint。
    TaintSet fillValueTaint;
    if (c != 0) {
        fillValueTaint = state->getRegisterTaint(REG_RSI);
    }

    if (s != 0 && n != 0) {
        for (ADDRINT i = 0; i < n; ++i) {
            const uint64_t addr = static_cast<uint64_t>(s + i);
            if (fillValueTaint.empty()) {
                state->clearMemoryTaint(addr);
            } else {
                state->setMemoryTaint(addr, fillValueTaint);
            }
        }
    }
    
    std::vector<std::string> taintedArgs;
    if (!state->getRegisterTaint(REG_RDI).empty()) taintedArgs.push_back("s");
    if (!state->getRegisterTaint(REG_RSI).empty()) taintedArgs.push_back("c");
    if (!state->getRegisterTaint(REG_RDX).empty()) taintedArgs.push_back("n");
    
    if (!taintedArgs.empty()) {
        Logger::getInstance().logSystemFunction(tid, "memset", taintedArgs);
    }
}

static VOID bzeroBeforeCallback(THREADID tid, ADDRINT s, ADDRINT n) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;

    if (s != 0 && n != 0) {
        for (ADDRINT i = 0; i < n; ++i) {
            state->clearMemoryTaint(static_cast<uint64_t>(s + i));
        }
    }

    std::vector<std::string> taintedArgs;
    if (!state->getRegisterTaint(REG_RDI).empty()) taintedArgs.push_back("s");
    if (!state->getRegisterTaint(REG_RSI).empty()) taintedArgs.push_back("n");

    if (!taintedArgs.empty()) {
        Logger::getInstance().logSystemFunction(tid, "bzero", taintedArgs);
    }
}

static VOID clearMemoryRangeTaint(TaintState* state, ADDRINT base, size_t size) {
    if (!state || base == 0 || size == 0) return;
    const uint64_t begin = static_cast<uint64_t>(base);
    const uint64_t end = begin + static_cast<uint64_t>(size);
    if (end < begin) {
        return;
    }

    std::map<uint64_t, TaintSet>::iterator it = state->memoryTaint.lower_bound(begin);
    while (it != state->memoryTaint.end() && it->first < end) {
        it = state->memoryTaint.erase(it);
    }
}

static VOID clearSystemTimeOutputCallback(THREADID tid, ADDRINT output,
                                          ADDRINT retval, size_t outputSize) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;

    if (retval == 0) {
        clearMemoryRangeTaint(state, output, outputSize);
    }
    state->clearRegisterTaint(REG_RAX);
}

static VOID gettimeofdayCallback(THREADID tid, ADDRINT timevalPtr, ADDRINT retval) {
    clearSystemTimeOutputCallback(tid, timevalPtr, retval, sizeof(struct timeval));
}

static VOID clockGettimeCallback(THREADID tid, ADDRINT timespecPtr, ADDRINT retval) {
    clearSystemTimeOutputCallback(tid, timespecPtr, retval, sizeof(struct timespec));
}

static VOID localtimeCallback(THREADID tid, ADDRINT retval) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;

    clearMemoryRangeTaint(state, retval, sizeof(struct tm));
    state->clearRegisterTaint(REG_RAX);
}

static VOID localtimeRCallback(THREADID tid, ADDRINT output, ADDRINT retval) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;

    if (retval != 0) {
        clearMemoryRangeTaint(state, output, sizeof(struct tm));
    }
    state->clearRegisterTaint(REG_RAX);
}

static bool checkedAllocSize(ADDRINT count, ADDRINT size, size_t* out) {
    if (!out) return false;
    if (count == 0 || size == 0) {
        *out = 0;
        return true;
    }
    const ADDRINT maxSize = static_cast<ADDRINT>(~static_cast<size_t>(0));
    if (count > maxSize / size) {
        return false;
    }
    *out = static_cast<size_t>(count * size);
    return true;
}

static void rememberAllocation(ADDRINT ptr, size_t size) {
    if (ptr == 0 || size == 0) return;
    PIN_GetLock(&g_allocLock, PIN_ThreadId());
    g_allocSizes[ptr] = size;
    PIN_ReleaseLock(&g_allocLock);
}

static bool forgetAllocation(ADDRINT ptr, size_t* sizeOut) {
    if (ptr == 0) return false;
    PIN_GetLock(&g_allocLock, PIN_ThreadId());
    std::map<ADDRINT, size_t>::iterator it = g_allocSizes.find(ptr);
    if (it == g_allocSizes.end()) {
        PIN_ReleaseLock(&g_allocLock);
        return false;
    }
    if (sizeOut) {
        *sizeOut = it->second;
    }
    g_allocSizes.erase(it);
    PIN_ReleaseLock(&g_allocLock);
    return true;
}

static VOID initializeENIPMessageBeforeCallback(THREADID tid, ADDRINT messagePtr) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    clearMemoryRangeTaint(state, messagePtr, 0x210);
}

static VOID initializeMessageRouterResponseBeforeCallback(THREADID tid, ADDRINT responsePtr) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    clearMemoryRangeTaint(state, responsePtr, 0x218);
}

// malloc BEFORE: 检查 size 是否被污染
static VOID mallocBeforeCallback(THREADID tid, ADDRINT size) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;
    
    std::vector<std::string> taintedArgs;
    if (!state->getRegisterTaint(REG_RDI).empty()) taintedArgs.push_back("size");
    
    if (!taintedArgs.empty()) {
        Logger::getInstance().logSystemFunction(tid, "malloc", taintedArgs);
    }
}

// calloc BEFORE: 检查 nmemb, size 是否被污染
static VOID callocBeforeCallback(THREADID tid, ADDRINT nmemb, ADDRINT size) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;
    
    std::vector<std::string> taintedArgs;
    if (!state->getRegisterTaint(REG_RDI).empty()) taintedArgs.push_back("nmemb");
    if (!state->getRegisterTaint(REG_RSI).empty()) taintedArgs.push_back("size");
    
    if (!taintedArgs.empty()) {
        Logger::getInstance().logSystemFunction(tid, "calloc", taintedArgs);
    }
}

// realloc BEFORE: 检查 ptr, size 是否被污染
static VOID reallocBeforeCallback(THREADID tid, ADDRINT ptr, ADDRINT size) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;
    
    std::vector<std::string> taintedArgs;
    if (!state->getRegisterTaint(REG_RDI).empty()) taintedArgs.push_back("ptr");
    if (!state->getRegisterTaint(REG_RSI).empty()) taintedArgs.push_back("size");
    
    if (!taintedArgs.empty()) {
        Logger::getInstance().logSystemFunction(tid, "realloc", taintedArgs);
    }
}

// free BEFORE: 检查 ptr 是否被污染
static VOID freeBeforeCallback(THREADID tid, ADDRINT ptr) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;

    size_t allocSize = 0;
    if (forgetAllocation(ptr, &allocSize)) {
        clearMemoryRangeTaint(state, ptr, allocSize);
    }
    
    std::vector<std::string> taintedArgs;
    if (!state->getRegisterTaint(REG_RDI).empty()) taintedArgs.push_back("ptr");
    
    if (!taintedArgs.empty()) {
        Logger::getInstance().logSystemFunction(tid, "free", taintedArgs);
    }
}

// ============== IPOINT_AFTER 回调：污点传播 ==============

// memcpy hook - 检查源参数是否有污点
static VOID memcpyCallback(THREADID tid, ADDRINT dst, ADDRINT src, ADDRINT len,
                          ADDRINT retval) {
    if (len <= 0) {
        return;
    }

    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) {
        return;
    }

    // 逐字节复制 taint：源字节无污点时，应同步清除目标字节的旧污点。
    size_t copyLen = (len > 1024) ? 1024 : len;
    for (size_t i = 0; i < copyLen; i++) {
        TaintSet srcTaint = state->getMemoryTaint((uint64_t)src + i);
        uint64_t dstAddr = (uint64_t)dst + i;
        if (srcTaint.empty()) {
            state->clearMemoryTaint(dstAddr);
        } else {
            TaintSet newTaint = TaintPropagation::propagateStoreMemory(srcTaint);
            state->setMemoryTaint(dstAddr, newTaint);
        }
    }
}

// malloc hook - 记录内存分配
static VOID mallocCallback(THREADID tid, ADDRINT size, ADDRINT retval) {
    if (retval == 0 || size == 0) {
        return;
    }

    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;

    const size_t allocSize = static_cast<size_t>(size);
    rememberAllocation(retval, allocSize);
    // Heap allocators routinely reuse addresses. A new object must not inherit
    // taint left on that address by a previous packet/object lifetime.
    clearMemoryRangeTaint(state, retval, allocSize);
}

// calloc hook - 类似 malloc
static VOID callocCallback(THREADID tid, ADDRINT nmemb, ADDRINT size, ADDRINT retval) {
    if (retval == 0) {
        return;
    }

    size_t allocSize = 0;
    if (!checkedAllocSize(nmemb, size, &allocSize) || allocSize == 0) {
        return;
    }

    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;

    rememberAllocation(retval, allocSize);
    // calloc returns zeroed memory, so all prior taint on the reused range is stale.
    clearMemoryRangeTaint(state, retval, allocSize);
}

static VOID reallocCallback(THREADID tid, ADDRINT oldPtr, ADDRINT size, ADDRINT retval) {
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) return;

    // realloc failure leaves the old allocation untouched.
    if (retval == 0 && size != 0) {
        return;
    }

    size_t oldSize = 0;
    if (oldPtr != 0 && forgetAllocation(oldPtr, &oldSize)) {
        clearMemoryRangeTaint(state, oldPtr, oldSize);
    }

    if (retval == 0 || size == 0) {
        return;
    }

    const size_t newSize = static_cast<size_t>(size);
    rememberAllocation(retval, newSize);
    // Conservative choice: do not preserve taint through realloc because allocator
    // reuse/move semantics are otherwise a common source of stale object taint.
    clearMemoryRangeTaint(state, retval, newSize);
}


VOID onBasicBlockEntry(THREADID tid, uint64_t bblAddr, uint32_t bblSize, CONTEXT* ctxt) {
    NetHook::clearRecentInjection(tid);

    if (!config::RECORD_BASICBLOCK) {
        return;
    }

    // 获取RSP值用于栈帧跟踪
    uint64_t rsp = PIN_GetContextReg(ctxt, REG_STACK_PTR);

    // 循环检测：使用新接口enterAndCheckLoop
    // 该接口会：1)清理已返回函数的历史 2)检测是否循环 3)将当前BBL加入历史
    // 返回值：0=不在循环，1=在循环中
    int loopType = InstrumentationManager::loopDetector.enterAndCheckLoop(tid, bblAddr, bblSize, rsp);
    bool isLoopType1 = (loopType == 1);  // LOOP标记
    bool isLoopType2 = false;            // REPEAT标记（暂时未使用）
    
    // 记录循环/重复状态供指令级使用
    InstrumentationManager::setCurrentBBLLoopState(tid, isLoopType1, isLoopType2);

    // 保存当前 BBL 信息，延迟到有污点指令时才输出
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (state) {
        state->currentBblAddr = bblAddr;
        state->currentBblSize = bblSize;
        state->currentBblTainted = false;
        state->currentBblLogged = false;
    }

    // 如果未启用“只输出污染BBL”模式，则保持原行为：每个BBL都输出
    if (!config::RECORD_ONLY_TAINTED_BASICBLOCKS) {
        Address addr = ModuleInfoManager::getAddress(bblAddr);
        std::string addrStr = addr.toString();
        // 过滤 vdso 和 unknown 模块的基本块
        if (addrStr.find("[vdso]") == std::string::npos &&
            addrStr.find("unknown") == std::string::npos) {
            Logger::getInstance().logBasicBlock(tid, addr, bblSize);
        }
        if (state) {
            state->currentBblLogged = true;  // 已输出，避免重复
        }
    }
    
    // 注意：不再需要单独调用enterBBL，enterAndCheckLoop已经完成了这个工作
}

// 问题3修复：BBL退出时不再弹栈，以保持历史记录用于循环检测
// 原先每个BBL退出都弹栈导致栈始终为空，无法检测到第2次迭代
// 现在改为：只在函数返回时清理（由ret指令的onReturn触发）
VOID onBasicBlockExit(THREADID tid) {
    // 重置当前 BBL 的 tainted 标记
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (state) {
        state->currentBblTainted = false;
        state->currentBblLogged = false;
    }
}

// 辅助函数：标记当前 BBL 为 tainted，并在首次标记时立即输出 BBL 行
// 确保 BasicBlock 行出现在其包含的 Instruction 行之前
static void markCurrentBblTainted(THREADID tid, TaintState* state) {
    if (!state || state->currentBblLogged) {
        return;  // 已输出或无状态，跳过
    }
    state->currentBblTainted = true;
    state->currentBblLogged = true;
    
    // 立即输出 BBL 行
    Address addr = ModuleInfoManager::getAddress(state->currentBblAddr);
    std::string addrStr = addr.toString();
    if (addrStr.find("[vdso]") == std::string::npos &&
        addrStr.find("unknown") == std::string::npos) {
        Logger::getInstance().logBasicBlock(tid, addr, state->currentBblSize);
    }
}

static bool isLikelyPointerValue(uint64_t value) {
    // Linux x86_64 用户态对象/模块/堆栈地址通常位于高地址区间。
    return value >= 0x0000400000000000ULL;
}

static bool isSmallFieldLikeTaint(const TaintSet& taint) {
    // 协议字段通常只覆盖少量连续字节；当这类 taint 演化为高地址值时，往往是对象指针噪声。
    return !taint.empty() && taint.size() <= 8;
}

static void updateMemoryWriteTaint(TaintState* state, ADDRINT addr, UINT32 writeSize,
                                   const TaintSet& taint) {
    if (!state || addr == 0) {
        return;
    }

    UINT32 size = (writeSize > 0) ? writeSize : 1;
    if (size > 1024) {
        size = 1024;
    }

    for (UINT32 i = 0; i < size; ++i) {
        uint64_t cur = static_cast<uint64_t>(addr) + i;
        if (taint.empty()) {
            state->clearMemoryTaint(cur);
        } else {
            state->setMemoryTaint(cur, taint);
        }
    }
}

static bool isStringStoreInstruction(const std::string& mnemonicText) {
    return mnemonicText == "stosb" || mnemonicText == "stosw" ||
           mnemonicText == "stosd" || mnemonicText == "stosq";
}

static REG getStringStoreAccumulatorReg(const std::string& mnemonicText) {
    if (mnemonicText == "stosb") return REG_AL;
    if (mnemonicText == "stosw") return REG_AX;
    if (mnemonicText == "stosd") return REG_EAX;
    if (mnemonicText == "stosq") return REG_RAX;
    return REG_INVALID();
}

// 辅助函数：将反汇编中的绝对跳转目标地址替换为 module+offset 格式
static std::string replaceJumpTargetAddress(const std::string& disasmText, ADDRINT jumpTarget) {
    if (jumpTarget == 0) {
        return disasmText;  // 间接跳转，无法替换
    }
    
    // 将目标地址转换为字符串进行匹配
    std::ostringstream targetStream;
    targetStream << "0x" << std::hex << jumpTarget;
    std::string targetStr = targetStream.str();
    
    // 在反汇编文本中查找目标地址
    size_t pos = disasmText.find(targetStr);
    if (pos == std::string::npos) {
        // 尝试不带0x前缀匹配（某些情况下PIN可能输出不同格式）
        std::ostringstream altStream;
        altStream << std::hex << jumpTarget;
        std::string altStr = altStream.str();
        pos = disasmText.find(altStr);
        if (pos == std::string::npos) {
            return disasmText;  // 未找到匹配
        }
        targetStr = altStr;
    }
    
    // 获取目标地址的 module+offset 格式
    Address addr = ModuleInfoManager::getAddress(jumpTarget);
    
    // 替换绝对地址为 module+offset 格式
    std::string result = disasmText;
    result.replace(pos, targetStr.length(), addr.toString());
    return result;
}

VOID onInstructionExecute(THREADID tid, ADDRINT insAddr,
                         const std::string* disasm,
                         const std::string* mnemonic, CONTEXT* ctxt,
                         ADDRINT bblAddr, UINT32 memOpCount, ADDRINT memOp0,
                         ADDRINT memOp1, UINT32 memOp0Read,
                         UINT32 memOp0Write, UINT32 memOp1Read,
                         UINT32 memOp1Write, UINT32 op0IsMem, UINT32 op1IsMem,
                         UINT32 op0IsReg, UINT32 op1IsReg,
                         UINT32 memReadSize, UINT32 memWriteSize,
                         ADDRINT srcReg0, ADDRINT srcReg1,
                         ADDRINT dstReg0, ADDRINT dstReg1, UINT32 branchTaken,
                         UINT32 isJumpInstruction, ADDRINT jumpTargetAddr) {
    // ===== 重要注意 =====
    // 不删除 disasm 和 mnemonic 指针，因为它们由 PIN 框架管理
    // 参数中的 srcReg0 等已经是 UINT32 格式，通过 static_cast<REG> 转换使用
    
    if (!config::RECORD_INSTRUCTION) {
        return;
    }

    // 获取线程状态
    TaintState* state = InstrumentationManager::getTaintState(tid);
    if (!state) {
        return;
    }

    // 获取指令的模块信息
    Address addr = ModuleInfoManager::getAddress(insAddr);

    // 获取反汇编文本（安全读取），并替换跳转目标地址为 module+offset 格式
    std::string disasmText = disasm ? *disasm : "";
    if (isJumpInstruction && jumpTargetAddr != 0) {
        disasmText = replaceJumpTargetAddress(disasmText, jumpTargetAddr);
    }
    
    // 获取当前 BBL 的循环/重复状态（问题3/4修复）
    bool isLoop = InstrumentationManager::getCurrentBBLIsLoop(tid);
    bool isRepeat = InstrumentationManager::getCurrentBBLIsRepeat(tid);

    // 解析指令助记符（转小写）
    std::string mnemonicText = mnemonic ? *mnemonic : "";
    std::transform(mnemonicText.begin(), mnemonicText.end(), mnemonicText.begin(), ::tolower);

    // 将 UINT32 REG 参数转换为 REG 类型
    REG srcReg0_val = static_cast<REG>(srcReg0);
    REG srcReg1_val = static_cast<REG>(srcReg1);
    REG dstReg0_val = static_cast<REG>(dstReg0);
    REG dstReg1_val = static_cast<REG>(dstReg1);

    // 收集源操作数污点（问题2修复：分别跟踪以支持DST;SRC格式）
    TaintSet srcTaint0;  // 第一个源操作数
    TaintSet srcTaint1;  // 第二个源操作数
    if (srcReg0_val != REG_INVALID()) {
        REG normalizedReg0 = REG_FullRegName(srcReg0_val);
        if (normalizedReg0 != REG_INVALID()) {
            srcTaint0 = state->getRegisterTaint(normalizedReg0);
        }
    }
    if (srcReg1_val != REG_INVALID()) {
        REG normalizedReg1 = REG_FullRegName(srcReg1_val);
        if (normalizedReg1 != REG_INVALID()) {
            srcTaint1 = state->getRegisterTaint(normalizedReg1);
        }
    }
    
    // 合并所有源污点（用于污点传播）
    TaintSet srcTaint = srcTaint0.unite(srcTaint1);

    // 内存读取污点聚合（遍历所有读取的字节）
    // 同时记录被污染字节的位置，用于后续只读取被污染字节的值
    TaintSet memReadTaint;
    std::vector<size_t> memOp0TaintedOffsets;  // memOp0中被污染的字节偏移
    std::vector<size_t> memOp1TaintedOffsets;  // memOp1中被污染的字节偏移
    
    if (memOpCount > 0 && memOp0Read && memOp0 != 0) {
        size_t readSize = (memReadSize > 0) ? memReadSize : 1;
        for (size_t i = 0; i < readSize; ++i) {
            TaintSet taint = state->getMemoryTaint(static_cast<uint64_t>(memOp0 + i));
            if (!taint.empty()) {
                memOp0TaintedOffsets.push_back(i);  // 记录被污染的偏移
            }
            memReadTaint += taint;
            srcTaint += taint;
        }
    }
    if (memOpCount > 1 && memOp1Read && memOp1 != 0) {
        size_t readSize = (memReadSize > 0) ? memReadSize : 1;
        for (size_t i = 0; i < readSize; ++i) {
            TaintSet taint = state->getMemoryTaint(static_cast<uint64_t>(memOp1 + i));
            if (!taint.empty()) {
                memOp1TaintedOffsets.push_back(i);  // 记录被污染的偏移
            }
            memReadTaint += taint;
            srcTaint += taint;
        }
    }

    // 收集目的寄存器
    std::vector<REG> dstRegs;
    if (dstReg0_val != REG_INVALID()) {
        REG normalizedDst0 = REG_FullRegName(dstReg0_val);
        if (normalizedDst0 != REG_INVALID()) {
            dstRegs.push_back(normalizedDst0);
        }
    }
    if (dstReg1_val != REG_INVALID()) {
        REG normalizedDst1 = REG_FullRegName(dstReg1_val);
        if (normalizedDst1 != REG_INVALID()) {
            dstRegs.push_back(normalizedDst1);
        }
    }

    // 污点传播（简化规则）
    TaintSet dstOldTaint;
    for (REG reg : dstRegs) {
        REG normalizedReg = REG_FullRegName(reg);
        if (normalizedReg == REG_RFLAGS || normalizedReg == REG_EFLAGS || normalizedReg == REG_FLAGS) {
            continue;
        }
        dstOldTaint += state->getRegisterTaint(reg);
    }

    auto updateDstRegs = [&](const TaintSet& newTaint) {
        for (REG reg : dstRegs) {
            if (newTaint.empty()) {
                state->clearRegisterTaint(reg);
            } else {
                state->setRegisterTaint(reg, newTaint);
            }
        }
    };

    // ===== 提取操作数值（仅记录被污染的操作数） =====
    // 格式规范（per core_requests.txt 4.3.2）：
    // - 仅DST被污染：DST=<hex>
    // - 仅SRC被污染：SRC=<hex>
    // - DST和SRC均被污染：DST=<hex>;SRC=<hex>
    std::string valueField;
    bool suppressPointerLikeTaint = false;
    
    // 辅助函数：检查寄存器是否可以用 PIN_GetContextReg 读取
    auto isRegisterReadable = [](REG reg) -> bool {
        if (reg == REG_INVALID()) return false;
        // 排除不支持的寄存器类型
        if (REG_is_xmm(reg) || REG_is_ymm(reg) || REG_is_zmm(reg)) {
            return false;  // AVX/AVX2/AVX512 向量寄存器
        }
        if (REG_is_mm(reg)) {
            return false;  // MMX 寄存器
        }
        if (REG_is_seg(reg)) {
            return false;  // 段寄存器
        }
        return true;
    };
    
    // 辅助函数：从内存读取仅被污染字节的值
    // 根据 core_requests.txt 4.3.2: "值的宽度与对应的源/目的操作数宽度一致"
    // 这里我们理解为：只输出被污染字节的值，而不是整个操作数
    auto readTaintedBytesValue = [](ADDRINT memAddr, const std::vector<size_t>& taintedOffsets) -> uint64_t {
        if (taintedOffsets.empty()) {
            return 0;
        }
        
        // 只读取被污染的字节，按序组合成值
        uint64_t value = 0;
        size_t byteIndex = 0;
        for (size_t offset : taintedOffsets) {
            if (byteIndex >= 8) break;  // 最多8字节
            
            if (PIN_CheckReadAccess((void*)(memAddr + offset))) {
                uint8_t byte = 0;
                PIN_SafeCopy(&byte, (void*)(memAddr + offset), 1);
                value |= (static_cast<uint64_t>(byte) << (byteIndex * 8));
                byteIndex++;
            }
        }
        return value;
    };

    auto readMemoryValue = [](ADDRINT memAddr, size_t readSize) -> uint64_t {
        if (memAddr == 0 || readSize == 0) {
            return 0;
        }

        uint64_t value = 0;
        size_t width = (readSize > 8) ? 8 : readSize;
        for (size_t i = 0; i < width; ++i) {
            if (!PIN_CheckReadAccess((void*)(memAddr + i))) {
                break;
            }
            uint8_t byte = 0;
            PIN_SafeCopy(&byte, (void*)(memAddr + i), 1);
            value |= (static_cast<uint64_t>(byte) << (i * 8));
        }
        return value;
    };
    
    // 检查源操作数是否被污染
    bool srcTainted = !srcTaint.empty();
    bool isBinaryOpForValue = TaintPropagation::isBinaryArithmetic(mnemonicText);
    bool isShiftOpForValue = TaintPropagation::isShiftInstruction(mnemonicText);
    bool isCmpForValue = TaintPropagation::isCmpInstruction(mnemonicText);
    TaintSet cmpOp0Taint;
    TaintSet cmpOp1Taint;
    if (isCmpForValue) {
        cmpOp0Taint = op0IsMem ? memReadTaint : srcTaint0;
        cmpOp1Taint = op1IsMem ? memReadTaint : srcTaint1;
    }
    TaintSet arithOp0Taint;
    TaintSet arithOp1Taint;
    if (isBinaryOpForValue || isShiftOpForValue) {
        if (op0IsMem) {
            arithOp0Taint = memReadTaint;
        } else if (op0IsReg) {
            arithOp0Taint = srcTaint0;
        }

        if (op1IsMem) {
            arithOp1Taint = memReadTaint;
        } else if (op1IsReg) {
            // 对于 add [mem], reg / shl [mem], cl 这类指令，唯一显式寄存器源会落在 srcTaint0
            arithOp1Taint = op0IsReg ? srcTaint1 : srcTaint0;
        }
    }
    
    // 检查目的操作数是否被污染
    bool dstTainted = false;
    if (isCmpForValue) {
        // CMP/TEST: DST 是第一个源操作数（不是真正的目的寄存器，因为 CMP 只写 FLAGS）
        dstTainted = !cmpOp0Taint.empty();
    } else if (isBinaryOpForValue || isShiftOpForValue) {
        // 二元算术/移位：DST 是第一个显式操作数执行前的旧值，可能是寄存器也可能是内存
        dstTainted = !arithOp0Taint.empty();
    } else {
        for (size_t i = 0; i < dstRegs.size(); ++i) {
            REG reg = dstRegs[i];
            REG normalizedReg = REG_FullRegName(reg);
            if (!state->getRegisterTaint(normalizedReg).empty()) {
                dstTainted = true;
                break;
            }
        }
    }
    
    // 对于二进制操作，需要检查合并后的污点
    if (TaintPropagation::isBinaryArithmetic(mnemonicText) && srcTainted && dstTainted) {
        // 二进制操作：merge 后，目的同时被源和目的污染
        dstTainted = true;
    }
    
    // 提取目的值（如果被污染）
    uint64_t dstValue = 0;
    bool dstValueAvailable = false;
    if ((dstTainted || isCmpForValue) && ctxt) {
        if (isCmpForValue) {
            // CMP/TEST: DST 值从第一个源操作数获取
            if (op0IsMem && memOpCount > 0 && memOp0Read && memOp0 != 0) {
                dstValue = !memOp0TaintedOffsets.empty()
                               ? readTaintedBytesValue(memOp0, memOp0TaintedOffsets)
                               : readMemoryValue(memOp0, memReadSize > 0 ? memReadSize : 1);
                dstValueAvailable = true;
            } else if (srcReg0_val != REG_INVALID()) {
                REG normalizedReg0 = REG_FullRegName(srcReg0_val);
                if (normalizedReg0 != REG_INVALID() && isRegisterReadable(normalizedReg0)) {
                    dstValue = PIN_GetContextReg(ctxt, normalizedReg0);
                    dstValueAvailable = true;
                }
            }
        } else if (isBinaryOpForValue || isShiftOpForValue) {
            if (op0IsMem && memOpCount > 0 && memOp0Read && memOp0 != 0 && !memReadTaint.empty()) {
                dstValue = readTaintedBytesValue(memOp0, memOp0TaintedOffsets);
                dstValueAvailable = true;
            } else if (op0IsReg && srcReg0_val != REG_INVALID()) {
                REG normalizedReg0 = REG_FullRegName(srcReg0_val);
                if (normalizedReg0 != REG_INVALID() && isRegisterReadable(normalizedReg0)) {
                    dstValue = PIN_GetContextReg(ctxt, normalizedReg0);
                    dstValueAvailable = true;
                }
            }
        } else if (!dstRegs.empty()) {
            REG dstReg = dstRegs[0];
            if (isRegisterReadable(dstReg)) {
                dstValue = PIN_GetContextReg(ctxt, dstReg);
                dstValueAvailable = true;
            }
        }
    }
    
    // 提取源值（如果被污染）
    uint64_t srcValue = 0;
    bool srcValueAvailable = false;
    std::string srcValueStr;  // 用于存储多源值的字符串表示
    bool valueSrcTainted = srcTainted;
    if (isBinaryOpForValue) {
        // 二元操作：只考虑第二个源或内存读取（立即数不污染）
        valueSrcTainted = !arithOp1Taint.empty();
    } else if (isShiftOpForValue) {
        // 移位指令的 SRC 是移位量；立即数不污染，寄存器计数（如 cl）则可能被污染
        valueSrcTainted = !arithOp1Taint.empty();
    } else if (isCmpForValue) {
        // CMP/TEST: SRC 是第二个源操作数
        valueSrcTainted = !cmpOp1Taint.empty();
    }
    
    if ((valueSrcTainted || isCmpForValue) && ctxt) {
        // CMP/TEST 特殊处理：从第二个源操作数获取 srcValue
        if (isCmpForValue) {
            if (op1IsMem && memOpCount > 0 && memOp0Read && memOp0 != 0) {
                srcValue = !memOp0TaintedOffsets.empty()
                               ? readTaintedBytesValue(memOp0, memOp0TaintedOffsets)
                               : readMemoryValue(memOp0, memReadSize > 0 ? memReadSize : 1);
                srcValueAvailable = true;
                goto format_values;
            }
            if (srcReg1_val != REG_INVALID()) {
                REG normalizedReg1 = REG_FullRegName(srcReg1_val);
                if (normalizedReg1 != REG_INVALID() && isRegisterReadable(normalizedReg1)) {
                    srcValue = PIN_GetContextReg(ctxt, normalizedReg1);
                    srcValueAvailable = true;
                    goto format_values;
                }
            }
            // 检查内存操作数
            if (op0IsMem && memOpCount > 0 && memOp0Read && memOp0 != 0) {
                if (!memReadTaint.empty()) {
                    // 只读取被污染字节的值
                    srcValue = readTaintedBytesValue(memOp0, memOp0TaintedOffsets);
                    srcValueAvailable = true;
                    goto format_values;
                }
            }
        } else if (isBinaryOpForValue || isShiftOpForValue) {
            if (op1IsMem && memOpCount > 0 && memOp0Read && memOp0 != 0 && !memReadTaint.empty()) {
                srcValue = readTaintedBytesValue(memOp0, memOp0TaintedOffsets);
                srcValueAvailable = true;
                goto format_values;
            }
            REG explicitSrcReg = REG_INVALID();
            if (op1IsReg) {
                explicitSrcReg = (op0IsReg ? srcReg1_val : srcReg0_val);
            }
            if (explicitSrcReg != REG_INVALID()) {
                REG normalizedSrc = REG_FullRegName(explicitSrcReg);
                if (normalizedSrc != REG_INVALID() && isRegisterReadable(normalizedSrc)) {
                    srcValue = PIN_GetContextReg(ctxt, normalizedSrc);
                    srcValueAvailable = true;
                    goto format_values;
                }
            }
        }
        
        // 问题修复2：处理LEA指令的多源操作数
        // LEA格式：lea edx, [rbp+rdx*1+0xa]
        // 包含多个源寄存器（基址rbp、索引rdx），需要提取所有值
        bool isLeaInstruction = mnemonicText == "lea";
        
        if (isLeaInstruction) {
            // LEA指令：收集所有源寄存器值
            std::vector<uint64_t> leaSrcValues;
            
            // 提取显式源寄存器（如果有）
            if (srcReg0_val != REG_INVALID()) {
                REG normalizedReg0 = REG_FullRegName(srcReg0_val);
                if (normalizedReg0 != REG_INVALID() && isRegisterReadable(normalizedReg0)) {
                    uint64_t val = PIN_GetContextReg(ctxt, normalizedReg0);
                    leaSrcValues.push_back(val);
                    srcValueAvailable = true;
                }
            }
            if (srcReg1_val != REG_INVALID()) {
                REG normalizedReg1 = REG_FullRegName(srcReg1_val);
                if (normalizedReg1 != REG_INVALID() && isRegisterReadable(normalizedReg1)) {
                    uint64_t val = PIN_GetContextReg(ctxt, normalizedReg1);
                    leaSrcValues.push_back(val);
                    srcValueAvailable = true;
                }
            }
            
            // 提取内存操作数中的寄存器（这是LEA的特殊之处：虽然是内存格式但不真正读内存）
            // 对于内存操作数，我们已经从memOp0Read中检查了，但需要从srcReg0/srcReg1中提取
            // 注：这里已经通过上面的srcReg0_val/srcReg1_val处理了
            
            // 将所有源值格式化为字符串
            if (!leaSrcValues.empty()) {
                char buf[256];
                if (leaSrcValues.size() == 1) {
                    snprintf(buf, sizeof(buf), "0x%lx", leaSrcValues[0]);
                } else {
                    snprintf(buf, sizeof(buf), "0x%lx,0x%lx", leaSrcValues[0], leaSrcValues[1]);
                    if (leaSrcValues.size() > 2) {
                        // 如果有超过2个源值（罕见），继续拼接
                        for (size_t i = 2; i < leaSrcValues.size(); ++i) {
                            char tmp[64];
                            snprintf(tmp, sizeof(tmp), ",0x%lx", leaSrcValues[i]);
                            strncat(buf, tmp, sizeof(buf) - strlen(buf) - 1);
                        }
                    }
                }
                srcValueStr = buf;
                srcValue = leaSrcValues[0];  // 主源值取第一个
            }
            goto format_values;  // LEA已处理，跳到格式化
        }
        
        // 问题修复1：处理隐含操作数指令（CDQ/CQO/CWDE等）
        // 这些指令的源操作数是隐含的 RAX 或 EAX，需要显式读取
        bool isImplicitSrcInstruction = mnemonicText == "cdq"
                                       || mnemonicText == "cqo"
                                       || mnemonicText == "cwde"
                                       || mnemonicText == "cbw";
        
        if (isImplicitSrcInstruction) {
            // CDQ/CQO/CWDE/CBW 的隐含源是 RAX（符号扩展源）
            REG implicitSrcReg = REG_FullRegName(REG_RAX);
            if (isRegisterReadable(implicitSrcReg)) {
                srcValue = PIN_GetContextReg(ctxt, implicitSrcReg);
                srcValueAvailable = true;
                goto format_values;
            }
        }
        
        if (isBinaryOpForValue) {
            // 二元操作：优先使用第二个源寄存器值
            if (srcReg1_val != REG_INVALID()) {
                REG normalizedReg1 = REG_FullRegName(srcReg1_val);
                if (normalizedReg1 != REG_INVALID() && isRegisterReadable(normalizedReg1)) {
                    srcValue = PIN_GetContextReg(ctxt, normalizedReg1);
                    srcValueAvailable = true;
                    goto format_values;  // 已获取源值，跳到格式化
                }
            }
            // 二元操作：检查源是否来自内存
            if (memOpCount > 0 && memOp0Read && memOp0 != 0) {
                if (!memReadTaint.empty()) {
                    // 只读取被污染字节的值
                    srcValue = readTaintedBytesValue(memOp0, memOp0TaintedOffsets);
                    srcValueAvailable = true;
                }
            }
        } else {
            // 非二元操作：对于 load 指令，优先从内存读取（内存是真正的源）
            // 基址寄存器（如 r12）只是地址，不是数据
            bool hasMemoryRead = (memOpCount > 0 && memOp0Read && memOp0 != 0);
            
            if (hasMemoryRead && !memReadTaint.empty()) {
                // 优先从内存读取被污染字节的值
                srcValue = readTaintedBytesValue(memOp0, memOp0TaintedOffsets);
                srcValueAvailable = true;
                goto format_values;
            }
            
            // 如果没有内存读取（如 mov rax, rbx），才使用源寄存器值
            if (srcReg0_val != REG_INVALID()) {
                REG normalizedReg0 = REG_FullRegName(srcReg0_val);
                if (normalizedReg0 != REG_INVALID() && isRegisterReadable(normalizedReg0)) {
                    srcValue = PIN_GetContextReg(ctxt, normalizedReg0);
                    srcValueAvailable = true;
                    goto format_values;
                }
            }
        }
    }
    
format_values:
    // 检测指令类型，决定是否记录值字段
    // 问题修复：处理多隐含操作数指令（IDIV/DIV等）
    // 这些指令隐含使用EDX:EAX作为被除数，需要单独处理
    bool isImplicitMultiOpInstruction = mnemonicText == "idiv"
                                       || mnemonicText == "div";
    bool isMovLikeInstruction = TaintPropagation::isMovInstruction(mnemonicText)
                               || mnemonicText == "movzx"
                               || mnemonicText == "movsx"
                               || mnemonicText == "movsxd";
    bool isLeaInstruction = TaintPropagation::isLeaInstruction(mnemonicText);
    bool isStringStoreOp = isStringStoreInstruction(mnemonicText);
    
    if (isImplicitMultiOpInstruction && srcTainted) {
        // IDIV/DIV: 隐含使用 EDX:EAX 作为被除数，显式使用另一个操作数
        // 为了完整性，记录被除数的高低32位
        uint64_t edxValue = 0, eaxValue = 0;
        
        REG rax = REG_FullRegName(REG_RAX);
        REG rdx = REG_FullRegName(REG_RDX);
        
        if (isRegisterReadable(rax)) {
            eaxValue = PIN_GetContextReg(ctxt, rax);
        }
        if (isRegisterReadable(rdx)) {
            edxValue = PIN_GetContextReg(ctxt, rdx);
        }
        
        // 输出格式：被除数的高低位和除数
        // SRC记录的是被除数和除数
        char buf[256];
        snprintf(buf, sizeof(buf), "SRC=0x%lx,0x%lx,0x%lx", edxValue, eaxValue, srcValue);
        valueField = buf;
    } else {
        // 常规指令的值字段处理
    // 问题1修复：跳转指令不输出值字段（仅输出TAKEN/NOT_TAKEN）
    
    if (isJumpInstruction) {
        // 根据core_requests 5.3节：跳转指令仅标明TAKEN/NOT_TAKEN，不输出值
        valueField = "";
    } else {
        // 非跳转指令：正常处理值字段
        bool isLoadInstruction = (memOpCount > 0 && memOp0Read && !memOp0Write) ||
                                 (memOpCount > 1 && memOp1Read && !memOp1Read);
        bool isStoreInstruction = (memOpCount > 0 && memOp0Write && !memOp0Read) ||
                                  (memOpCount > 1 && memOp1Write && !memOp1Read);
        bool isBinaryOp = isBinaryOpForValue;
        bool isShiftOp = isShiftOpForValue;

        // 问题修复：识别MOV/LEA类指令 - 这些指令的DST会被完全覆盖，执行前的值不是"使用的值"

        // 格式化值字段：按规范组织
        // 对于加载和存储指令，仅记录源值（SRC）
        // 对于二元操作，可能需要记录源和目的
        bool cmpHasImmediateOperand = isCmpForValue && !(op1IsMem || op1IsReg);
        bool cmpUseStarLabels = isCmpForValue && !cmpHasImmediateOperand;
        bool binaryHasImmediateOperand = isBinaryOpForValue && !(op1IsMem || op1IsReg);
        bool binaryUseStarLabels = isBinaryOpForValue && !binaryHasImmediateOperand;
        if (cmpUseStarLabels && (dstValueAvailable || srcValueAvailable)) {
            char buf[128];
            const char* dstLabel = dstTainted ? "DST*" : "DST";
            const char* srcLabel = valueSrcTainted ? "SRC*" : "SRC";
            snprintf(buf, sizeof(buf), "%s=0x%lx;%s=0x%lx", dstLabel, dstValue, srcLabel, srcValue);
            valueField = buf;
        } else if (binaryUseStarLabels && (dstValueAvailable || srcValueAvailable)) {
            char buf[128];
            const char* dstLabel = dstTainted ? "DST*" : "DST";
            const char* srcLabel = valueSrcTainted ? "SRC*" : "SRC";
            snprintf(buf, sizeof(buf), "%s=0x%lx;%s=0x%lx", dstLabel, dstValue, srcLabel, srcValue);
            valueField = buf;
        } else if (isMovLikeInstruction && valueSrcTainted) {
            // MOV类指令：只记录真实使用的值(SRC)，不记录执行前的DST
            // 因为DST会被完全覆盖，执行前的值不属于"使用的值"
            char buf[64];
            snprintf(buf, sizeof(buf), "SRC=0x%lx", srcValue);
            valueField = buf;
        } else if (isLeaInstruction && valueSrcTainted) {
            // LEA为地址计算指令：仅记录参与计算的源值
            char buf[256];
            if (!srcValueStr.empty()) {
                snprintf(buf, sizeof(buf), "SRC=%s", srcValueStr.c_str());
            } else {
                snprintf(buf, sizeof(buf), "SRC=0x%lx", srcValue);
            }
            valueField = buf;
        } else if (isLoadInstruction && valueSrcTainted) {
            // 问题修复3：Load指令应从内存读取真实值（不是地址）
            // 若已通过SafeCopy读取则srcValue为真实值，直接输出
            char buf[64];
            snprintf(buf, sizeof(buf), "SRC=0x%lx", srcValue);
            valueField = buf;
        } else if (isStoreInstruction && valueSrcTainted) {
            // Store指令：仅记录SRC，不记录DST（DST是地址）
            char buf[64];
            snprintf(buf, sizeof(buf), "SRC=0x%lx", srcValue);
            valueField = buf;
        } else if (isShiftOp && dstTainted) {
            // 移位指令：只记录被移位的值（DST），移位量是立即数不需要记录
            // 例如：shr al, 0x7 只记录 al 的值，不记录立即数 0x7
            char buf[64];
            snprintf(buf, sizeof(buf), "DST=0x%lx", dstValue);
            valueField = buf;
        } else if (isBinaryOp && valueSrcTainted && dstTainted) {
            // 二元操作：源与目的均被污染(都被使用)：DST=<hex>;SRC=<hex>
            char buf[128];
            snprintf(buf, sizeof(buf), "DST=0x%lx;SRC=0x%lx", dstValue, srcValue);
            valueField = buf;
        } else if (isBinaryOp && dstTainted) {
            // 二元操作：仅目的被污染（作为第二个源操作数被使用）：DST=<hex>
            char buf[64];
            snprintf(buf, sizeof(buf), "DST=0x%lx", dstValue);
            valueField = buf;
        } else if (valueSrcTainted && dstTainted) {
            // 其他情况：源与目的均被污染：DST=<hex>;SRC=<hex>
            char buf[128];
            snprintf(buf, sizeof(buf), "DST=0x%lx;SRC=0x%lx", dstValue, srcValue);
            valueField = buf;
        } else if (dstTainted) {
            // 仅目的被污染：DST=<hex>
            char buf[64];
            snprintf(buf, sizeof(buf), "DST=0x%lx", dstValue);
            valueField = buf;
        } else if (valueSrcTainted) {
            // 仅源被污染：SRC=<hex>
            char buf[64];
            snprintf(buf, sizeof(buf), "SRC=0x%lx", srcValue);
            valueField = buf;
        }
    }
    }  // 闭合 if isImplicitMultiOpInstruction else 块

    if (TaintPropagation::isSetccInstruction(mnemonicText)) {
        valueField.clear();
        dstTainted = false;
    }
    
    bool isRegisterOnlyMov = isMovLikeInstruction && memReadTaint.empty();
    bool isPointerCarrierInstruction = isLeaInstruction || isRegisterOnlyMov;
    TaintSet pointerCarrierTaint = isLeaInstruction ? srcTaint : srcTaint;

    if (isPointerCarrierInstruction &&
        isSmallFieldLikeTaint(pointerCarrierTaint) &&
        isLikelyPointerValue(srcValue)) {
        suppressPointerLikeTaint = true;

        // 这类单/少数字段 taint 传播成高地址对象指针，属于地址噪声。
        // 直接切断本条指令的 taint 传播与记录，避免后续字段集合被错误并入。
        srcTaint0 = TaintSet();
        srcTaint1 = TaintSet();
        srcTaint = TaintSet();
        memReadTaint = TaintSet();
        valueSrcTainted = false;
        dstTainted = false;
        valueField.clear();
    }

    // 转换为旧格式 map（用于传递给 Logger）
    // Logger::logInstruction 会根据 _formatted 标记直接使用 valueField
    std::map<std::string, uint64_t> values;
    if (!valueField.empty()) {
        values["_formatted"] = 1;  // 标记已格式化，Logger 会特殊处理
    }

    // 污点传播规则
    if (TaintPropagation::isCmpInstruction(mnemonicText)) {
        // CMP/TEST: 不修改任何寄存器污点（CMP只读取，不写入）
        // 什么都不做
    } else if (TaintPropagation::isSetccInstruction(mnemonicText)) {
        // SETCC: 目的操作数被设置为0或1，清除其污点
        // 因为SETCC结果与输入数据包无关，只与条件码相关
        updateDstRegs(TaintSet());
    } else if (TaintPropagation::isLeaInstruction(mnemonicText)) {
        if (suppressPointerLikeTaint) {
            updateDstRegs(TaintSet());
        } else {
        TaintSet srcRegTaint;
        if (srcReg0_val != REG_INVALID()) {
            REG normalizedSrc0 = REG_FullRegName(srcReg0_val);
            if (normalizedSrc0 != REG_INVALID()) {
                srcRegTaint += state->getRegisterTaint(normalizedSrc0);
            }
        }
        if (srcReg1_val != REG_INVALID()) {
            REG normalizedSrc1 = REG_FullRegName(srcReg1_val);
            if (normalizedSrc1 != REG_INVALID()) {
                srcRegTaint += state->getRegisterTaint(normalizedSrc1);
            }
        }
        // 问题修复1：LEA仅在基址/索引被污染时才传播污点并记录
        if (!srcRegTaint.empty()) {
            TaintSet newTaint = TaintPropagation::propagateLEA(srcRegTaint, TaintSet());
            updateDstRegs(newTaint);
        } else {
            // 若源寄存器均不被污染，则清除目的寄存器污点（参考旧工具OpLeaNoReg行为）
            updateDstRegs(TaintSet());
        }
        }
    } else if (isStringStoreOp) {
        if (memOpCount > 0 && memOp0Write && memOp0 != 0) {
            updateMemoryWriteTaint(state, memOp0, memWriteSize,
                                   TaintPropagation::propagateStoreMemory(srcTaint));
        }
        if (memOpCount > 1 && memOp1Write && memOp1 != 0) {
            updateMemoryWriteTaint(state, memOp1, memWriteSize,
                                   TaintPropagation::propagateStoreMemory(srcTaint));
        }
    } else if (TaintPropagation::isMovInstruction(mnemonicText)) {
        TaintSet newTaint = memReadTaint.empty()
                                ? TaintPropagation::propagateMOV(srcTaint)
                                : TaintPropagation::propagateLoadMemory(memReadTaint);
        updateDstRegs(newTaint);

        // 内存写入污点更新
        if (memOpCount > 0 && memOp0Write && memOp0 != 0) {
            updateMemoryWriteTaint(state, memOp0, memWriteSize,
                                   TaintPropagation::propagateStoreMemory(srcTaint));
        }
        if (memOpCount > 1 && memOp1Write && memOp1 != 0) {
            updateMemoryWriteTaint(state, memOp1, memWriteSize,
                                   TaintPropagation::propagateStoreMemory(srcTaint));
        }
	    } else if (TaintPropagation::isBinaryArithmetic(mnemonicText)) {
	        // 特殊处理：xor reg, reg 或 sub reg, reg 清零操作（参考旧工具逻辑）
	        // 这类指令结果恒为 0，与原值无关，应清除污点
	        bool isSelfClearOp = (mnemonicText == "xor" || mnemonicText == "sub") &&
	                             (srcReg0_val != REG_INVALID() && srcReg1_val != REG_INVALID()) &&
	                             (REG_FullRegName(srcReg0_val) == REG_FullRegName(srcReg1_val));
	        
	        if (isSelfClearOp) {
	            // xor/sub 自身操作：结果恒为 0，清除污点
	            updateDstRegs(TaintSet());
	            if (memOpCount > 0 && memOp0Write && memOp0 != 0) {
	                updateMemoryWriteTaint(state, memOp0, memWriteSize,
	                                       TaintPropagation::propagateStoreMemory(TaintSet()));
	            }
	            if (memOpCount > 1 && memOp1Write && memOp1 != 0) {
	                updateMemoryWriteTaint(state, memOp1, memWriteSize,
	                                       TaintPropagation::propagateStoreMemory(TaintSet()));
	            }
	        } else {
	            // 正常二元运算：第一个操作数旧值与第二个操作数一起参与结果
	            TaintSet combined = srcTaint.unite(arithOp0Taint);
	            TaintSet newTaint = TaintPropagation::propagateBINARY(combined, TaintSet());
	            updateDstRegs(newTaint);
	            if (memOpCount > 0 && memOp0Write && memOp0 != 0) {
	                updateMemoryWriteTaint(state, memOp0, memWriteSize,
	                                       TaintPropagation::propagateStoreMemory(newTaint));
	            }
	            if (memOpCount > 1 && memOp1Write && memOp1 != 0) {
	                updateMemoryWriteTaint(state, memOp1, memWriteSize,
	                                       TaintPropagation::propagateStoreMemory(newTaint));
	            }
	        }
	    } else if (TaintPropagation::isShiftInstruction(mnemonicText)) {
        // shift指令只传播被移位值与移位量的污点，目的操作数可能是寄存器也可能是内存
        TaintSet combined = srcTaint.unite(arithOp0Taint);
        TaintSet newTaint = TaintPropagation::propagateSHIFT(combined);
        updateDstRegs(newTaint);
        if (memOpCount > 0 && memOp0Write && memOp0 != 0) {
            updateMemoryWriteTaint(state, memOp0, memWriteSize,
                                   TaintPropagation::propagateStoreMemory(newTaint));
        }
        if (memOpCount > 1 && memOp1Write && memOp1 != 0) {
            updateMemoryWriteTaint(state, memOp1, memWriteSize,
                                   TaintPropagation::propagateStoreMemory(newTaint));
        }
    } else if (TaintPropagation::isPopInstruction(mnemonicText)) {
        // pop reg: reg <- [rsp]
        // 污点来源是栈内存，不是寄存器的旧值（参考旧工具 ReadMempop）
        if (!memReadTaint.empty()) {
            // 内存被污染 → 传播到寄存器
            updateDstRegs(memReadTaint);
        } else {
            // 内存无污点 → 清除寄存器污点
            updateDstRegs(TaintSet());
        }
    } else if (TaintPropagation::isPushInstruction(mnemonicText)) {
        // push reg: [rsp] <- reg
        // 内存写入污点更新（参考旧工具 WriteMem）
        if (memOpCount > 0 && memOp0Write && memOp0 != 0) {
            updateMemoryWriteTaint(state, memOp0, memWriteSize,
                                   TaintPropagation::propagateStoreMemory(srcTaint));
        }
    }

    // ========== 问题2修复：污点分离记录 ==========
    // 计算用于记录的污点
    // 核心原则：DST记录目的操作数执行前的污染，SRC记录源操作数的污染（不包括目的操作数）
    TaintSet dstTaintForLog;
    TaintSet srcTaintForLog;
    
    if (TaintPropagation::isCmpInstruction(mnemonicText)) {
        // CMP/TEST 需要按前两个操作数的真实类型区分：
        // - cmp reg, mem: DST=第一个寄存器操作数, SRC=内存操作数
        // - cmp mem, reg/imm: DST=内存操作数, SRC=第二个寄存器操作数
        // - cmp reg, reg: DST=第一个寄存器, SRC=第二个寄存器
        dstTaintForLog = cmpOp0Taint;
        srcTaintForLog = cmpOp1Taint;
    } else if (TaintPropagation::isSetccInstruction(mnemonicText)) {
        // SETcc 结果为 0/1，不应继承旧目的寄存器或条件源操作数的字段污点。
        dstTaintForLog = TaintSet();
        srcTaintForLog = TaintSet();
    } else if (TaintPropagation::isBinaryArithmetic(mnemonicText) || TaintPropagation::isShiftInstruction(mnemonicText)) {
        // 二元操作和移位指令：DST是目的操作数执行前的污染，SRC是第二个源操作数（不含DST污染）
        // 例如 add eax, edx: DST是执行前的eax，SRC是edx（不包括eax的污染）
        // 例如 shl eax, 0x8: DST是执行前的eax，SRC是立即数（无污染）
        dstTaintForLog = arithOp0Taint;
        srcTaintForLog = arithOp1Taint;
    } else if (TaintPropagation::isPopInstruction(mnemonicText)) {
        // pop: DST 是目的寄存器（执行前的旧值不参与计算），SRC 是栈内存
        dstTaintForLog = TaintSet();
        srcTaintForLog = memReadTaint;
    } else if (TaintPropagation::isPushInstruction(mnemonicText)) {
        // push: DST 是栈内存（无需记录旧值），SRC 是源寄存器
        dstTaintForLog = TaintSet();
        srcTaintForLog = srcTaint;
    } else {
        // 其他指令：使用目的和源污点
        dstTaintForLog = dstOldTaint;
        srcTaintForLog = srcTaint;
    }

    // MOV/LEA 类指令：目的操作数旧值不参与计算，记录时不输出DST污点
    if (!TaintPropagation::isCmpInstruction(mnemonicText) && (isMovLikeInstruction || isLeaInstruction)) {
        dstTaintForLog = TaintSet();
    }

    // ========== 污染ID输出格式与值字段格式保持一致 ==========
    // 根据值字段格式，调整污染ID的输出格式
    // 例如：如果值字段是 DST=0x80，污染ID只输出DST部分
    //       如果值字段是 SRC=0x10，污染ID只输出SRC部分
    //       如果值字段是 DST=0x80;SRC=0x10，污染ID输出 DST;SRC
    if (!valueField.empty() && !TaintPropagation::isCmpInstruction(mnemonicText)) {
        bool hasDstValue = (valueField.find("DST=") != std::string::npos);
        bool hasSrcValue = (valueField.find("SRC=") != std::string::npos);
        
        if (hasDstValue && !hasSrcValue) {
            // 只有DST值，清空SRC污点
            srcTaintForLog = TaintSet();
        } else if (hasSrcValue && !hasDstValue) {
            // 只有SRC值，清空DST污点
            dstTaintForLog = TaintSet();
        }
        // 如果两者都有或都没有，保持原样
    }

    TaintSet logTaint = dstTaintForLog.unite(srcTaintForLog);

    // ========== 使用 PIN 的 isJumpInstruction 参数判断分支指令 ==========
    // isJumpInstruction 来自 PIN 的 INS_IsBranch(ins)，包含 jmp 和条件跳转
    bool isBranchInstruction = (isJumpInstruction != 0);
    
    // 区分无条件跳转（jmp）和条件跳转（jnz, jz, je, ...）
    bool isUnconditionalJump = (mnemonicText == "jmp" || mnemonicText == "jmpq");
    bool isConditionalJump = isBranchInstruction && !isUnconditionalJump;
    
    // 条件跳转指令不应有自己的污点（它们依赖 FLAGS，不直接读写通用寄存器）
    // 强制清空，让它们统一通过控制链追踪机制记录
    if (isConditionalJump) {
        logTaint = TaintSet();
    }
    
    // branchTaken 的含义：分支实际上跳转了（由PIN提供）
    // 对于非分支指令，branchTaken 无意义（传入0）
    bool actuallyTaken = (branchTaken != 0);
    
    // CMP/TEST: 不强制去除DST字段，值/污点保持一致
    bool isCmpOrTest = TaintPropagation::isCmpInstruction(mnemonicText);
    bool isSetccInstruction = TaintPropagation::isSetccInstruction(mnemonicText);
    
    // ========== 控制链追踪逻辑 ==========
    // 控制链：CMP/TEST(tainted) -> setcc -> 条件跳转
    // 当 CMP/TEST 指令的操作数被污染时，设置控制链标志
    // 使后续的 setcc 和条件跳转指令也能被记录
    if (isCmpOrTest) {
        if (!logTaint.empty()) {
            state->hasTaintedControlChain = true;
        } else {
            state->hasTaintedControlChain = false;
        }
    } else if (isSetccInstruction) {
        // setcc: 检查控制链标志，不修改它（继续传递给后续条件跳转）
        // setcc 本身清除目标污点（在传播规则中已处理），但控制链标志保持
    } else if (isConditionalJump) {
        // 条件跳转指令：检查控制链标志，不修改它（将在记录后清除）
    } else {
        // 其他指令：清除控制链标志
        state->hasTaintedControlChain = false;
    }
    
    
    // 问题1修复：LEA指令仅在源被污染时才记录
    if (isLeaInstruction && srcTaintForLog.empty()) {
        // LEA的源不被污染，不记录该指令
        return;
    }
    
    // 问题1修复（扩展）：MOV类指令仅在源被污染时才记录
    // 参考旧工具逻辑：如果只有目的被污染但源未被污染，则不记录（只清除污点）
    bool isMovLike = isMovLikeInstruction;
    if (isMovLike && srcTaintForLog.empty()) {
        // MOV的源不被污染，不记录该指令
        return;
    }
    
    // 仅记录有污点的指令（或关键指令）
    if (!config::RECORD_ONLY_TAINTED_INSTRUCTIONS) {
        // setcc 指令的特殊处理：如果控制链标志为 true，即使本身无污点也要记录
        // setcc 结果不携带污点ID（已被清除），但需要记录其存在
        if (isSetccInstruction && state->hasTaintedControlChain) {
            TaintSet emptyTaint;
            markCurrentBblTainted(tid, state);
            Logger::getInstance().logInstruction(tid, addr, disasmText, emptyTaint,
                                                values, Logger::BRANCH_NONE, Logger::LOOP_NONE, "",
                                                emptyTaint, emptyTaint);
            // 不清除控制链标志，让后续条件跳转也能被记录
            return;
        }
        
        // 条件跳转指令的特殊处理：如果控制链标志为 true，即使本身无污点也要记录
        if (isConditionalJump && state->hasTaintedControlChain && logTaint.empty()) {
            // 记录跳转指令（无污点，只有 TAKEN/NOT_TAKEN）
            Logger::BranchState branchState = actuallyTaken ? Logger::BRANCH_TAKEN : Logger::BRANCH_NOT_TAKEN;
            Logger::LoopType loopType = Logger::LOOP_NONE;
            if (isLoop) loopType = Logger::LOOP_TYPE;
            else if (isRepeat) loopType = Logger::REPEAT_TYPE;
            // 使用空污点集合记录
            TaintSet emptyTaint;
            markCurrentBblTainted(tid, state);
            Logger::getInstance().logInstruction(tid, addr, disasmText, emptyTaint,
                                                values, branchState, loopType, "",
                                                emptyTaint, emptyTaint);
            state->hasTaintedControlChain = false;  // 清除控制链标志
            return;
        }
        
        if (TaintPropagation::shouldRecordInstruction(logTaint, mnemonicText)) {
            // 对于分支指令，使用actuallyTaken确定分支状态
            Logger::BranchState branchState = Logger::BRANCH_NONE;
            if (isBranchInstruction && !logTaint.empty()) {
                branchState = actuallyTaken ? Logger::BRANCH_TAKEN : Logger::BRANCH_NOT_TAKEN;
            }
            // 确定循环类型
            Logger::LoopType loopType = Logger::LOOP_NONE;
            if (isLoop) {
                loopType = Logger::LOOP_TYPE;
            } else if (isRepeat) {
                loopType = Logger::REPEAT_TYPE;
            }
            markCurrentBblTainted(tid, state);
            Logger::getInstance().logInstruction(tid, addr, disasmText, logTaint,
                                                values, branchState, loopType, valueField,
                                                dstTaintForLog, srcTaintForLog);
            // 条件跳转指令记录后清除控制链标志
            if (isConditionalJump) {
                state->hasTaintedControlChain = false;
            }
        }
        return;
    }

    // setcc 指令的特殊处理（RECORD_ONLY_TAINTED_INSTRUCTIONS 模式）
    if (isSetccInstruction && state->hasTaintedControlChain) {
        TaintSet emptyTaint;
        markCurrentBblTainted(tid, state);
        Logger::getInstance().logInstruction(tid, addr, disasmText, emptyTaint,
                                            values, Logger::BRANCH_NONE, Logger::LOOP_NONE, "",
                                            emptyTaint, emptyTaint);
        // 不清除控制链标志
        return;
    }

    // 条件跳转指令的特殊处理（RECORD_ONLY_TAINTED_INSTRUCTIONS 模式）
    if (isConditionalJump && state->hasTaintedControlChain && logTaint.empty()) {
        Logger::BranchState branchState = actuallyTaken ? Logger::BRANCH_TAKEN : Logger::BRANCH_NOT_TAKEN;
        Logger::LoopType loopType = Logger::LOOP_NONE;
        if (isLoop) loopType = Logger::LOOP_TYPE;
        else if (isRepeat) loopType = Logger::REPEAT_TYPE;
        TaintSet emptyTaint;
        markCurrentBblTainted(tid, state);
        Logger::getInstance().logInstruction(tid, addr, disasmText, emptyTaint,
                                            values, branchState, loopType, "",
                                            emptyTaint, emptyTaint);
        state->hasTaintedControlChain = false;
        return;
    }

    if (!logTaint.empty()) {
        Logger::BranchState branchState = Logger::BRANCH_NONE;
        if (isBranchInstruction) {
            branchState = actuallyTaken ? Logger::BRANCH_TAKEN : Logger::BRANCH_NOT_TAKEN;
        }
        // 确定循环类型
        Logger::LoopType loopType = Logger::LOOP_NONE;
        if (isLoop) {
            loopType = Logger::LOOP_TYPE;
        } else if (isRepeat) {
            loopType = Logger::REPEAT_TYPE;
        }
        markCurrentBblTainted(tid, state);
        Logger::getInstance().logInstruction(tid, addr, disasmText, logTaint,
                                            values, branchState, loopType, valueField,
                                            dstTaintForLog, srcTaintForLog);
        // 条件跳转指令记录后清除控制链标志
        if (isConditionalJump) {
            state->hasTaintedControlChain = false;
        }
    }
}

// ============== 插桩注册函数 ==============

// 辅助函数：提取模块名称
static std::string getModuleName(const std::string& imgPath) {
    size_t lastSlash = imgPath.find_last_of("/\\");
    return (lastSlash != std::string::npos) ? imgPath.substr(lastSlash + 1) : imgPath;
}

static std::string canonicalRtnName(const std::string& rawName) {
    std::string name = rawName;

    size_t atPos = name.find('@');
    if (atPos != std::string::npos) {
        name = name.substr(0, atPos);
    }

    static const std::string giPrefix = "__GI_";
    if (name.compare(0, giPrefix.size(), giPrefix) == 0) {
        name = name.substr(giPrefix.size());
    }

    return name;
}

static bool isMemsetLikeRoutine(const std::string& matchName) {
    return matchName == "memset" ||
           matchName == "__memset" ||
           matchName == "__memset_chk";
}

static bool isBzeroLikeRoutine(const std::string& matchName) {
    return matchName == "bzero" ||
           matchName == "__bzero" ||
           matchName == "explicit_bzero";
}

static bool isMemcpyLikeRoutine(const std::string& matchName) {
    return matchName == "memcpy" ||
           matchName == "__memcpy";
}

static bool isMemmoveLikeRoutine(const std::string& matchName) {
    return matchName == "memmove" ||
           matchName == "__memmove";
}

static void rememberRoutineByName(IMG img, const char* name, std::set<ADDRINT>& targets) {
    RTN rtn = RTN_FindByName(img, name);
    if (RTN_Valid(rtn)) {
        targets.insert(RTN_Address(rtn));
    }
}

static std::string routineNameAtAddress(ADDRINT addr) {
    std::string name = RTN_FindNameByAddress(addr);
    if (name.empty()) {
        RTN rtn = RTN_FindByAddress(addr);
        if (RTN_Valid(rtn)) {
            name = RTN_Name(rtn);
        }
    }
    return canonicalRtnName(name);
}

// 辅助函数：C++ 符号名反修饰
static std::string demangleSymbol(const std::string& name) {
    if (name.empty()) {
        return "unknown";
    }

    int status = 0;
    char* demangled = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string result(demangled);
        std::free(demangled);
        return result;
    }

    if (demangled) {
        std::free(demangled);
    }
    return name;
}

// 辅助函数：过滤噪声函数（std/asio 等运行时与模板基础设施）
static bool shouldSkipFunctionByName(const std::string& funcName) {
    if (config::ENABLE_FUNCTION_WHITELIST) {
        for (const char** p = config::FUNCTION_WHITELIST_PREFIXES; *p != nullptr; ++p) {
            const std::string prefix(*p);
            if (funcName.compare(0, prefix.size(), prefix) == 0) {
                return false;  // 命中白名单，保留
            }
        }
        return true;  // 白名单模式下，未命中则过滤
    }

    static const char* kNoisePrefixes[] = {
        "std::",
        "asio::",
        "__gnu_cxx::",
        nullptr
    };

    for (const char** p = kNoisePrefixes; *p != nullptr; ++p) {
        const std::string prefix(*p);
        if (funcName.compare(0, prefix.size(), prefix) == 0) {
            return true;
        }
    }

    static const char* kNoiseContains[] = {
        "typeinfo for std::",
        "vtable for std::",
        "typeinfo for asio::",
        "vtable for asio::",
        nullptr
    };

    for (const char** p = kNoiseContains; *p != nullptr; ++p) {
        if (funcName.find(*p) != std::string::npos) {
            return true;
        }
    }

    return false;
}

// 辅助函数：检查模块是否应该被插桩（黑名单检查）
static bool shouldInstrumentModule(const std::string& imgPath) {
    if (!config::ENABLE_MODULE_FILTERING) {
        return true;  // 如果未启用过滤，则插桩所有模块
    }

    std::string moduleName = getModuleName(imgPath);

    // 检查黑名单
    for (const char** blacklist = config::MODULE_BLACKLIST;
         *blacklist != nullptr; blacklist++) {
        if (moduleName.find(*blacklist) != std::string::npos) {
            if (config::DEBUG_MODE) {
                fprintf(stderr, "[FILTER] Skipping blacklisted module: %s\n", 
                        moduleName.c_str());
            }
            return false;
        }
    }

    return true;
}

// 辅助函数：指令级过滤（参考旧工具的 filter_ins）
// 过滤控制流指令和不太重要的指令
static bool shouldFilterInstruction(INS ins) {
    // LEA 属于重要计算指令，不应被过滤
    if (INS_Opcode(ins) == XED_ICLASS_LEA) {
        return false;
    }
    std::string mnemonicText = INS_Mnemonic(ins);
    std::transform(mnemonicText.begin(), mnemonicText.end(), mnemonicText.begin(), ::tolower);
    if (isStringStoreInstruction(mnemonicText)) {
        return false;
    }
    // 允许分支指令通过（不过滤）
    // 过滤 call 和 ret 指令
    if (INS_IsCall(ins) || INS_IsRet(ins)) {
        return true;
    }

    // 过滤 NOP 指令
    if (INS_IsNop(ins)) {
        return true;
    }

    // 过滤操作数过少或过多的指令
    UINT32 opcount = INS_OperandCount(ins);
    if (opcount <= 1 || opcount >= 5) {
        return true;
    }

    // 过滤扩展指令（指令长度过长）
    if (INS_Size(ins) > 40) {
        return true;
    }

    return false;
}

VOID instrumentImage(IMG img, VOID* v) {
    if (!IMG_Valid(img)) {
        return;
    }

    std::string imgName = IMG_Name(img);
    std::string moduleName = getModuleName(imgName);
    
    // 检查是否是 libc（用于网络函数hook）
    bool isLibc = (moduleName.find("libc") != std::string::npos);
    
    // 模块黑名单检查（除了libc中可能的网络函数）
    if (!isLibc && !shouldInstrumentModule(imgName)) {
        return;
    }

    if (config::DEBUG_MODE && !isLibc) {
        fprintf(stderr, "[INSTRUMENT] Instrumenting module: %s\n", moduleName.c_str());
    }

    if (!isLibc) {
        rememberRoutineByName(img, "memset", g_memsetPltTargets);
        rememberRoutineByName(img, "memset@plt", g_memsetPltTargets);
        rememberRoutineByName(img, "__memset", g_memsetPltTargets);
        rememberRoutineByName(img, "__memset@plt", g_memsetPltTargets);
        rememberRoutineByName(img, "bzero", g_bzeroPltTargets);
        rememberRoutineByName(img, "bzero@plt", g_bzeroPltTargets);
        rememberRoutineByName(img, "__bzero", g_bzeroPltTargets);
        rememberRoutineByName(img, "__bzero@plt", g_bzeroPltTargets);
        rememberRoutineByName(img, "memcpy", g_memcpyPltTargets);
        rememberRoutineByName(img, "memcpy@plt", g_memcpyPltTargets);
        rememberRoutineByName(img, "__memcpy", g_memcpyPltTargets);
        rememberRoutineByName(img, "__memcpy@plt", g_memcpyPltTargets);
        rememberRoutineByName(img, "memmove", g_memmovePltTargets);
        rememberRoutineByName(img, "memmove@plt", g_memmovePltTargets);
        rememberRoutineByName(img, "__memmove", g_memmovePltTargets);
        rememberRoutineByName(img, "__memmove@plt", g_memmovePltTargets);
    }

    for (SEC sec = IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec)) {
        for (RTN rtn = SEC_RtnHead(sec); RTN_Valid(rtn); rtn = RTN_Next(rtn)) {
            std::string rtnName = RTN_Name(rtn);
            std::string matchName = canonicalRtnName(rtnName);
            bool isMemsetLike = isMemsetLikeRoutine(matchName);
            bool isBzeroLike = isBzeroLikeRoutine(matchName);
            bool isMemcpyLike = isMemcpyLikeRoutine(matchName);
            bool isMemmoveLike = isMemmoveLikeRoutine(matchName);

            // 过滤 PLT 跳板函数
            std::string secName = SEC_Name(RTN_Sec(rtn));
            bool isPltSection = (secName == ".plt" || secName == ".plt.sec");
            bool shouldHookPltClearRoutine = (!isLibc && isPltSection && (isMemsetLike || isBzeroLike));
            bool shouldHookPltCopyRoutine = (!isLibc && isPltSection && (isMemcpyLike || isMemmoveLike));

            if (rtnName == ".plt" || rtnName == ".plt.sec" ||
                (isPltSection && !shouldHookPltClearRoutine && !shouldHookPltCopyRoutine)) {
                continue;
            }

            if (shouldHookPltClearRoutine && isMemsetLike) {
                g_memsetPltTargets.insert(RTN_Address(rtn));
                continue;
            }

            if (shouldHookPltClearRoutine && isBzeroLike) {
                g_bzeroPltTargets.insert(RTN_Address(rtn));
                continue;
            }

            if (shouldHookPltCopyRoutine && isMemcpyLike) {
                g_memcpyPltTargets.insert(RTN_Address(rtn));
                continue;
            }

            if (shouldHookPltCopyRoutine && isMemmoveLike) {
                g_memmovePltTargets.insert(RTN_Address(rtn));
                continue;
            }

            // 如果是 libc，仅处理网络函数和系统函数
            if (isLibc) {
                bool isNetworkRoutine =
                    (matchName == "recv" || matchName == "__recv" || matchName == "__recv_chk" || matchName == "__libc_recv" ||
                     matchName == "recvfrom" || matchName == "__recvfrom_chk" || matchName == "__libc_recvfrom" ||
                     matchName == "recvmsg" || matchName == "__recvmsg" || matchName == "__recvmsg_chk" || matchName == "__libc_recvmsg");
                bool isGettimeofdayRoutine =
                    (matchName == "gettimeofday" || matchName == "__gettimeofday");
                bool isClockGettimeRoutine =
                    (matchName == "clock_gettime" || matchName == "__clock_gettime");
                bool isLocaltimeRoutine =
                    (matchName == "localtime" || matchName == "localtime64" ||
                     matchName == "__localtime64");
                bool isLocaltimeRRoutine =
                    (matchName == "localtime_r" || matchName == "localtime64_r" ||
                     matchName == "__localtime_r" || matchName == "__localtime64_r");
                bool isTimeRoutine =
                    isGettimeofdayRoutine || isClockGettimeRoutine ||
                    isLocaltimeRoutine || isLocaltimeRRoutine;

                if (isNetworkRoutine &&
                    !g_hookedNetworkRoutineAddrs.insert(RTN_Address(rtn)).second) {
                    continue;
                }
                if (isTimeRoutine &&
                    !g_hookedTimeRoutineAddrs.insert(RTN_Address(rtn)).second) {
                    continue;
                }

                RTN_Open(rtn);
                
                if (isNetworkRoutine) {

                    // recv / __recv: int recv(int fd, void* buf, size_t len, int flags)
                    if (matchName == "recv" || matchName == "__recv" || matchName == "__libc_recv") {
                        RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)recvCallback,
                                      IARG_THREAD_ID,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 2,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 3,
                                      IARG_FUNCRET_EXITPOINT_VALUE,
                                      IARG_END);
                    }
                    // __recv_chk: int __recv_chk(int fd, void* buf, size_t len, size_t buflen, int flags)
                    else if (matchName == "__recv_chk") {
                        RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)recvCallback,
                                      IARG_THREAD_ID,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 2,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 4,
                                      IARG_FUNCRET_EXITPOINT_VALUE,
                                      IARG_END);
                    }
                    // recvfrom: int recvfrom(int fd, void* buf, size_t len, int flags, ...)
                    else if (matchName == "recvfrom" || matchName == "__libc_recvfrom") {
                        RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)recvfromCallback,
                                      IARG_THREAD_ID,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 2,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 3,
                                      IARG_FUNCRET_EXITPOINT_VALUE,
                                      IARG_END);
                    }
                    // __recvfrom_chk: int __recvfrom_chk(int fd, void* buf, size_t len, size_t buflen, int flags, ...)
                    else if (matchName == "__recvfrom_chk") {
                        RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)recvfromCallback,
                                      IARG_THREAD_ID,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 2,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 4,
                                      IARG_FUNCRET_EXITPOINT_VALUE,
                                      IARG_END);
                    }
                    // recvmsg / __recvmsg
                    else if (matchName == "recvmsg" || matchName == "__recvmsg" || matchName == "__libc_recvmsg") {
                        RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)recvmsgCallback,
                                      IARG_THREAD_ID,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 2,
                                      IARG_FUNCRET_EXITPOINT_VALUE,
                                      IARG_END);
                    }
                    // __recvmsg_chk: int __recvmsg_chk(int fd, struct msghdr* msg, size_t buflen, int flags)
                    else if (matchName == "__recvmsg_chk") {
                        RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)recvmsgCallback,
                                      IARG_THREAD_ID,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                      IARG_FUNCARG_ENTRYPOINT_VALUE, 3,
                                      IARG_FUNCRET_EXITPOINT_VALUE,
                                      IARG_END);
                    }
                } else if (isGettimeofdayRoutine) {
                    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)gettimeofdayCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                  IARG_FUNCRET_EXITPOINT_VALUE,
                                  IARG_END);
                } else if (isClockGettimeRoutine) {
                    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)clockGettimeCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                  IARG_FUNCRET_EXITPOINT_VALUE,
                                  IARG_END);
                } else if (isLocaltimeRoutine) {
                    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)localtimeCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCRET_EXITPOINT_VALUE,
                                  IARG_END);
                } else if (isLocaltimeRRoutine) {
                    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)localtimeRCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                  IARG_FUNCRET_EXITPOINT_VALUE,
                                  IARG_END);
                } else if (matchName == "memcpy" || matchName == "__memcpy") {
                    // memcpy hook - BEFORE: 检查参数污点
                    RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)memcpyBeforeCallback,
                                  IARG_THREAD_ID,
                                  IARG_ADDRINT, "memcpy",
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 2,
                                  IARG_END);
                    // memcpy hook - AFTER: 污点传播
                    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)memcpyCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 2,
                                  IARG_FUNCRET_EXITPOINT_VALUE,
                                  IARG_END);
                } else if (matchName == "memmove" || matchName == "__memmove") {
                    // memmove hook - BEFORE: 检查参数污点
                    RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)memcpyBeforeCallback,
                                  IARG_THREAD_ID,
                                  IARG_ADDRINT, "memmove",
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 2,
                                  IARG_END);
                    // memmove hook - AFTER: 污点传播 (和 memcpy 相同)
                    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)memcpyCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 2,
                                  IARG_FUNCRET_EXITPOINT_VALUE,
                                  IARG_END);
                } else if (matchName == "malloc") {
                    // malloc hook - BEFORE: 检查参数污点
                    RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)mallocBeforeCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                  IARG_END);
                    // malloc hook - AFTER
                    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)mallocCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                  IARG_FUNCRET_EXITPOINT_VALUE,
                                  IARG_END);
                } else if (matchName == "calloc") {
                    // calloc hook - BEFORE: 检查参数污点
                    RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)callocBeforeCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                  IARG_END);
                    // calloc hook - AFTER
                    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)callocCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                  IARG_FUNCRET_EXITPOINT_VALUE,
                                  IARG_END);
                } else if (matchName == "realloc") {
                    // realloc hook - BEFORE: 检查参数污点
                    RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)reallocBeforeCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                  IARG_END);
                    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)reallocCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                  IARG_FUNCRET_EXITPOINT_VALUE,
                                  IARG_END);
                } else if (matchName == "free") {
                    // free hook - BEFORE: 检查参数污点
                    RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)freeBeforeCallback,
                                  IARG_THREAD_ID,
                                  IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                                  IARG_END);
                }
                
                RTN_Close(rtn);
                continue;
            }

            // 跳过某些函数
            bool skip = false;
            for (const char** blacklist = config::BLACKLIST_FUNCTIONS;
                 *blacklist != nullptr; blacklist++) {
                if (rtnName.find(*blacklist) != std::string::npos) {
                    skip = true;
                    break;
                }
            }

            if (skip) {
                continue;
            }

            if (matchName == "InitializeENIPMessage") {
                RTN_Open(rtn);
                RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)initializeENIPMessageBeforeCallback,
                              IARG_THREAD_ID,
                              IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                              IARG_END);
                RTN_Close(rtn);
                continue;
            }

            if (matchName == "InitializeMessageRouterResponse") {
                RTN_Open(rtn);
                RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)initializeMessageRouterResponseBeforeCallback,
                              IARG_THREAD_ID,
                              IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                              IARG_END);
                RTN_Close(rtn);
                continue;
            }

            // 反修饰 C++ 函数名
            std::string readableName = demangleSymbol(rtnName);
            if (shouldSkipFunctionByName(readableName)) {
                continue;
            }

            RTN_Open(rtn);

            // 函数入口
            RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)onFunctionEntry,
                          IARG_THREAD_ID, IARG_PTR,
                          new std::string(readableName), IARG_PTR,
                          new Address(Address::fromAbsolute(RTN_Address(rtn), img)),
                          IARG_END);

            // 函数出口
            RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)onFunctionExit,
                          IARG_THREAD_ID, IARG_PTR,
                          new std::string(readableName), IARG_PTR,
                          new Address(Address::fromAbsolute(RTN_Address(rtn), img)),
                          IARG_END);

            RTN_Close(rtn);
        }
    }
}

VOID instrumentTrace(TRACE trace, VOID* v) {
    // 模块级过滤：检查 trace 所属的模块
    RTN rtn = TRACE_Rtn(trace);
    if (RTN_Valid(rtn)) {
        SEC sec = RTN_Sec(rtn);
        if (SEC_Valid(sec)) {
            IMG img = SEC_Img(sec);
            if (IMG_Valid(img)) {
                std::string imgName = IMG_Name(img);
                std::string moduleName = getModuleName(imgName);
                // 对 ld-linux 进行严格过滤
                if (moduleName.find("ld-linux") != std::string::npos) {
                    return;  // 完全跳过 ld-linux
                }
                // 检查其他黑名单模块
                if (!shouldInstrumentModule(imgName)) {
                    return;  // 跳过系统库的 trace
                }
            }
        }
    }

    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl)) {
        uint64_t bblAddr = BBL_Address(bbl);
        uint32_t bblSize = BBL_Size(bbl);

        BBL_InsertCall(bbl, IPOINT_BEFORE, (AFUNPTR)onBasicBlockEntry,
                      IARG_THREAD_ID, IARG_ADDRINT, bblAddr, IARG_UINT32,
                      bblSize, IARG_CONTEXT, IARG_END);
        
        // 问题修复：在BBL执行完后弹出栈，保持循环检测的准确性
        // 注意：ret/jmp等指令对IPOINT_AFTER无效，需查找最后一条有效指令
        INS lastValidIns = BBL_InsTail(bbl);
        if (INS_Valid(lastValidIns) && INS_IsValidForIpointAfter(lastValidIns)) {
            // 如果最后一条指令有效，则在其后插入exit回调
            INS_InsertCall(lastValidIns, IPOINT_AFTER, (AFUNPTR)onBasicBlockExit,
                          IARG_THREAD_ID, IARG_END);
        } else {
            // 如果最后一条指令无效（ret/jmp等），在倒数第二条指令后插入
            // 这样可以保证BBL逻辑完成后立即弹栈
            INS prevIns = INS_Prev(lastValidIns);
            if (INS_Valid(prevIns) && INS_IsValidForIpointAfter(prevIns)) {
                INS_InsertCall(prevIns, IPOINT_AFTER, (AFUNPTR)onBasicBlockExit,
                              IARG_THREAD_ID, IARG_END);
            }
        }
    }
}

VOID instrumentInstruction(INS ins, VOID* v) {
    if (!config::RECORD_INSTRUCTION) {
        return;
    }
    // ===== 模块级过滤 =====
    // 彻底排除 ld-linux 指令，避免记录系统加载器的代码
    IMG img = IMG_FindByAddress(INS_Address(ins));
    if (IMG_Valid(img)) {
        std::string moduleName = getModuleName(IMG_Name(img));
        if (moduleName.find("ld-linux") != std::string::npos) {
            return;  // 完全跳过 ld-linux 模块的所有指令
        }
        // 检查其他黑名单模块（但保留 libc，因为网络函数在那）
        if (!shouldInstrumentModule(IMG_Name(img))) {
            return;
        }
    }

    const bool isCall = INS_IsCall(ins);
    const bool isDirectCf = INS_IsDirectControlFlow(ins);
    ADDRINT directTarget = isDirectCf ? INS_DirectControlFlowTargetAddress(ins) : 0;

    if (isCall && isDirectCf) {
        ADDRINT target = directTarget;
        std::string targetName;
        bool targetIsMemset = (g_memsetPltTargets.find(target) != g_memsetPltTargets.end());
        bool targetIsBzero = (g_bzeroPltTargets.find(target) != g_bzeroPltTargets.end());
        bool targetIsMemcpy = (g_memcpyPltTargets.find(target) != g_memcpyPltTargets.end());
        bool targetIsMemmove = (g_memmovePltTargets.find(target) != g_memmovePltTargets.end());

        if (!targetIsMemset && !targetIsBzero && !targetIsMemcpy && !targetIsMemmove) {
            targetName = routineNameAtAddress(target);
            targetIsMemset = isMemsetLikeRoutine(targetName);
            targetIsBzero = isBzeroLikeRoutine(targetName);
            targetIsMemcpy = isMemcpyLikeRoutine(targetName);
            targetIsMemmove = isMemmoveLikeRoutine(targetName);
        }

        /*
         * Some PIE binaries expose PLT entries only as ".plt.sec" to Pin instead
         * of "memcpy@plt"/"memmove@plt".  For these unresolved direct PLT calls,
         * record a conservative copy-like evidence line only when the second
         * argument points to tainted source memory.  Do not propagate taint here:
         * the target might be a non-copy libc function with a similar ABI shape.
         */
        const bool targetIsOpaquePlt =
            (!targetIsMemset && !targetIsBzero && !targetIsMemcpy && !targetIsMemmove &&
             targetName == ".plt.sec");

        if (targetIsMemset) {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)memsetBeforeCallback,
                          IARG_THREAD_ID,
                          IARG_REG_VALUE, REG_RDI,
                          IARG_REG_VALUE, REG_RSI,
                          IARG_REG_VALUE, REG_RDX,
                          IARG_END);
            return;
        }
        if (targetIsBzero) {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)bzeroBeforeCallback,
                          IARG_THREAD_ID,
                          IARG_REG_VALUE, REG_RDI,
                          IARG_REG_VALUE, REG_RSI,
                          IARG_END);
            return;
        }
        if (targetIsMemcpy) {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)memcpyBeforeCallback,
                          IARG_THREAD_ID,
                          IARG_ADDRINT, "memcpy",
                          IARG_REG_VALUE, REG_RDI,
                          IARG_REG_VALUE, REG_RSI,
                          IARG_REG_VALUE, REG_RDX,
                          IARG_END);
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)memcpyCallback,
                          IARG_THREAD_ID,
                          IARG_REG_VALUE, REG_RDI,
                          IARG_REG_VALUE, REG_RSI,
                          IARG_REG_VALUE, REG_RDX,
                          IARG_ADDRINT, 0,
                          IARG_END);
            return;
        }
        if (targetIsMemmove) {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)memcpyBeforeCallback,
                          IARG_THREAD_ID,
                          IARG_ADDRINT, "memmove",
                          IARG_REG_VALUE, REG_RDI,
                          IARG_REG_VALUE, REG_RSI,
                          IARG_REG_VALUE, REG_RDX,
                          IARG_END);
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)memcpyCallback,
                          IARG_THREAD_ID,
                          IARG_REG_VALUE, REG_RDI,
                          IARG_REG_VALUE, REG_RSI,
                          IARG_REG_VALUE, REG_RDX,
                          IARG_ADDRINT, 0,
                          IARG_END);
            return;
        }
        if (targetIsOpaquePlt) {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)pltCopyLikeBeforeCallback,
                          IARG_THREAD_ID,
                          IARG_REG_VALUE, REG_RSI,
                          IARG_REG_VALUE, REG_RDX,
                          IARG_END);
        }
    }

    // 指令级过滤：跳过控制流指令和不重要的指令
    if (shouldFilterInstruction(ins)) {
        return;
    }

    // 获取指令所在的 BBL 地址
    uint64_t bblAddr = 0;
    RTN rtn = INS_Rtn(ins);
    if (RTN_Valid(rtn)) {
        // 获取所在 BBL 的起始地址（简化处理，使用指令地址）
        bblAddr = INS_Address(ins);
    }

    // 插桩所有指令，传递内存操作数地址与读写标志
    UINT32 opcount = INS_OperandCount(ins);
    UINT32 memOpCount = INS_MemoryOperandCount(ins);
    UINT32 memOp0Read = (memOpCount > 0 && INS_MemoryOperandIsRead(ins, 0)) ? 1 : 0;
    UINT32 memOp0Write = (memOpCount > 0 && INS_MemoryOperandIsWritten(ins, 0)) ? 1 : 0;
    UINT32 memOp1Read = (memOpCount > 1 && INS_MemoryOperandIsRead(ins, 1)) ? 1 : 0;
    UINT32 memOp1Write = (memOpCount > 1 && INS_MemoryOperandIsWritten(ins, 1)) ? 1 : 0;
    UINT32 op0IsMem = (opcount > 0 && INS_OperandIsMemory(ins, 0)) ? 1 : 0;
    UINT32 op1IsMem = (opcount > 1 && INS_OperandIsMemory(ins, 1)) ? 1 : 0;
    UINT32 op0IsReg = (opcount > 0 && INS_OperandIsReg(ins, 0)) ? 1 : 0;
    UINT32 op1IsReg = (opcount > 1 && INS_OperandIsReg(ins, 1)) ? 1 : 0;

    // 收集寄存器操作数（最多 2 个源寄存器与 2 个目的寄存器）
    UINT32 srcRegCount = 0;
    UINT32 dstRegCount = 0;
    REG srcRegs[2] = {REG_INVALID(), REG_INVALID()};
    REG dstRegs[2] = {REG_INVALID(), REG_INVALID()};

    // 问题2修复：特别处理LEA指令
    // LEA的源操作数在内存操作数中（基址和索引寄存器），需要特殊提取
    std::string mnemonic = INS_Mnemonic(ins);
    std::transform(mnemonic.begin(), mnemonic.end(), mnemonic.begin(), ::tolower);
    bool isLeaInstruction = (mnemonic == "lea");
    bool isStringStore = isStringStoreInstruction(mnemonic);

    for (UINT32 i = 0; i < opcount; ++i) {
        // 常规寄存器操作数
        if (INS_OperandIsReg(ins, i)) {
            REG reg = INS_OperandReg(ins, i);
            // LEA指令：跳过常规寄存器操作数的源收集（目的寄存器会被覆盖，其旧值不是源）
            // LEA的真正源是内存操作数中的基址/索引寄存器，在下面的else if分支处理
            if (!isLeaInstruction && INS_OperandRead(ins, i) && srcRegCount < 2) {
                srcRegs[srcRegCount++] = reg;
            }
            // 目的寄存器正常收集
            if (INS_OperandWritten(ins, i) && dstRegCount < 2) {
                dstRegs[dstRegCount++] = reg;
            }
        }
        // LEA指令：从内存操作数中提取基址和索引寄存器（这是LEA的真正源）
        else if (isLeaInstruction && INS_OperandIsMemory(ins, i)) {
            // 提取基址寄存器（如 [rbp+rdx*1+0xa] 中的 rbp）
            REG baseReg = INS_OperandMemoryBaseReg(ins, i);
            if (REG_valid(baseReg) && srcRegCount < 2) {
                srcRegs[srcRegCount++] = baseReg;
            }
            // 提取索引寄存器（如 [rbp+rdx*1+0xa] 中的 rdx）
            REG indexReg = INS_OperandMemoryIndexReg(ins, i);
            if (REG_valid(indexReg) && srcRegCount < 2) {
                srcRegs[srcRegCount++] = indexReg;
            }
        }
    }

    // LEA补充：直接从内存操作数获取基址/索引，避免操作数计数差异导致缺失
    if (isLeaInstruction) {
        auto addSrcRegIfMissing = [&](REG reg) {
            if (!REG_valid(reg) || srcRegCount >= 2) {
                return;
            }
            for (UINT32 i = 0; i < srcRegCount; ++i) {
                if (srcRegs[i] == reg) {
                    return;
                }
            }
            srcRegs[srcRegCount++] = reg;
        };

        REG baseReg = INS_MemoryBaseReg(ins);
        REG indexReg = INS_MemoryIndexReg(ins);
        addSrcRegIfMissing(baseReg);
        addSrcRegIfMissing(indexReg);
    }

    if (isStringStore) {
        REG accReg = getStringStoreAccumulatorReg(mnemonic);
        if (REG_valid(accReg) && srcRegCount < 2) {
            srcRegs[srcRegCount++] = accReg;
        }
    }

    // 问题1修复：计算是否为跳转指令（在插桩时完成）
    // 使用PIN API检测跳转类指令
    UINT32 isJumpFlag = INS_IsBranch(ins) ? 1 : 0;
    
    // 获取直接跳转的目标地址（用于替换反汇编中的绝对地址为 module+offset）
    ADDRINT jumpTargetAddr = 0;
    if (INS_IsDirectControlFlow(ins)) {
        jumpTargetAddr = INS_DirectControlFlowTargetAddress(ins);
    }

    if (memOpCount >= 2) {
        UINT32 readSize = 0;
        UINT32 writeSize = 0;
        if (memOp0Read) readSize = INS_MemoryOperandSize(ins, 0);
        else if (memOp1Read) readSize = INS_MemoryOperandSize(ins, 1);
        if (memOp0Write) writeSize = INS_MemoryOperandSize(ins, 0);
        else if (memOp1Write) writeSize = INS_MemoryOperandSize(ins, 1);
        
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)(void*)onInstructionExecute,
                      IARG_THREAD_ID, IARG_INST_PTR,
                      IARG_PTR, new std::string(INS_Disassemble(ins)),
                      IARG_PTR, new std::string(INS_Mnemonic(ins)),
                      IARG_CONST_CONTEXT,
                      IARG_ADDRINT, bblAddr,
                      IARG_UINT32, memOpCount,
                      IARG_MEMORYOP_EA, 0,
                      IARG_MEMORYOP_EA, 1,
                      IARG_UINT32, memOp0Read,
                      IARG_UINT32, memOp0Write,
                      IARG_UINT32, memOp1Read,
                      IARG_UINT32, memOp1Write,
                      IARG_UINT32, op0IsMem,
                      IARG_UINT32, op1IsMem,
                      IARG_UINT32, op0IsReg,
                      IARG_UINT32, op1IsReg,
                      IARG_UINT32, readSize,
                      IARG_UINT32, writeSize,
                      IARG_ADDRINT, srcRegs[0],      // 直接传递 REG
                      IARG_ADDRINT, srcRegs[1],      // 直接传递 REG
                      IARG_ADDRINT, dstRegs[0],      // 直接传递 REG
                      IARG_ADDRINT, dstRegs[1],      // 直接传递 REG
                      IARG_BRANCH_TAKEN,             // 问题2修复：获取分支实际状态
                      IARG_UINT32, isJumpFlag,       // 问题1修复
                      IARG_ADDRINT, jumpTargetAddr,  // 跳转目标地址
                      IARG_END);
    } else if (memOpCount == 1) {
        UINT32 readSize = memOp0Read ? INS_MemoryOperandSize(ins, 0) : 0;
        UINT32 writeSize = memOp0Write ? INS_MemoryOperandSize(ins, 0) : 0;
        
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)(void*)onInstructionExecute,
                      IARG_THREAD_ID, IARG_INST_PTR,
                      IARG_PTR, new std::string(INS_Disassemble(ins)),
                      IARG_PTR, new std::string(INS_Mnemonic(ins)),
                      IARG_CONST_CONTEXT,
                      IARG_ADDRINT, bblAddr,
                      IARG_UINT32, memOpCount,
                      IARG_MEMORYOP_EA, 0,
                      IARG_ADDRINT, 0,
                      IARG_UINT32, memOp0Read,
                      IARG_UINT32, memOp0Write,
                      IARG_UINT32, 0,
                      IARG_UINT32, 0,
                      IARG_UINT32, op0IsMem,
                      IARG_UINT32, op1IsMem,
                      IARG_UINT32, op0IsReg,
                      IARG_UINT32, op1IsReg,
                      IARG_UINT32, readSize,
                      IARG_UINT32, writeSize,
                      IARG_ADDRINT, srcRegs[0],      // 直接传递 REG
                      IARG_ADDRINT, srcRegs[1],      // 直接传递 REG
                      IARG_ADDRINT, dstRegs[0],      // 直接传递 REG
                      IARG_ADDRINT, dstRegs[1],      // 直接传递 REG
                      IARG_BRANCH_TAKEN,             // 问题2修复：获取分支实际状态
                      IARG_UINT32, isJumpFlag,       // 问题1修复
                      IARG_ADDRINT, jumpTargetAddr,  // 跳转目标地址
                      IARG_END);
    } else {
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)(void*)onInstructionExecute,
                      IARG_THREAD_ID, IARG_INST_PTR,
                      IARG_PTR, new std::string(INS_Disassemble(ins)),
                      IARG_PTR, new std::string(INS_Mnemonic(ins)),
                      IARG_CONST_CONTEXT,
                      IARG_ADDRINT, bblAddr,
                      IARG_UINT32, 0,
                      IARG_ADDRINT, 0,
                      IARG_ADDRINT, 0,
                      IARG_UINT32, 0,
                      IARG_UINT32, 0,
                      IARG_UINT32, 0,
                      IARG_UINT32, 0,
                      IARG_UINT32, op0IsMem,
                      IARG_UINT32, op1IsMem,
                      IARG_UINT32, op0IsReg,
                      IARG_UINT32, op1IsReg,
                      IARG_UINT32, 0,
                      IARG_UINT32, 0,
                      IARG_ADDRINT, srcRegs[0],      // 直接传递 REG
                      IARG_ADDRINT, srcRegs[1],      // 直接传递 REG
                      IARG_ADDRINT, dstRegs[0],      // 直接传递 REG
                      IARG_ADDRINT, dstRegs[1],      // 直接传递 REG
                      IARG_BRANCH_TAKEN,             // 问题2修复：获取分支实际状态
                      IARG_UINT32, isJumpFlag,       // 问题1修复
                      IARG_ADDRINT, jumpTargetAddr,  // 跳转目标地址
                      IARG_END);
    }
    
    // ===== 分支检测已集成到onInstructionExecute中 =====
    // 不再需要单独的IPOINT_TAKEN_BRANCH插桩
}
