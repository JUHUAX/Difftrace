#ifndef LOOPDETECTOR_H
#define LOOPDETECTOR_H

#include "pin.H"
#include <map>
#include <vector>
#include <set>

// 循环检测器（每线程独立）
// 参考旧工具loop.cpp的实现，使用RSP跟踪函数调用边界
// 关键：当函数返回时（RSP增大），清理对应的历史记录，避免函数复用误判为循环
class LoopDetector {
private:
    // BBL 条目：记录地址、大小、栈帧
    struct BBLEntry {
        uint64_t addr;
        uint32_t size;
    };

    // 函数上下文：记录某次函数调用时的所有BBL历史
    struct FunctionContext {
        uint64_t rsp;              // 进入此函数时的RSP值
        std::vector<BBLEntry> bbls; // 该函数内执行过的BBL
    };

    // 每线程的函数上下文栈（按RSP降序排列，栈顶是最内层函数）
    std::map<THREADID, std::vector<FunctionContext>> contextStacks;
    mutable PIN_LOCK stackLock;

public:
    LoopDetector() {
        PIN_InitLock(&stackLock);
    }

    ~LoopDetector() {
        // PIN_LOCK doesn't require explicit cleanup
    }

    // 清理已返回函数的上下文（参考旧工具BlockTrace::push中的清理逻辑）
    // 当RSP增大时，说明有函数返回，需要清理这些函数的历史
    void cleanupReturnedFunctions(THREADID tid, uint64_t currentRsp) {
        // 假设已持有锁
        if (contextStacks.find(tid) == contextStacks.end()) {
            return;
        }
        
        auto& contexts = contextStacks[tid];
        // 旧工具逻辑: while (!functions.empty() && functions.back().first < rsp)
        // RSP变大意味着函数返回，清理那些RSP更小的上下文
        while (!contexts.empty() && contexts.back().rsp < currentRsp) {
            contexts.pop_back();
        }
    }

    // 检查是否在循环中，并进入BBL
    // 返回值：
    // 0 - 不在循环中
    // 1 - 在真实循环中（LOOP）
    int enterAndCheckLoop(THREADID tid, uint64_t bblAddr, uint32_t bblSize, uint64_t rsp) {
        PIN_GetLock(&stackLock, PIN_ThreadId());

        // 初始化线程的上下文栈
        if (contextStacks.find(tid) == contextStacks.end()) {
            contextStacks[tid] = std::vector<FunctionContext>();
        }

        // 第一步：清理已返回的函数上下文（RSP增大意味着函数返回）
        cleanupReturnedFunctions(tid, rsp);

        auto& contexts = contextStacks[tid];
        
        // 第二步：找到或创建当前RSP对应的函数上下文
        FunctionContext* currentContext = nullptr;
        for (auto& ctx : contexts) {
            if (ctx.rsp == rsp) {
                currentContext = &ctx;
                break;
            }
        }
        
        // 如果没有找到相同RSP的上下文，创建新的（新的函数调用）
        if (currentContext == nullptr) {
            FunctionContext newCtx;
            newCtx.rsp = rsp;
            contexts.push_back(newCtx);
            currentContext = &contexts.back();
        }

        // 第三步：在当前函数上下文中检查是否是循环
        // 参考旧工具Block::valid: 检查BBL起始或结束地址是否与历史匹配
        // 这样可以处理PIN因不同入口点创建不同BBL但实际是同一循环的情况
        int loopType = 0;
        uint64_t bblEnd = bblAddr + bblSize;  // 当前BBL结束地址
        for (const auto& bbl : currentContext->bbls) {
            uint64_t histEnd = bbl.addr + bbl.size;  // 历史BBL结束地址
            // 检查：起始地址相等 或 结束地址相等
            if (bbl.addr == bblAddr || histEnd == bblEnd) {
                // 同一函数上下文中，同一BBL（或同一循环的不同入口）被再次执行 → 循环
                loopType = 1;
                break;
            }
        }

        // 第四步：将当前BBL加入历史
        BBLEntry entry;
        entry.addr = bblAddr;
        entry.size = bblSize;
        currentContext->bbls.push_back(entry);

        PIN_ReleaseLock(&stackLock);

        return loopType;
    }

    // 兼容旧接口
    void enterBBL(THREADID tid, uint64_t bblAddr, uint32_t bblSize, uint64_t rsp, int callDepth) {
        // 调用新接口，忽略返回值
        (void)enterAndCheckLoop(tid, bblAddr, bblSize, rsp);
        (void)callDepth;
    }

    // 兼容旧接口
    int getLoopType(THREADID tid, uint64_t bblAddr, uint32_t bblSize, uint64_t currentRsp, int currentCallDepth) {
        // 注意：这个接口不应该单独使用，因为检测和进入应该是原子的
        // 返回0表示不在循环中（实际检测在enterAndCheckLoop中完成）
        (void)tid; (void)bblAddr; (void)bblSize; (void)currentRsp; (void)currentCallDepth;
        return 0;
    }

    // 旧接口兼容（不带size参数）
    int getLoopType(THREADID tid, uint64_t bblAddr, uint64_t currentRsp, int currentCallDepth) {
        return getLoopType(tid, bblAddr, 0, currentRsp, currentCallDepth);
    }

    // 简化接口
    bool isInLoop(THREADID tid, uint64_t bblAddr, uint64_t currentRsp, int currentCallDepth) {
        return getLoopType(tid, bblAddr, currentRsp, currentCallDepth) > 0;
    }

    // 函数入口/出口钩子（保留接口）
    void onFunctionEnter(THREADID tid, int callDepth) {
        (void)tid; (void)callDepth;
    }

    void onFunctionExit(THREADID tid, int callDepth) {
        (void)tid; (void)callDepth;
    }

    // 退出基本块
    uint64_t exitBBL(THREADID tid) {
        (void)tid;
        return 0;
    }

    // 获取当前栈深度
    size_t getStackDepth(THREADID tid) const {
        PIN_GetLock(&stackLock, PIN_ThreadId());

        size_t depth = 0;
        if (contextStacks.find(tid) != contextStacks.end()) {
            for (const auto& ctx : contextStacks.at(tid)) {
                depth += ctx.bbls.size();
            }
        }

        PIN_ReleaseLock(&stackLock);

        return depth;
    }

    // 清空某线程的栈
    void clearStack(THREADID tid) {
        PIN_GetLock(&stackLock, PIN_ThreadId());

        if (contextStacks.find(tid) != contextStacks.end()) {
            contextStacks[tid].clear();
        }

        PIN_ReleaseLock(&stackLock);
    }

    // 清空所有线程的栈
    void clearAllStacks() {
        PIN_GetLock(&stackLock, PIN_ThreadId());

        contextStacks.clear();

        PIN_ReleaseLock(&stackLock);
    }
};

#endif  // LOOPDETECTOR_H
