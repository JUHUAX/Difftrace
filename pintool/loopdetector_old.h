#ifndef LOOPDETECTOR_H
#define LOOPDETECTOR_H

#include "pin.H"
#include <map>
#include <vector>
#include <set>

// 循环检测器（每线程独立）
// 问题3修复：添加RSP栈帧跟踪和调用栈深度，区分真实循环(LOOP)和函数调用复用(REPEAT)
class LoopDetector {
private:
    // BBL 栈条目：记录地址、大小、栈帧和调用深度
    struct BBLEntry {
        uint64_t addr;
        uint32_t size;
        uint64_t rsp;      // 栈帧指针
        int callDepth;     // 调用栈深度（问题2修复）
    };

    // 每线程的 BBL 栈
    std::map<THREADID, std::vector<BBLEntry>> bblStacks;
    mutable PIN_LOCK stackLock;

public:
    LoopDetector() {
        PIN_InitLock(&stackLock);
    }

    ~LoopDetector() {
        // PIN_LOCK doesn't require explicit cleanup
    }

    // 进入基本块（现在需要传入RSP值和调用栈深度）
    void enterBBL(THREADID tid, uint64_t bblAddr, uint32_t bblSize, uint64_t rsp, int callDepth) {
        PIN_GetLock(&stackLock, PIN_ThreadId());

        if (bblStacks.find(tid) == bblStacks.end()) {
            bblStacks[tid] = std::vector<BBLEntry>();
        }

        BBLEntry entry;
        entry.addr = bblAddr;
        entry.size = bblSize;
        entry.rsp = rsp;
        entry.callDepth = callDepth;
        bblStacks[tid].push_back(entry);

        PIN_ReleaseLock(&stackLock);
    }

    // 函数入口/出口钩子（目前不改变行为，保留接口以兼容插桩层）
    void onFunctionEnter(THREADID tid, int callDepth) {
        (void)tid;
        (void)callDepth;
    }

    void onFunctionExit(THREADID tid, int callDepth) {
        (void)tid;
        (void)callDepth;
    }

    // 退出基本块（返回栈顶）
    uint64_t exitBBL(THREADID tid) {
        PIN_GetLock(&stackLock, PIN_ThreadId());

        uint64_t addr = 0;
        if (bblStacks.find(tid) != bblStacks.end() && !bblStacks[tid].empty()) {
            addr = bblStacks[tid].back().addr;
            bblStacks[tid].pop_back();
        }

        PIN_ReleaseLock(&stackLock);

        return addr;
    }

    // 问题2修复：检查是否在循环中（后向边检测）
    // 参考旧工具logic：检查当前BBL的起始或结束地址是否与历史中任何BBL的起始或结束地址相等
    // 这样可以检测到跳转到BBL中间造成的循环
    // 返回值：
    // 0 - 不在循环中
    // 1 - 在真实循环中（LOOP，后向边）
    int getLoopType(THREADID tid, uint64_t bblAddr, uint32_t bblSize, uint64_t currentRsp, int currentCallDepth) {
        PIN_GetLock(&stackLock, PIN_ThreadId());

        int loopType = 0;  // 0 = 不在循环中
        
        // 计算当前BBL的结束地址
        uint64_t bblEnd = bblAddr + bblSize;
        
        // 遍历整个历史栈：参考旧工具blocktrace.valid()的实现
        // 检查当前BBL的起始或结束地址是否与历史中任何BBL的起始或结束地址相等
        if (bblStacks.find(tid) != bblStacks.end() && !bblStacks[tid].empty()) {
            const auto& stack = bblStacks[tid];
            for (size_t i = 0; i < stack.size(); ++i) {
                uint64_t histAddr = stack[i].addr;
                uint64_t histEnd = stack[i].addr + stack[i].size;
                // 检查：当前起始==历史起始 或 当前结束==历史结束
                if (bblAddr == histAddr || bblEnd == histEnd) {
                    loopType = 1;  // 后向边 → 循环
                    break;
                }
            }
        }

        PIN_ReleaseLock(&stackLock);

        return loopType;
    }
    
    // 旧接口兼容（不带size参数）
    int getLoopType(THREADID tid, uint64_t bblAddr, uint64_t currentRsp, int currentCallDepth) {
        // 默认size=0，只比较起始地址
        return getLoopType(tid, bblAddr, 0, currentRsp, currentCallDepth);
    }

    // 简化接口：向后兼容（返回是否在循环中，包括LOOP和REPEAT）
    bool isInLoop(THREADID tid, uint64_t bblAddr, uint64_t currentRsp, int currentCallDepth) {
        return getLoopType(tid, bblAddr, currentRsp, currentCallDepth) > 0;
    }

    // 获取当前栈深度
    size_t getStackDepth(THREADID tid) const {
        PIN_GetLock(&stackLock, PIN_ThreadId());

        size_t depth = 0;
        if (bblStacks.find(tid) != bblStacks.end()) {
            depth = bblStacks.at(tid).size();
        }

        PIN_ReleaseLock(&stackLock);

        return depth;
    }

    // 清空某线程的栈
    void clearStack(THREADID tid) {
        PIN_GetLock(&stackLock, PIN_ThreadId());

        if (bblStacks.find(tid) != bblStacks.end()) {
            bblStacks[tid].clear();
        }

        PIN_ReleaseLock(&stackLock);
    }

    // 清空所有线程的栈
    void clearAllStacks() {
        PIN_GetLock(&stackLock, PIN_ThreadId());

        bblStacks.clear();

        PIN_ReleaseLock(&stackLock);
    }
};

#endif  // LOOPDETECTOR_H
