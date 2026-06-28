#ifndef TAINTSTATE_H
#define TAINTSTATE_H

#include "taintset.h"
#include "pin.H"
#include <map>
#include <vector>
#include <set>

// 每线程的污点状态容器
struct TaintState {
    // ============== 寄存器污点 ==============
    // 规范化的寄存器 → 污点集合
    std::map<REG, TaintSet> registerTaint;

    // ============== 内存污点 ==============
    // 内存地址 → 污点集合
    std::map<uint64_t, TaintSet> memoryTaint;

    // ============== 循环检测 =============="
    // 基本块栈（用于循环检测）
    std::vector<uint64_t> bblStack;

    // 已记录的基本块集合（去重）
    std::set<uint64_t> recordedBBLs;

    // 已记录的循环指令集合（去重）
    std::set<uint64_t> recordedLoops;

    // ============== BBL 延迟输出（只输出被污染的基本块） ==============
    uint64_t currentBblAddr = 0;        // 当前 BBL 的原始地址
    uint32_t currentBblSize = 0;        // 当前 BBL 的大小
    bool     currentBblTainted = false;  // 当前 BBL 内是否有被记录的污点指令
    bool     currentBblLogged = false;   // 当前 BBL 是否已输出日志（防止重复输出）

    // ============== 控制链追踪 ==============
    // 当 CMP/TEST 指令操作数被污染时设置为 true
    // 用于记录紧随其后的条件跳转指令
    bool hasTaintedControlChain = false;

    // ============== 构造与销毁 ==============
    TaintState() {}

    ~TaintState() {}

    // ============== 辅助函数 ==============

    // 检查寄存器是否被污染
    bool isRegisterTainted(REG reg) const {
        REG normalizedReg = REG_FullRegName(reg);
        return registerTaint.count(normalizedReg) > 0 &&
               !registerTaint.at(normalizedReg).empty();
    }

    // 检查内存地址是否被污染
    bool isMemoryTainted(uint64_t addr) const {
        return memoryTaint.count(addr) > 0 && !memoryTaint.at(addr).empty();
    }

    // 获取寄存器污点
    TaintSet getRegisterTaint(REG reg) const {
        REG normalizedReg = REG_FullRegName(reg);
        if (registerTaint.count(normalizedReg) > 0) {
            return registerTaint.at(normalizedReg);
        }
        return TaintSet();
    }

    // 获取内存污点
    TaintSet getMemoryTaint(uint64_t addr) const {
        if (memoryTaint.count(addr) > 0) {
            return memoryTaint.at(addr);
        }
        return TaintSet();
    }

    // 设置寄存器污点
    void setRegisterTaint(REG reg, const TaintSet& taint) {
        REG normalizedReg = REG_FullRegName(reg);
        registerTaint[normalizedReg] = taint;
    }

    // 设置内存污点（可选范围）
    void setMemoryTaint(uint64_t addr, const TaintSet& taint) {
        memoryTaint[addr] = taint;
    }

    // 清除寄存器污点
    void clearRegisterTaint(REG reg) {
        REG normalizedReg = REG_FullRegName(reg);
        registerTaint.erase(normalizedReg);
    }

    // 清除内存污点
    void clearMemoryTaint(uint64_t addr) {
        memoryTaint.erase(addr);
    }

    // 清空所有污点
    void clearAll() {
        registerTaint.clear();
        memoryTaint.clear();
    }

    // 清空循环检测栈
    void clearBBLStack() {
        bblStack.clear();
    }

    // 循环检测：进入基本块
    void enterBBL(uint64_t bblAddr) {
        bblStack.push_back(bblAddr);
    }

    // 循环检测：退出基本块
    void exitBBL() {
        if (!bblStack.empty()) {
            bblStack.pop_back();
        }
    }

    // 检查是否在循环中（基本块是否已在栈中）
    bool isInLoop(uint64_t bblAddr) const {
        for (uint64_t addr : bblStack) {
            if (addr == bblAddr) {
                return true;
            }
        }
        return false;
    }
};

#endif  // TAINTSTATE_H
