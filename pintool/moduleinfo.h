#ifndef MODULEINFO_H
#define MODULEINFO_H

#include "pin.H"
#include "address.h"
#include <map>
#include <string>

// 模块信息管理器
// 在插桩阶段构建模块映射表，运行时无锁查询
class ModuleInfoManager {
private:
    struct ModuleRange {
        ADDRINT lowAddr;
        ADDRINT highAddr;
        std::string name;
        
        ModuleRange() : lowAddr(0), highAddr(0), name("unknown") {}
        ModuleRange(ADDRINT low, ADDRINT high, const std::string& n)
            : lowAddr(low), highAddr(high), name(n) {}
    };
    
    // 模块范围映射表（按起始地址排序）
    static std::map<ADDRINT, ModuleRange> moduleMap;
    static PIN_LOCK mapLock;
    
public:
    // 初始化
    static void init() {
        PIN_InitLock(&mapLock);
    }
    
    // 在插桩阶段添加模块（线程安全）
    static void addModule(IMG img) {
        if (!IMG_Valid(img)) {
            return;
        }
        
        ADDRINT low = IMG_LowAddress(img);
        ADDRINT high = IMG_HighAddress(img);
        std::string name = IMG_Name(img);
        
        // 提取模块名（去除路径）
        size_t pos = name.find_last_of("/\\");
        if (pos != std::string::npos) {
            name = name.substr(pos + 1);
        }
        
        PIN_GetLock(&mapLock, PIN_ThreadId());
        moduleMap[low] = ModuleRange(low, high, name);
        PIN_ReleaseLock(&mapLock);
    }
    
    // 运行时查询地址所属模块（无锁，读操作线程安全）
    static Address getAddress(ADDRINT addr) {
        // 使用 lower_bound 查找最接近的模块
        auto it = moduleMap.upper_bound(addr);
        
        if (it != moduleMap.begin()) {
            --it;
            const ModuleRange& range = it->second;
            
            // 检查地址是否在该模块范围内
            if (addr >= range.lowAddr && addr < range.highAddr) {
                return Address::fromAbsolute(addr, range.name, range.lowAddr);
            }
        }
        
        // 未找到模块，使用原始地址
        return Address("unknown", addr);
    }
    
    // 获取函数地址（带模块信息）
    static Address getFunctionAddress(RTN rtn) {
        ADDRINT addr = RTN_Address(rtn);
        return getAddress(addr);
    }
    
    // 获取基本块地址（带模块信息）
    static Address getBBLAddress(BBL bbl) {
        ADDRINT addr = BBL_Address(bbl);
        return getAddress(addr);
    }
    
    // 获取指令地址（带模块信息）
    static Address getInstructionAddress(INS ins) {
        ADDRINT addr = INS_Address(ins);
        return getAddress(addr);
    }
    
    // 清理
    static void fini() {
        PIN_GetLock(&mapLock, PIN_ThreadId());
        moduleMap.clear();
        PIN_ReleaseLock(&mapLock);
    }
};

#endif // MODULEINFO_H
