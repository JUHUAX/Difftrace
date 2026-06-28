#include "moduleinfo.h"

// 静态成员定义
std::map<ADDRINT, ModuleInfoManager::ModuleRange> ModuleInfoManager::moduleMap;
PIN_LOCK ModuleInfoManager::mapLock;
