#ifndef ADDRESS_H
#define ADDRESS_H

#include "pin.H"
#include <string>
#include <sstream>
#include <iomanip>

// 表示 module+offset 格式的地址
struct Address {
    std::string module;
    uint64_t offset;

    Address() : module(""), offset(0) {}

    Address(const std::string& m, uint64_t o) : module(m), offset(o) {}

    // 从绝对地址转换为 module+offset 格式（使用 IMG）
    static Address fromAbsolute(uint64_t absAddr, IMG img) {
        if (!IMG_Valid(img)) {
            return Address("unknown", absAddr);
        }

        std::string imgName = IMG_Name(img);
        // 提取文件名（去除路径）
        size_t pos = imgName.find_last_of("/\\");
        if (pos != std::string::npos) {
            imgName = imgName.substr(pos + 1);
        }
        // 去除 .so 后缀
        if (imgName.size() > 3 && imgName.substr(imgName.size() - 3) == ".so") {
            imgName = imgName.substr(0, imgName.size() - 3);
        }

        uint64_t loadOffset = IMG_LoadOffset(img);
        uint64_t moduleOffset = absAddr - loadOffset;

        return Address(imgName, moduleOffset);
    }

    // 从绝对地址转换为 module+offset（使用模块名和基址）
    static Address fromAbsolute(uint64_t absAddr, const std::string& moduleName, uint64_t baseAddr) {
        std::string name = moduleName;
        
        // 去除 .so 后缀
        if (name.size() > 3 && name.substr(name.size() - 3) == ".so") {
            name = name.substr(0, name.size() - 3);
        }
        
        uint64_t offset = absAddr - baseAddr;
        return Address(name, offset);
    }

    // 转换为字符串格式：module+0xoffset
    std::string toString() const {
        std::ostringstream oss;
        oss << module << "+0x" << std::hex << offset;
        return oss.str();
    }

    // 转换为紧凑格式（用于比较）
    std::string toCompactString() const {
        std::ostringstream oss;
        oss << module << "+" << std::hex << offset;
        return oss.str();
    }

    bool operator<(const Address& other) const {
        if (module != other.module) {
            return module < other.module;
        }
        return offset < other.offset;
    }

    bool operator==(const Address& other) const {
        return module == other.module && offset == other.offset;
    }

    bool isValid() const {
        return !module.empty() && module != "unknown";
    }
};

#endif  // ADDRESS_H
