#include "taintset.h"

// 转换为输出格式：直接输出污点索引（已经是相对偏移）
std::string TaintSet::toString() const {
    if (indices.empty()) {
        return "";
    }

    std::ostringstream oss;
    bool first = true;
    for (uint64_t idx : indices) {
        if (!first) {
            oss << ",";
        }
        oss << idx;
        first = false;
    }
    return oss.str();
}
