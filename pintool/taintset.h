#ifndef TAINTSET_H
#define TAINTSET_H

#include <set>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

// 污点集合：维护被污染的字节索引集合
class TaintSet {
private:
    std::set<uint64_t> indices;  // 有序污点索引集合

public:
    TaintSet() {}

    // 添加单个污点索引
    void addIndex(uint64_t idx) {
        indices.insert(idx);
    }

    // 添加一个范围的污点索引 [start, end)
    void addRange(uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i++) {
            indices.insert(i);
        }
    }

    // 从另一个 TaintSet 合并污点
    void merge(const TaintSet& other) {
        indices.insert(other.indices.begin(), other.indices.end());
    }

    // 返回合并后的新 TaintSet
    TaintSet unite(const TaintSet& other) const {
        TaintSet result = *this;
        result.merge(other);
        return result;
    }

    // 清空污点集合
    void clear() {
        indices.clear();
    }

    // 检查是否为空
    bool empty() const {
        return indices.empty();
    }

    // 获取污点数量
    size_t size() const {
        return indices.size();
    }

    // 检查是否包含某个索引
    bool contains(uint64_t idx) const {
        return indices.count(idx) > 0;
    }

    // 获取内部集合（用于高级操作）
    const std::set<uint64_t>& getIndices() const {
        return indices;
    }

    // 转换为输出格式：污点索引列表
    std::string toString() const;

    // 转换为 hex 格式列表（可选）
    std::string toHexString() const {
        if (indices.empty()) {
            return "";
        }

        std::ostringstream oss;
        bool first = true;
        for (uint64_t idx : indices) {
            if (!first) {
                oss << ",";
            }
            oss << "0x" << std::hex << idx;
            first = false;
        }
        return oss.str();
    }

    // 获取第一个污点索引
    uint64_t getFirst() const {
        if (indices.empty()) {
            return 0;
        }
        return *indices.begin();
    }

    // 获取最后一个污点索引
    uint64_t getLast() const {
        if (indices.empty()) {
            return 0;
        }
        return *indices.rbegin();
    }

    // 运算符重载
    TaintSet& operator+=(const TaintSet& other) {
        merge(other);
        return *this;
    }

    bool operator==(const TaintSet& other) const {
        return indices == other.indices;
    }

    bool operator!=(const TaintSet& other) const {
        return !(*this == other);
    }
};

#endif  // TAINTSET_H
