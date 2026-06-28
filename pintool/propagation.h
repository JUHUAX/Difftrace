#ifndef PROPAGATION_H
#define PROPAGATION_H

#include "taintset.h"
#include "pin.H"
#include <string>

// 污点传播规则集合
class TaintPropagation {
public:
    // ============== 寄存器规范化 ==============
    // 将寄存器规范化为完整的 64 位寄存器名
    static REG normalizeRegister(REG reg) {
        return REG_FullRegName(reg);
    }

    // ============== 污点传播规则 ==============

    // MOV 指令：DST = SRC
    static TaintSet propagateMOV(const TaintSet& src) {
        return src;
    }

    // MOVZX/MOVSX 指令：零/符号扩展，污点不变
    static TaintSet propagateEXTEND(const TaintSet& src) {
        return src;
    }

    // 二元运算：ADD/SUB/XOR/AND/OR/IMUL
    // DST = SRC1 ∪ SRC2
    static TaintSet propagateBINARY(const TaintSet& src1, const TaintSet& src2) {
        return src1.unite(src2);
    }

    // 移位操作：SHL/SHR/SAR/ROL/ROR
    // 污点仅来自被移位的值，移位量不影响污点
    static TaintSet propagateSHIFT(const TaintSet& src) {
        return src;
    }

    // LEA（加载有效地址）
    // 仅传播来自基址和索引寄存器的污点，不包括立即数偏移
    static TaintSet propagateLEA(const TaintSet& baseRegTaint,
                                const TaintSet& indexRegTaint) {
        return baseRegTaint.unite(indexRegTaint);
    }

    // CMP/TEST 指令
    // 不修改目的地，仅影响 FLAGS
    // 返回值应该赋给 FLAGS
    static TaintSet propagateCMP(const TaintSet& src1, const TaintSet& src2) {
        return src1.unite(src2);
    }

    // SETcc 指令（SETNZ, SETE 等）
    // DST = 条件污点（来自最近的CMP/TEST）
    // 注意：不产生新污点ID，直接继承FLAGS污点
    static TaintSet propagateSETCC(const TaintSet& flagsTaint) {
        return flagsTaint;
    }

    // ============== 内存访问规则 ==============

    // 读取内存：DST_REG = MEM[addr]
    // 污点应该来自内存中该地址的污点数据
    static TaintSet propagateLoadMemory(const TaintSet& memTaint) {
        return memTaint;
    }

    // 写入内存：MEM[addr] = SRC_REG
    // 内存的污点状态更新为 SRC_REG 的污点
    static TaintSet propagateStoreMemory(const TaintSet& srcTaint) {
        return srcTaint;
    }

    // ============== 函数参数规则 ==============

    // 函数调用时的污点继承
    // 通常参数寄存器（RDI, RSI, RDX, RCX, R8, R9）继承污点
    static bool isFunctionParameterRegister(REG reg) {
        REG normalized = normalizeRegister(reg);
        return normalized == REG_RDI || normalized == REG_RSI ||
               normalized == REG_RDX || normalized == REG_RCX ||
               normalized == REG_R8 || normalized == REG_R9;
    }

    // ============== 污点决策辅助函数 ==============

    // 判断是否应该记录该指令（有污点或是关键指令）
    static bool shouldRecordInstruction(const TaintSet& taint,
                                        const std::string& mnemonic) {
        // 总是记录有污点的指令
        if (!taint.empty()) {
            return true;
        }

        // 记录某些关键的无污点指令（如 cmp, jcc）
        // 这些指令可能在控制流中很重要
        return mnemonic == "cmp" || mnemonic == "test" ||
               mnemonic == "jz" || mnemonic == "jnz" ||
               mnemonic == "je" || mnemonic == "jne" ||
               mnemonic == "jle" || mnemonic == "jge" ||
               mnemonic == "jl" || mnemonic == "jg" ||
               mnemonic == "ja" || mnemonic == "jb" ||
               mnemonic == "setnz" || mnemonic == "sete" ||
               mnemonic == "setle" || mnemonic == "setge";
    }

    // ============== 指令分类 ==============

    // 判断是否为 CMP 类指令
    static bool isCmpInstruction(const std::string& mnemonic) {
        return mnemonic == "cmp" || mnemonic == "test" ||
               mnemonic == "cmpsq" || mnemonic == "cmpsd" ||
               mnemonic == "cmps" || mnemonic == "scas";
    }

    // 判断是否为 SETcc 类指令
    static bool isSetccInstruction(const std::string& mnemonic) {
        return mnemonic.substr(0, 3) == "set";  // SETE, SETNZ, SETLE 等
    }

    // 判断是否为条件跳转指令
    static bool isConditionalJump(const std::string& mnemonic) {
        return mnemonic.substr(0, 1) == "j" && mnemonic != "jmp" &&
               mnemonic != "jmpq";
    }

    // 判断是否为 MOV 类指令
    static bool isMovInstruction(const std::string& mnemonic) {
        return mnemonic == "mov" || mnemonic == "movq" ||
               mnemonic == "movl" || mnemonic == "movw" ||
               mnemonic == "movb" || mnemonic == "movzx" ||
               mnemonic == "movsx" || mnemonic == "movsxd";
    }

    // 判断是否为二元算术指令
    static bool isBinaryArithmetic(const std::string& mnemonic) {
        return mnemonic == "add" || mnemonic == "sub" ||
               mnemonic == "xor" || mnemonic == "and" ||
               mnemonic == "or" || mnemonic == "imul" ||
               mnemonic == "mul" || mnemonic == "div" ||
               mnemonic == "idiv";
    }

    // 判断是否为移位指令
    static bool isShiftInstruction(const std::string& mnemonic) {
        return mnemonic == "shl" || mnemonic == "shr" ||
               mnemonic == "sar" || mnemonic == "sal" ||
               mnemonic == "rol" || mnemonic == "ror";
    }

    // 判断是否为 LEA 指令
    static bool isLeaInstruction(const std::string& mnemonic) {
        return mnemonic == "lea" || mnemonic == "leaq" ||
               mnemonic == "leal" || mnemonic == "leaw";
    }

    // 判断是否为 POP 指令
    static bool isPopInstruction(const std::string& mnemonic) {
        return mnemonic == "pop" || mnemonic == "popq" ||
               mnemonic == "popl" || mnemonic == "popw";
    }

    // 判断是否为 PUSH 指令
    static bool isPushInstruction(const std::string& mnemonic) {
        return mnemonic == "push" || mnemonic == "pushq" ||
               mnemonic == "pushl" || mnemonic == "pushw";
    }
};

#endif  // PROPAGATION_H
