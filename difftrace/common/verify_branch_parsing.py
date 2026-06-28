#!/usr/bin/env python3
"""
直接检查日志文件中的分支信息 - 用于验证解析是否正确
"""

import sys
import re


def parse_and_display_branches(log_path: str, max_show: int = 50):
    """解析并显示日志文件中的所有分支信息"""
    branches = []
    line_num = 0
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line_num += 1
            line = raw_line.rstrip("\n")
            
            if not line:
                continue
            
            if line.startswith("Instruction\t"):
                # 提取分支信息
                # ⚠️ 重要：必须先检查 NOT_TAKEN，因为 NOT_TAKEN 也以 TAKEN 结尾
                if line.endswith("NOT_TAKEN"):
                    m = re.match(r"^Instruction\t([^:]+):\s+(.*)$", line)
                    if m:
                        addr = m.group(1).strip()
                        rest = m.group(2)
                        outcome = "NOT_TAKEN"
                        instr = rest[:-10].strip()  # 去掉 "\tNOT_TAKEN"
                        
                        branches.append({
                            'line': line_num,
                            'addr': addr,
                            'instruction': instr,
                            'outcome': outcome,
                            'raw': line
                        })
                elif line.endswith("TAKEN"):
                    m = re.match(r"^Instruction\t([^:]+):\s+(.*)$", line)
                    if m:
                        addr = m.group(1).strip()
                        rest = m.group(2)
                        outcome = "TAKEN"
                        instr = rest[:-6].strip()  # 去掉 "\tTAKEN"
                        
                        branches.append({
                            'line': line_num,
                            'addr': addr,
                            'instruction': instr,
                            'outcome': outcome,
                            'raw': line
                        })
    
    print("=" * 100)
    print(f"日志文件分支信息解析验证: {log_path}")
    print("=" * 100)
    print()
    print(f"共找到 {len(branches)} 个分支指令")
    print()
    
    if branches:
        print("-" * 100)
        print(f"{'行号':<8} {'地址':<20} {'指令':<40} {'结果':<12}")
        print("-" * 100)
        
        for i, br in enumerate(branches):
            if i < max_show or max_show == 0:
                print(f"{br['line']:<8} {br['addr']:<20} {br['instruction']:<40} {br['outcome']:<12}")
        
        if max_show > 0 and len(branches) > max_show:
            print(f"\n... 还有 {len(branches) - max_show} 个分支未显示")
            print(f"    使用参数 'all' 查看全部")
        print()
    
    # 统计
    taken_count = sum(1 for br in branches if br['outcome'] == 'TAKEN')
    not_taken_count = sum(1 for br in branches if br['outcome'] == 'NOT_TAKEN')
    
    print("-" * 100)
    print("统计:")
    print(f"  TAKEN: {taken_count}")
    print(f"  NOT_TAKEN: {not_taken_count}")
    print()
    
    # 检查是否有重复地址
    addr_outcomes = {}
    for br in branches:
        addr = br['addr']
        if addr not in addr_outcomes:
            addr_outcomes[addr] = []
        addr_outcomes[addr].append((br['line'], br['outcome']))
    
    # 找出有不同结果的地址
    conflicting = {}
    for addr, outcomes in addr_outcomes.items():
        unique_outcomes = set(o[1] for o in outcomes)
        if len(unique_outcomes) > 1:
            conflicting[addr] = outcomes
    
    if conflicting:
        print("-" * 100)
        print("⚠️  发现同一地址有不同的分支结果:")
        print("-" * 100)
        for addr, outcomes in sorted(conflicting.items()):
            print(f"\n  地址: {addr}")
            for line_num, outcome in outcomes:
                print(f"    行 {line_num}: {outcome}")
    
    return branches


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_branch_parsing.py <log_file_path> [max_show]")
        print()
        print("参数:")
        print("  log_file_path: 日志文件路径")
        print("  max_show: 显示的分支数量，使用 'all' 显示全部，默认 50")
        print()
        print("示例:")
        print("  python verify_branch_parsing.py out/mutations/5/001_zero.log")
        print("  python verify_branch_parsing.py out/mutations/5/001_zero.log all")
        print("  python verify_branch_parsing.py out/mutations/5/001_zero.log 100")
        sys.exit(1)
    
    log_path = sys.argv[1]
    
    max_show = 50
    if len(sys.argv) > 2:
        if sys.argv[2].lower() == 'all':
            max_show = 0
        else:
            max_show = int(sys.argv[2])
    
    parse_and_display_branches(log_path, max_show)


if __name__ == "__main__":
    main()
