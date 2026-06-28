#!/usr/bin/env python3
"""
对比所有 mutations 的 bb_multiset_l1 指标
"""

import json
from tabulate import tabulate


def main():
    report_path = "./out/report.json"
    with open(report_path, "r") as f:
        report = json.load(f)
    
    field = report["fields"][0]
    baseline_run = field["runs"][0]
    
    print("=" * 100)
    print("所有 Mutations 的 bb_multiset_l1 指标对比")
    print("=" * 100)
    print()
    
    print(f"字段范围: [{field['range']['a']}, {field['range']['b']}]")
    print(f"Baseline 值: {baseline_run['field_value']['requested_value']}")
    print(f"Baseline BasicBlock 执行次数: {sum(json.dumps(field['runs'][0]).count(addr) for addr in ['BasicBlock'] if 'BasicBlock' in field['runs'][0].get('parse_health', {}))}")
    print()
    
    # 收集数据
    mutations_data = []
    for per_mut in sorted(field["diff"]["per_mutation"], key=lambda x: x["metrics"]["bb_multiset_l1"]):
        run_id = per_mut["run_id"]
        
        # 找到对应的 run
        run = None
        for r in field["runs"]:
            if r["run_id"] == run_id:
                run = r
                break
        
        if not run:
            continue
        
        metrics = per_mut["metrics"]
        
        mutations_data.append({
            "策略": run["strategy"],
            "Run ID": run_id,
            "修改值": f"{run['field_value']['requested_value']} → {run['field_value']['final_value']}",
            "碰撞处理": "✓ 是" if run['field_value']['collision_resolved'] else "✗ 否",
            "bb_multiset_l1": metrics["bb_multiset_l1"],
            "bb_set_jaccard": f"{metrics['bb_set_jaccard']:.4f}",
            "instr_prefix_len": metrics["lcp_instr_prefix_len"],
            "instr_ratio": f"{metrics['lcp_ratio']:.4f}",
        })
    
    print(tabulate(
        mutations_data,
        headers="keys",
        tablefmt="grid",
        floatfmt=".4f"
    ))
    print()
    
    print("=" * 100)
    print("统计信息:")
    print("=" * 100)
    
    bb_l1_values = [m["bb_multiset_l1"] for m in mutations_data]
    print(f"bb_multiset_l1 最小值: {min(bb_l1_values)} (策略: {mutations_data[bb_l1_values.index(min(bb_l1_values))]['策略']})")
    print(f"bb_multiset_l1 最大值: {max(bb_l1_values)} (策略: {mutations_data[bb_l1_values.index(max(bb_l1_values))]['策略']})")
    print(f"bb_multiset_l1 平均值: {sum(bb_l1_values) / len(bb_l1_values):.2f}")
    print()
    
    print("=" * 100)
    print("关键观察:")
    print("=" * 100)
    print(f"""
1. bb_multiset_l1 衡量的是 BasicBlock 执行次数差异的总和
   - 值越大 → 执行路径差异越大（更多 BB 地址出现计数差异）
   - 值越小 → 执行路径更相似

2. 在本实验中:
   - {mutations_data[0]['策略']} 策略影响最小 (L1={min(bb_l1_values)})
   - {mutations_data[-1]['策略']} 策略影响最大 (L1={max(bb_l1_values)})
   
3. 这反映了:
   - 不同的 payload 值对目标程序的执行路径影响不同
   - 某些策略变化触发更多的条件分支和循环次数变化
   """)


if __name__ == "__main__":
    try:
        from tabulate import tabulate
        main()
    except ImportError:
        print("错误: 需要安装 tabulate 模块")
        print("请运行: pip install tabulate")
