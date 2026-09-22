# 决策组件固定工程反例评测

本报告只验证固定合成反例的工程行为；不是独立人岗效果评测，不证明真实用户收益或统计显著性。

## 汇总

| 检查维度 | 固定案例数 | 对照符合预期 | 当前组件符合预期 |
|---|---:|---:|---:|
| 动作证据对齐 | 10 | 4 | 10 |
| 主动追问 | 4 | 1 | 4 |
| 逻辑行动规划 | 6 | 2 | 6 |
| 区间与多目标比较 | 5 | 1 | 5 |

六个指定动作混配反例：旧全局判定错误支持 6 个，当前绑定判定错误支持 0 个。该计数只适用于这些固定输入。

## 逐例归因

### alignment_separate_projects

跨项目工具与任务不能拼成同一操作

- 预期：`{"state": "unknown"}`
- 对照（现有全局技能/任务三态匹配）：`{"state": "pass"}`
- 当前（要求组与同动作工具—任务绑定）：`{"bound_groups": 2, "downgraded_groups": 2, "state": "unknown"}`
- 检查：符合预期
- 归因：跨项目工具与任务不能拼成同一操作

### alignment_separate_actions

同一段中的两种操作仍有不同工具

- 预期：`{"state": "unknown"}`
- 对照（现有全局技能/任务三态匹配）：`{"state": "pass"}`
- 当前（要求组与同动作工具—任务绑定）：`{"bound_groups": 2, "downgraded_groups": 2, "state": "unknown"}`
- 检查：符合预期
- 归因：同一段中的两种操作仍有不同工具

### alignment_skill_and_task

掌握工具和承担任务分别成立，不证明工具用于该任务

- 预期：`{"state": "unknown"}`
- 对照（现有全局技能/任务三态匹配）：`{"state": "pass"}`
- 当前（要求组与同动作工具—任务绑定）：`{"bound_groups": 2, "downgraded_groups": 2, "state": "unknown"}`
- 检查：符合预期
- 归因：掌握工具和承担任务分别成立，不证明工具用于该任务

### alignment_different_tool

清洗使用的工具与岗位要求不一致，不能确认Python清洗经历

- 预期：`{"state": "unknown"}`
- 对照（现有全局技能/任务三态匹配）：`{"state": "pass"}`
- 当前（要求组与同动作工具—任务绑定）：`{"bound_groups": 2, "downgraded_groups": 2, "state": "unknown"}`
- 检查：符合预期
- 归因：清洗使用的工具与岗位要求不一致，不能确认Python清洗经历

### alignment_skill_section

技能清单不能替代具体任务中的操作证据

- 预期：`{"state": "unknown"}`
- 对照（现有全局技能/任务三态匹配）：`{"state": "pass"}`
- 当前（要求组与同动作工具—任务绑定）：`{"bound_groups": 2, "downgraded_groups": 2, "state": "unknown"}`
- 检查：符合预期
- 归因：技能清单不能替代具体任务中的操作证据

### alignment_different_work

另一个项目只说明任务，没有说明使用Python

- 预期：`{"state": "unknown"}`
- 对照（现有全局技能/任务三态匹配）：`{"state": "pass"}`
- 当前（要求组与同动作工具—任务绑定）：`{"bound_groups": 2, "downgraded_groups": 2, "state": "unknown"}`
- 检查：符合预期
- 归因：另一个项目只说明任务，没有说明使用Python

### alignment_positive_control

同一动作中明确记录工具和任务，文本支持成立

- 预期：`{"state": "pass"}`
- 对照（现有全局技能/任务三态匹配）：`{"state": "pass"}`
- 当前（要求组与同动作工具—任务绑定）：`{"bound_groups": 2, "downgraded_groups": 0, "state": "pass"}`
- 检查：符合预期
- 归因：同一动作中明确记录工具和任务，文本支持成立

### alignment_negative_control

明确否定不能当成支持

- 预期：`{"state": "fail"}`
- 对照（现有全局技能/任务三态匹配）：`{"state": "fail"}`
- 当前（要求组与同动作工具—任务绑定）：`{"bound_groups": 2, "downgraded_groups": 0, "state": "fail"}`
- 检查：符合预期
- 归因：明确否定不能当成支持

### alignment_team_control

团队行为不能直接归于本人

- 预期：`{"state": "unknown"}`
- 对照（现有全局技能/任务三态匹配）：`{"state": "unknown"}`
- 当前（要求组与同动作工具—任务绑定）：`{"bound_groups": 2, "downgraded_groups": 0, "state": "unknown"}`
- 检查：符合预期
- 归因：团队行为不能直接归于本人

### alignment_future_control

计划不代表已完成经历

- 预期：`{"state": "unknown"}`
- 对照（现有全局技能/任务三态匹配）：`{"state": "unknown"}`
- 当前（要求组与同动作工具—任务绑定）：`{"bound_groups": 2, "downgraded_groups": 0, "state": "unknown"}`
- 检查：符合预期
- 归因：计划不代表已完成经历

### question_tenure_only

学历已知，候选岗位之间的年资判断仍未知，应先核对本人总年资

- 预期：`{"employer_routed": true, "irrelevant_questions": 0, "profile_unchanged": true, "useful_questions": 1}`
- 对照（对照启发式：固定先问学历（不是旧平台实际策略））：`{"employer_routed": true, "irrelevant_questions": 1, "profile_unchanged": true, "question_fields": ["education"], "useful_questions": 0}`
- 当前（候选条件与逻辑区间驱动的追问）：`{"employer_routed": true, "irrelevant_questions": 0, "profile_unchanged": true, "question_fields": ["experience_years"], "useful_questions": 1}`
- 检查：符合预期
- 归因：学历已知，候选岗位之间的年资判断仍未知，应先核对本人总年资

### question_education_only

本人学历未知而岗位门槛明确，学历追问在这个控制样例中有用

- 预期：`{"employer_routed": true, "irrelevant_questions": 0, "profile_unchanged": true, "useful_questions": 1}`
- 对照（对照启发式：固定先问学历（不是旧平台实际策略））：`{"employer_routed": true, "irrelevant_questions": 0, "profile_unchanged": true, "question_fields": ["education"], "useful_questions": 1}`
- 当前（候选条件与逻辑区间驱动的追问）：`{"employer_routed": true, "irrelevant_questions": 0, "profile_unchanged": true, "question_fields": ["education"], "useful_questions": 1}`
- 检查：符合预期
- 归因：本人学历未知而岗位门槛明确，学历追问在这个控制样例中有用

### question_employer_unknown

缺失来自JD，不能通过反问求职者补出招聘方门槛

- 预期：`{"employer_routed": true, "irrelevant_questions": 0, "profile_unchanged": true, "useful_questions": 0}`
- 对照（对照启发式：固定先问学历（不是旧平台实际策略））：`{"employer_routed": false, "irrelevant_questions": 1, "profile_unchanged": true, "question_fields": ["education"], "useful_questions": 0}`
- 当前（候选条件与逻辑区间驱动的追问）：`{"employer_routed": true, "irrelevant_questions": 0, "profile_unchanged": true, "question_fields": [], "useful_questions": 0}`
- 检查：符合预期
- 归因：缺失来自JD，不能通过反问求职者补出招聘方门槛

### question_or_satisfied

Python已满足OR分支，不应强制追问Java或重复询问已知学历

- 预期：`{"employer_routed": true, "irrelevant_questions": 0, "profile_unchanged": true, "useful_questions": 0}`
- 对照（对照启发式：固定先问学历（不是旧平台实际策略））：`{"employer_routed": true, "irrelevant_questions": 1, "profile_unchanged": true, "question_fields": ["education"], "useful_questions": 0}`
- 当前（候选条件与逻辑区间驱动的追问）：`{"employer_routed": true, "irrelevant_questions": 0, "profile_unchanged": true, "question_fields": [], "useful_questions": 0}`
- 检查：符合预期
- 归因：Python已满足OR分支，不应强制追问Java或重复询问已知学历

### plan_and

AND需要两个技能的条件式行动，不能只补一个

- 预期：`{"conditional_only": true, "logical_completeness": true, "minimum_actions": 2, "profile_unchanged": true}`
- 对照（对照启发式：平铺未出现的全部技能选项）：`{"action_sets": [["Python", "SQL"]], "conditional_only": true, "logical_completeness": true, "minimum_actions": 2, "profile_unchanged": true}`
- 当前（保留AND/OR的最小条件式行动集）：`{"action_sets": [["Python", "SQL"]], "conditional_only": true, "logical_completeness": true, "minimum_actions": 2, "planner_status": "planned", "profile_unchanged": true}`
- 检查：符合预期
- 归因：AND需要两个技能的条件式行动，不能只补一个

### plan_or

OR只需一个完整分支，不应把两个选项都列为必要行动

- 预期：`{"conditional_only": true, "logical_completeness": true, "minimum_actions": 1, "profile_unchanged": true}`
- 对照（对照启发式：平铺未出现的全部技能选项）：`{"action_sets": [["Java", "Python"]], "conditional_only": true, "logical_completeness": true, "minimum_actions": 2, "profile_unchanged": true}`
- 当前（保留AND/OR的最小条件式行动集）：`{"action_sets": [["Java"], ["Python"]], "conditional_only": true, "logical_completeness": true, "minimum_actions": 1, "planner_status": "planned", "profile_unchanged": true}`
- 检查：符合预期
- 归因：OR只需一个完整分支，不应把两个选项都列为必要行动

### plan_nested

两个OR组必须各选一项，平铺会多要求两项

- 预期：`{"conditional_only": true, "logical_completeness": true, "minimum_actions": 2, "profile_unchanged": true}`
- 对照（对照启发式：平铺未出现的全部技能选项）：`{"action_sets": [["Java", "MySQL", "Python", "SQL"]], "conditional_only": true, "logical_completeness": true, "minimum_actions": 4, "profile_unchanged": true}`
- 当前（保留AND/OR的最小条件式行动集）：`{"action_sets": [["Java", "MySQL"], ["Java", "SQL"], ["MySQL", "Python"]], "conditional_only": true, "logical_completeness": true, "minimum_actions": 2, "planner_status": "bounded", "profile_unchanged": true}`
- 检查：符合预期
- 归因：两个OR组必须各选一项，平铺会多要求两项

### plan_supported_or

一个分支已有自述支持，不再将另一选项当必要缺口

- 预期：`{"conditional_only": true, "logical_completeness": true, "minimum_actions": 0, "profile_unchanged": true}`
- 对照（对照启发式：平铺未出现的全部技能选项）：`{"action_sets": [["Java"]], "conditional_only": true, "logical_completeness": true, "minimum_actions": 1, "profile_unchanged": true}`
- 当前（保留AND/OR的最小条件式行动集）：`{"action_sets": [[]], "conditional_only": true, "logical_completeness": true, "minimum_actions": 0, "planner_status": "planned", "profile_unchanged": true}`
- 检查：符合预期
- 归因：一个分支已有自述支持，不再将另一选项当必要缺口

### plan_binding_gap

已出现的技能与任务不应掩盖缺少动作关联的补证需求

- 预期：`{"binding_actions": 1, "conditional_only": true}`
- 对照（只按全局叶支持规划）：`{"binding_actions": 0, "conditional_only": true}`
- 当前（加入动作关联的条件式补证规划）：`{"binding_actions": 1, "conditional_only": true}`
- 检查：符合预期
- 归因：关联缺口只要求核对既有经历，不把行动清单写成已经会做；同一绑定在两个要求组中去重。

### plan_binding_control

已出现的技能与任务不应掩盖缺少动作关联的补证需求

- 预期：`{"binding_actions": 0, "conditional_only": true}`
- 对照（只按全局叶支持规划）：`{"binding_actions": 0, "conditional_only": true}`
- 当前（加入动作关联的条件式补证规划）：`{"binding_actions": 0, "conditional_only": true}`
- 检查：符合预期
- 归因：关联缺口只要求核对既有经历，不把行动清单写成已经会做；同一绑定在两个要求组中去重。

### interval_missing_skill

SQL没有材料是未知，上界应保留，不能当作明确不会

- 预期：`{"coverage_bounds": [0.5, 1.0]}`
- 对照（对照启发式：独立条件计分，缺失当0且不检查动作绑定）：`{"coverage_bounds": [0.5, 0.5]}`
- 当前（要求组逻辑区间＋动作绑定）：`{"coverage_bounds": [0.5, 1.0], "requirement_count": 2, "state": "needs_evidence"}`
- 检查：符合预期
- 归因：SQL没有材料是未知，上界应保留，不能当作明确不会

### interval_negative_skill

明确否定与缺证据不同，SQL分支不再保留可支持上界

- 预期：`{"coverage_bounds": [0.5, 0.5]}`
- 对照（对照启发式：独立条件计分，缺失当0且不检查动作绑定）：`{"coverage_bounds": [0.5, 0.5]}`
- 当前（要求组逻辑区间＋动作绑定）：`{"coverage_bounds": [0.5, 0.5], "requirement_count": 2, "state": "needs_evidence"}`
- 检查：符合预期
- 归因：明确否定与缺证据不同，SQL分支不再保留可支持上界

### interval_no_requirements

没有解析出要求应标未知，不能解释为全部满足或完全不匹配

- 预期：`{"coverage_bounds": [0.0, 1.0]}`
- 对照（对照启发式：独立条件计分，缺失当0且不检查动作绑定）：`{"coverage_bounds": [0.0, 0.0]}`
- 当前（要求组逻辑区间＋动作绑定）：`{"coverage_bounds": [0.0, 1.0], "requirement_count": 0, "state": "needs_evidence"}`
- 检查：符合预期
- 归因：没有解析出要求应标未知，不能解释为全部满足或完全不匹配

### interval_bound_gap

原全局分数满支持，但跨动作证据不能关闭逻辑不确定区间

- 预期：`{"coverage_bounds": [0.0, 1.0]}`
- 对照（对照启发式：独立条件计分，缺失当0且不检查动作绑定）：`{"coverage_bounds": [1.0, 1.0]}`
- 当前（要求组逻辑区间＋动作绑定）：`{"coverage_bounds": [0.0, 1.0], "requirement_count": 2, "state": "needs_evidence"}`
- 检查：符合预期
- 归因：原全局分数满支持，但跨动作证据不能关闭逻辑不确定区间

### frontier_tradeoffs

保留证据—薪资取舍与未知岗位，剔除明确被支配岗位

- 预期：`{"dominated_removed": true, "frontier_ids": ["higher-pay", "strong-evidence", "unknown-record"], "unknown_salary_preserved": true}`
- 对照（对照启发式：缺失当0后的点值Pareto）：`{"dominated_removed": true, "frontier_ids": ["higher-pay", "strong-evidence"], "unknown_salary_preserved": false}`
- 当前（逻辑区间上的稳健Pareto）：`{"dominated_removed": true, "frontier_ids": ["higher-pay", "strong-evidence", "unknown-record"], "unknown_salary_preserved": true}`
- 检查：符合预期
- 归因：未知记录不一定优秀；在证据不足时不能证明其被支配。前沿只表示当前候选间的取舍，不保证适岗。

## 来源与解释范围

固定样例摘要：`02ef6bbf0e6bb4ec4350812fa8ff6100bf786fc24e8b223347e992feda946a72`。源码摘要、版本与全部输入保存在同目录JSON中。

- 每项预期由固定场景人工式定义，不来自真实用户标注；禁止将通过数解释为人岗准确率。
- 对照中只有全局技能/任务匹配对应已有代码；固定先问学历、平铺缺口、缺失当0都是明示的对照启发式。
- 样本选择旨在暴露特定工程错误，不具有随机性、总体代表性或统计显著性。
- 最小行动数只计算条件式行动条目，不估计学习成本、真实能力变化或录用机会。
- 逻辑上下界不是统计置信区间；稳健前沿不是全库最优或录用概率。
