"use strict";
const $ = id => document.getElementById(id);
const state = {samples: [], overview: null, revision: 0, result: null, payload: null, target: null, rewritten: null, view: "demand", demandRequest: 0, detailRequest: 0, skillRequest: 0};
state.confirmation = ""; state.origins = {}; state.preview = null;
const fields = [["city","city"],["intent","intent"],["education","education"],["years","experience_years"],["salary","salary_min"],["salary-max","salary_max"],["district","district"],["education-full-time","education_full_time"]];
const escape = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const number = value => Number(value).toLocaleString("zh-CN");
const money = value => value == null ? "—" : `${number(value)}`;
let toastTimer;

function toast(message) { $("toast").textContent = message; $("toast").classList.remove("hidden"); clearTimeout(toastTimer); toastTimer = setTimeout(() => $("toast").classList.add("hidden"), 4000); }
function errorMessage(error) { return error instanceof Error ? error.message : "操作未完成，请重试。"; }
async function api(path, data, method) {
  const response = await fetch(`/api/v2/${path}`, {method: method || (data ? "POST" : "GET"), headers: data ? {"Content-Type":"application/json"} : {}, body: data ? JSON.stringify(data) : undefined, credentials:"same-origin"});
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : response.status === 422 ? "输入格式不正确，请检查年资、薪资和简历长度。" : `请求未完成（${response.status}），请稍后重试。`);
  return result;
}
function view(name) {
  if (!["demand", "match", "lab", "research"].includes(name)) return;
  state.view = name;
  document.querySelectorAll(".view").forEach(element => element.classList.toggle("hidden", element.id !== `view-${name}`));
  document.querySelectorAll("[data-view]").forEach(element => { element.classList.toggle("active", element.dataset.view === name); element.setAttribute("aria-current", element.dataset.view === name ? "page" : "false"); });
  $("breadcrumb").textContent = `工作台 / ${{demand:"岗位需求",match:"简历匹配",lab:"算法对比",research:"研究与标注"}[name]}`;
  history.replaceState(null, "", `#${name}`);
  if (name === "lab") updateExperiment();
  if (name === "research") loadResearch();
}
function bar(item, max = 1) { return `<div class="bar-row"><div class="bar-heading"><span>${escape(item.name)}</span><span class="bar-value">${number(item.count)} · ${(item.ratio*100).toFixed(1)}%</span></div><div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, item.ratio/max*100)}%"></div></div></div>`; }
function renderOverview(data) {
  $("filter-count").textContent = `${number(data.total)} 个去重岗位`;
  $("data-source").textContent = data.source_label;
  const skillTop = data.skills[0];
  $("metrics").innerHTML = [
    ["当前岗位样本",number(data.total),`原始 ${number(data.raw_total)} 条 · ${data.dedup_policy || "去重统计"}`],
    ["职位类型",$("category").value ? "1" : number(data.category_count),$("category").value || "以当前数据实际分类为准"],
    ["可比月薪样本",number(data.salary_count),`${data.total ? (data.salary_count/data.total*100).toFixed(1) : 0}% · 日薪、时薪另计`],
    ["城市信息未知",number(data.unknown_location),"保留未知，不补造工作地点"]
  ].map(([title,value,foot]) => `<div class="metric"><p class="metric-title">${escape(title)}</p><p class="metric-value">${escape(value)}</p><p class="metric-foot">${escape(foot)}</p></div>`).join("");
  $("skill-bars").innerHTML = data.skills.length ? data.skills.slice(0,12).map(item => bar(item, Math.max(skillTop.ratio, .01))).join("") : '<p class="muted">没有可识别的技能词。</p>';
  const q = data.salary_quantiles;
  $("salary-chart").innerHTML = q ? `<div class="salary-main">${money(q[1])}<small>元 / 月 · 中位数</small></div><div class="salary-range"></div><div class="quartiles"><div>${money(q[0])}<small>25% 分位</small></div><div>${money(q[1])}<small>50% 分位</small></div><div>${money(q[2])}<small>75% 分位</small></div></div>` : '<p class="loading">月薪样本不足30条，不输出分位数。</p>';
  $("salary-note").textContent = `基于 ${number(data.salary_count)} 条月薪广告；此处按方向汇总，未控制年资，不是个人报价建议。`;
  $("experience-bars").innerHTML = data.experience.map(item => bar(item)).join("");
  $("districts").innerHTML = data.districts.map(item => `<div class="district"><span>${escape(item.name)}</span><span class="muted">${number(item.count)}</span></div>`).join("");
  $("alternative-groups").innerHTML = data.alternative_examples.length ? data.alternative_examples.map(item => `<div class="logic-example"><span class="logic-chip">任选一项</span><strong>${escape(item.label)}</strong><small>原文片段：“${escape(item.quote)}” · 规则识别，需结合上下文核对</small></div>`).join("") : '<p class="footnote">当前方向没有识别到明确的替代技能组。</p>';
  $("source-note").textContent = `快照 ${data.snapshot}。${data.notes.join(" ")} 外部 merged_data.xlsx 尚未在本环境核验。`;
}
async function loadOverview() {
  const request = ++state.demandRequest;
  $("filter-count").textContent = "正在更新统计…";
  try { const data = await api(`overview?category=${encodeURIComponent($("category").value)}`); if (request !== state.demandRequest) return; renderOverview(data); }
  catch (error) { $("filter-count").textContent = "读取失败"; toast(errorMessage(error)); }
}
function invalidate() {
  state.revision += 1; state.result = null; state.payload = null; state.target = null; state.rewritten = null;
  state.detailRequest += 1; state.skillRequest += 1;
  state.confirmation="";state.preview=null;$("profile-confirmation").classList.add("hidden");
  if ($("detail-dialog").open) $("detail-dialog").close();
  $("char-count").textContent = `${$("resume-text").value.length} 字`;
  $("pipeline").classList.add("hidden"); $("profile-skills").classList.add("hidden"); $("results-meta").textContent = "";
  $("results-title").textContent = "你的候选岗位";
  $("results").innerHTML = '<div class="empty-state"><span class="empty-glyph">⌕</span><h3>简历已准备好</h3><p>点击“分析简历并匹配岗位”，以当前经历和条件重新检索。</p></div>';
  $("comparison-results").innerHTML = '<div class="empty-state compact"><h3>输入已更新</h3><p>运行三组对比，查看当前简历对应的结果。</p></div>';
  updateExperiment();
}
function applySample(id) {
  const sample = state.samples.find(item => item.id === id);
  if (!sample) return;
  $("resume-text").value = sample.text;
  for (const [id,key] of fields) { $(id).value = sample.preferences[key] ?? ""; state.origins[key]="sample"; }
  $("strict").checked = false;
  invalidate();
}
function formData() {
  return {text:$("resume-text").value, preferences:{city:$("city").value.trim(),intent:$("intent").value.trim(),education:$("education").value,education_full_time:$("education-full-time").value === "" ? null : $("education-full-time").value === "true",experience_years:$("years").value === "" ? null : Number($("years").value),salary_min:$("salary").value === "" ? null : Number($("salary").value),salary_max:$("salary-max").value === "" ? null : Number($("salary-max").value),district:$("district").value},limit:10,method:"hybrid",strict_unknown:$("strict").checked,query_mode:$("query-mode").value,confirmation_token:state.confirmation,field_origins:state.origins,research_consent:$("research-consent").checked};
}

function clearSamplePreferences(){
  for(const [id,key] of fields)if(state.origins[key]==="sample"){$(id).value="";state.origins[key]="unknown";}
}
async function previewProfile(){
  const revision=state.revision,payload=formData();
  const preview=await api("profile/preview",payload);if(revision!==state.revision)return;
  for(const [id,key] of fields){$(id).value=preview.preferences[key]??"";state.origins[key]=preview.fields.find(field=>field.key===key)?.source||"unknown";}
  state.preview=preview;
  const sources={sample:"虚构样例",user_input:"表单输入",text_extracted:"从简历抽取",unknown:"未说明",sample_stale:"旧样例已撤销"};
  $("profile-confirmation").classList.remove("hidden");
  $("profile-confirmation").innerHTML=`<span class="eyebrow">推荐前确认</span><h2>这份画像是否准确？</h2><p class="footnote">修改左侧字段可重新解析。空值保持未知。</p><div class="profile-field-list">${preview.fields.map(field=>`<div class="profile-field ${field.conflict?'has-conflict':''}"><strong>${escape(field.label)}</strong><div><span>${escape(typeof field.value === 'boolean' ? (field.value ? '是' : '否') : field.value??'未说明')||'未说明'}</span><small>${escape(sources[field.source]||field.source)}</small>${field.quote?`<p class="footnote">原文：${escape(field.quote)}</p>`:''}${field.conflict?`<p class="conflict-note">与抽取值“${escape(field.extracted)}”不同，请核对。</p>`:''}</div></div>`).join("")}</div>${preview.conflicts.length?'<label class="checkbox"><input type="checkbox" id="ack-conflicts">我已核对冲突，并确认采用当前表单值</label>':''}<button class="primary" id="confirm-profile" ${preview.conflicts.length?'disabled':''}>确认这份画像并匹配</button><p class="footnote">${escape(preview.note)}</p>`;
  if($("ack-conflicts"))$("ack-conflicts").addEventListener("change",()=>{$("confirm-profile").disabled=!$("ack-conflicts").checked;});
  $("confirm-profile").addEventListener("click",async()=>{
    if(revision!==state.revision)return;const button=$("confirm-profile");button.disabled=true;
    try{const result=await api("profile/confirm",{...formData(),user_confirmed:true,acknowledged_conflicts:preview.conflicts});if(revision!==state.revision)return;state.confirmation=result.confirmation_token;$("profile-confirmation").innerHTML='<p class="confirmed-note">✓ 已确认当前画像；编辑简历或偏好后会重新确认。</p>';await recommend();}
    catch(error){toast(errorMessage(error));button.disabled=false;}
  });
  $("results").innerHTML='<div class="empty-state compact"><h3>先核对画像，再检索岗位</h3><p>上方会标明样例、文本抽取和表单输入的来源。</p></div>';
}
function updateExperiment() {
  const sample = state.samples.find(item => item.id === $("sample").value);
  $("experiment-name").textContent = `${sample ? sample.name : "当前编辑简历"} · ${$("intent").value || "方向待补充"}`;
  $("engine-status").textContent = state.overview?.engine || "正在检查检索方式…";
}
function renderDecisionPanel(result) {
  const support = result.decision_support;
  if (!support) return "";
  const questions = support.questions || [];
  const points = result.comparison_frontier?.points || [];
  const candidateById = new Map(result.jobs.map(job => [job.id, job]));
  return `<section class="card decision-panel"><span class="eyebrow">帮助你做下一步判断</span><h3>先补关键证据，再比较岗位取舍</h3>${questions.length ? `<div class="decision-questions">${questions.map((item,index) => `<div><strong>${index+1}. ${escape(item.question)}</strong><p class="footnote">影响 ${item.affected_job_ids.length} 个候选的判断；${item.kind === 'profile_field' ? '补全后重新确认画像' : '补充真实经历后重新匹配'}。</p><button type="button" class="text-button" data-question-control="${escape(item.control)}">去补充 →</button></div>`).join("")}</div>` : '<p class="muted">目前没有需要优先补充的画像问题，可以逐个核对岗位证据。</p>'}<details class="decision-comparison"><summary>比较证据、广告薪资与区域 · ${points.filter(p => p.on_frontier).length} 个候选保留取舍</summary><p class="footnote">证据区间：下界为当前文本支持，上界允许尚未说明的要求成立。区间越宽，需要确认的信息越多。</p><div class="comparison-scroll"><table><thead><tr><th>岗位</th><th>必要要求证据区间</th><th>广告薪资</th><th>门槛确认</th><th>比较结果</th></tr></thead><tbody>${points.map(point => {const job=candidateById.get(point.job_id);return `<tr><td>${escape(job?.title)}</td><td>${point.axes.evidence.map(v => `${Math.round(v*100)}%`).join('–')}</td><td>${escape(job?.salary_raw || '未知')}</td><td>${job?.decision_certificate?.unknown_constraints?.length ? '部分待确认' : '已知条件无冲突'}</td><td>${point.on_frontier ? '保留取舍' : '有候选在所列维度稳健占优'}</td></tr>`;}).join("")}</tbody></table></div><p class="footnote">${escape(result.comparison_frontier?.note)}</p></details></section>`;
}
function renderEmployerChecks(result) {
  const checks=result.decision_support?.employer_checks || [];
  if (!checks.length) return '';
  const jobs=new Map(result.jobs.map(job=>[job.id,job]));
  return `<details class="card"><summary>联系招聘方前，先核对这些信息</summary><p class="footnote">这些字段属于岗位信息，补充个人简历不能代替招聘方确认。</p>${checks.map(item=>`<p><strong>${escape(jobs.get(item.job_id)?.title)}</strong>：${escape([...item.fields,'当前是否在招'].join('、'))}。</p>`).join('')}</details>`;
}
function decisionBadge(job) {
  const proof=job.decision_certificate;
  if (!proof) return '';
  const bounds=proof.coverage_bounds.map(v=>`${Math.round(v*100)}%`).join('–');
  return `<p class="decision-badge">可解析必要要求的证据区间 <strong>${bounds}</strong>${job.alignment_summary?.downgraded_groups ? ' · 工具与任务的对应关系待补证' : ''}</p>`;
}
function renderAlignment(result) {
  const alignment=result.alignment;
  if (!alignment) return '';
  const groups=alignment.groups.filter(group=>group.task_binding.required);
  if (!groups.length && alignment.summary.status !== 'not_evaluated') return '';
  return `<div class="detail-section"><h3>工具与任务是否来自同一段经历？</h3>${groups.map(group=>`<div class="gap-row"><strong>${escape(group.label)}</strong><span class="tag ${group.status==='pass' ? 'positive' : 'warning'}">${group.status==='pass' ? '已有对应文本' : '对应关系待确认'}</span><p class="footnote">${escape(group.reasons.join('；'))}</p>${group.best_block ? `<div class="quote-box resume"><small>最相关经历块 · ${group.best_block.start}–${group.best_block.end}</small><p>${escape(group.best_block.display_quote)}</p></div>` : '<p class="muted">尚未定位到对应的本人实践。</p>'}</div>`).join('')}<p class="footnote">${alignment.summary.status==='not_evaluated' ? '文本或要求树超出当前解析预算，尚未完成对齐。' : '区分技能分别出现与在同一动作中完成任务；文本关系支持仍需核对实际经历。'}</p></div>`;
}
function renderActionPlans(result) {
  const planner=result.action_plans;
  if (!planner) return '';
  return `<div class="detail-section action-plans"><h3>最小补证路线</h3><p class="footnote">比较以下条件式路线，按真实经历决定补证或先学习。有待确认项的路线不构成完整方案。</p>${planner.plans.map((plan,index)=>`<div class="gap-row"><strong>路线 ${index+1} · ${plan.actions.length} 项行动${plan.all_required_groups_accounted_for ? "" : " · 仍有要求待确认"}</strong>${plan.actions.length ? `<ol>${plan.actions.map(action=>`<li><strong>${escape(action.label || action.key)}</strong> · ${action.kind === 'learning' ? '学习并形成实践证据' : action.kind === 'task_binding' ? '补充工具与任务的对应经历' : '补充已有实践证据'}${action.instruction ? `<p class="footnote">${escape(action.instruction)}</p>` : ''}</li>`).join('')}</ol>` : (plan.all_required_groups_accounted_for ? '<p>当前可解析要求无需新增行动；仍需核对真实经历与岗位完整职责。</p>' : '<p>还有要求或经历关系未完成核对，暂不判断是否需要新增行动。</p>')}</div>`).join('') || '<p class="muted">当前要求不足以生成完整路线，请先核对原文。</p>'}${planner.truncated ? '<p class="footnote">要求较复杂，搜索已达到预算；当前路线不承诺全局最少。</p>' : ''}${planner.non_actionable_constraints?.length ? '<p class="footnote">另有学历、年资或岗位字段需要单独确认，补充技能不会自动消除这些条件。</p>' : ''}</div>`;
}
function renderResults(result) {
  $("pipeline").classList.remove("hidden");
  $("pipeline").innerHTML = result.trace.map(step => `<span class="${step.status === "needs_input" ? "pending" : ""}" title="${escape(step.detail)}">${step.status === "done" ? "✓" : "·"} ${escape(step.step)}</span>`).join("");
  $("profile-skills").classList.remove("hidden");
  $("profile-skills").innerHTML = '<span class="muted">简历技能证据</span> ' + Object.values(result.profile.skills).map(item => `<span class="tag ${item.level === "否定" ? "negative" : item.level === "实践" ? "positive" : ""}">${escape(item.skill)} · ${escape(item.level)}</span>`).join("");
  if (result.action === "clarify") {
    $("results-title").textContent = "先确认几项必要信息";
    $("results").innerHTML = `<div class="clarification"><h3>需要你补充</h3><ol>${result.questions.map(question => `<li>${escape(question)}</li>`).join("")}</ol><p>在左侧补充后再匹配。关键词清单不会被当作项目能力。</p></div>`;
    return;
  }
  $("results-title").textContent = `${result.jobs.length} 个候选岗位`;
  $("results-meta").textContent = `${number(result.total_eligible)} 个无明确硬冲突 · ${number(result.latency_ms)} ms`;
  if (!result.jobs.length) { $("results").innerHTML = `<div class="empty-state"><h3>当前条件下暂无匹配</h3><p>${escape(result.message)}</p></div>`; return; }
  $("results").innerHTML = renderDecisionPanel(result) + renderEmployerChecks(result) + result.jobs.map(job => {
    const unknown = job.constraints.filter(item => item.status === "unknown");
    const good = job.matched.filter(item => item.status === "pass" && item.strength >= .7).slice(0,3);
    const gaps = job.gaps.slice(0,3).map(item => item.skill).join("、");
    return `<article class="job-card"><div class="job-top"><div><h3 class="job-title">${escape(job.title)}</h3><p class="job-company">${escape(job.company || "企业信息未提供")}</p></div><span class="job-salary">${escape(job.salary_raw || "薪资未知")}</span></div><div class="job-meta"><span class="tag">${escape(job.district || job.city || "地点未知")}</span><span class="tag">${escape(job.requirements || "门槛待确认")}</span><span class="tag blue">${escape(job.category)}</span>${unknown.map(item => `<span class="tag warning">${escape(item.name)}待确认</span>`).join("")}</div><div class="job-evidence">${decisionBadge(job)}<p>${good.length ? `已有证据：${escape(good.map(item => item.skill).join("、"))}` : "技能证据较少，请先核对岗位职责"}</p>${gaps ? `<p class="muted">待补证：${escape(gaps)}</p>` : '<p class="muted">词典范围内未发现缺口，仍需核对完整JD。</p>'}</div><div class="job-bottom"><span class="score-label"><strong>${job.score.toFixed(1)}</strong>本次匹配分 · 非录用概率</span><div class="job-actions"><button class="feedback" data-feedback="like" data-job="${job.id}">感兴趣</button><button class="feedback" data-feedback="skip" data-job="${job.id}">暂不考虑</button><button class="feedback" data-exclude="${job.id}">不考虑该公司</button><button class="secondary" data-detail="${job.id}">查看证据与差距 →</button></div></div></article>`;
  }).join("") + `<p class="source-note">${escape(result.retriever)}。${escape(result.message)}未知条件不代表已经满足；历史岗位有效性需要另行确认。</p>`;
}
async function recommend(event) {
  event?.preventDefault();
  if (!$("resume-form").reportValidity()) return;
  if(!state.confirmation){try{await previewProfile();}catch(error){toast(errorMessage(error));}return;}
  const revision = state.revision; const payload = formData();
  state.result=null;state.payload=null;
  $("recommend").disabled = true; $("recommend").textContent = "正在分析与匹配…";
  $("results").innerHTML = '<div class="card"><p class="loading">检查条件，检索岗位，逐条核对技能证据…</p></div>';
  try { const result = await api("recommend",payload); if (revision !== state.revision) return; state.result=result;state.payload=payload;renderResults(result); }
  catch (error) { if (revision === state.revision) $("results").innerHTML=`<div class="alert error">${escape(errorMessage(error))}</div>`; }
  finally { $("recommend").disabled=false;$("recommend").innerHTML='重新匹配当前画像 <span>↗</span>'; }
}
function quoteBox(evidence,label,kind="") { return `<div class="quote-box ${kind}"><small>${escape(label)} · ${evidence.start}–${evidence.end}</small><mark>${escape(evidence.quote)}</mark><p>${escape(evidence.context || "")}</p></div>`; }
async function openDetail(id) {
  if (!state.payload) return;
  const revision = state.revision;
  const request = ++state.detailRequest; state.skillRequest += 1;
  state.target = id;
  $("detail-content").innerHTML='<p class="loading">正在核对岗位要求与简历…</p>';
  if (!$("detail-dialog").open) $("detail-dialog").showModal();
  try {
    const result = await api("diagnose",{...state.payload,job_id:id});
    if (revision !== state.revision || state.target !== id || request !== state.detailRequest || !$("detail-dialog").open) return;
    const q=result.market.quantiles;
    $("detail-content").innerHTML=`<div class="detail-heading"><div><h2 id="detail-title">${escape(result.job.title)}</h2><p class="muted">${escape(result.job.company)}</p></div><span class="job-salary">${escape(result.job.salary_raw)}</span></div><div class="job-meta">${result.constraints.map(item=>`<span class="tag ${item.status === "fail" ? "negative" : item.status === "unknown" ? "warning" : "positive"}">${escape(item.name)} · ${{pass:"满足",fail:"冲突",unknown:"待确认"}[item.status]}</span>`).join("")}</div><p class="footnote">快照 ${escape(result.snapshot)} · 画像 ${escape(result.profile_version)}。可解析要求证据覆盖 ${result.coverage}%；不代表整体胜任度。</p><div class="detail-section"><h3>已定位证据 · 仍需核对支持程度</h3>${result.matched.slice(0,10).map(item=>`<div><p><strong>${escape(item.skill)}</strong> <span class="tag">${escape(item.level)}</span>${item.kind === "any" ? `<span class="tag blue">替代组：${escape(item.group)}</span>` : ""}</p><div class="evidence-pair">${quoteBox(item.resume_evidence,"简历", "resume")}${quoteBox(item.job_evidence,item.job_evidence.field==="requirements"?"任职要求":"岗位职责")}</div><p class="footnote">${escape(item.note)}</p></div>`).join("") || '<p class="muted">当前未识别出双侧技能证据。</p>'}</div><div class="detail-section"><h3>简历尚未证明的要求</h3>${result.gaps.map(gap=>`<div class="gap-row"><strong>${escape(gap.skill)}${gap.preferred ? ' <span class="tag">优先项</span>' : ""}</strong><p>${escape(gap.message)}。</p><button class="secondary" data-start-action="${escape(gap.group_id)}">记录补证行动</button></div>`).join("") || '<p class="muted">词典范围内未发现缺口；请继续核对完整岗位职责。</p>'}</div>${renderAlignment(result)}${renderActionPlans(result)}<div class="detail-section"><h3>同方向、同年资薪资参考</h3><p>${q ? `${escape(result.market.category)} · ${number(result.market.n)} 条月薪样本，P25 / P50 / P75：${q.map(money).join(" / ")} 元。` : `${number(result.market.n)} 条同组月薪样本，不足30条，暂不输出分位数。`}</p><p class="footnote">${escape(result.market.note)}</p></div><div class="detail-actions"><button class="primary" id="rewrite-button">整理现有经历</button><button class="secondary" id="interview-button">生成面试练习</button><button class="secondary" id="coach-button">制定补证行动</button><button class="text-button" id="graph-button">查看岗位需求图</button></div><div id="skill-output"></div><details class="detail-section"><summary>展开岗位职责原文（联系方式已隐藏）</summary><div class="jd-text">${escape(result.description)}</div></details>`;
    $("rewrite-button").addEventListener("click",rewrite);
    $("interview-button").addEventListener("click",interview);
    $("coach-button").addEventListener("click",coach);
    $("graph-button").addEventListener("click",showGraph);
    $("detail-content").querySelectorAll("[data-start-action]").forEach(button=>button.addEventListener("click",()=>startAction(button.dataset.startAction,button)));
  } catch(error) { if (revision === state.revision && request === state.detailRequest && $("detail-dialog").open) $("detail-content").innerHTML=`<div class="alert error">${escape(errorMessage(error))}</div>`; }
}
async function rewrite() {
  const button=$("rewrite-button"); button.disabled=true;
  const revision=state.revision, target=state.target, request=++state.skillRequest;
  const current=()=>revision===state.revision && target===state.target && request===state.skillRequest && $("detail-dialog").open;
  try { const result=await api("rewrite",{...state.payload,job_id:target});if(!current())return;state.rewritten=result.revised;
    $("skill-output").innerHTML=`<div class="detail-section"><h3>事实保全整理 <span class="tag positive">新增事实 ${result.new_claims}</span></h3><p class="footnote">${escape(result.explanation)}</p><div class="rewrite-grid"><div><strong>原简历</strong><pre>${escape(result.original)}</pre></div><div><strong>建议顺序</strong><pre>${escape(result.revised)}</pre></div></div><button class="primary" id="apply-rewrite">审核后应用，并重新匹配</button><p class="footnote">未补造技能、公司或工作年限；当前整理器只调整完整经历块顺序。</p></div>`;
    $("apply-rewrite").addEventListener("click",async()=>{if(!current())return;const revised=result.revised;$("detail-dialog").close();$("resume-text").value=revised;$("sample").value="";invalidate();await recommend();toast("已应用完整经历块整理，请核对新画像。表达整理不意味着能力提高。");});
  }catch(error){toast(errorMessage(error));}finally{button.disabled=false;}
}
async function interview() {
  const button=$("interview-button");button.disabled=true;
  const revision=state.revision, target=state.target, request=++state.skillRequest;
  try { const result=await api("interview",{...state.payload,job_id:target});if(revision!==state.revision || target!==state.target || request!==state.skillRequest || !$("detail-dialog").open)return;$("skill-output").innerHTML=`<div class="detail-section"><h3>面试练习</h3>${result.questions.map((item,index)=>`<div class="quote-box"><strong>${index+1}. ${escape(item.question)}</strong><p class="footnote">自查：${item.rubric.map(escape).join("；")}。</p></div>`).join("") || '<p class="muted">先补充一段可核对的项目经历，再生成针对性问题。</p>'}<p class="footnote">${escape(result.note)}</p></div>`;}catch(error){toast(errorMessage(error));}finally{button.disabled=false;}
}
async function compare() {
  if (!$("resume-form").reportValidity()) {view("match");return;}
  if(!state.confirmation){view("match");try{await previewProfile();}catch(error){toast(errorMessage(error));}return;}
  const revision=state.revision;$("run-comparison").disabled=true;$("comparison-results").innerHTML='<p class="loading">正在用同一输入运行三组方法…</p>';
  try { const result=await api("compare",formData());if(revision!==state.revision)return;
    $("comparison-results").innerHTML=`<div class="table-wrap"><table><thead><tr><th>方法</th><th>返回岗位</th><th>要求证据覆盖均值</th><th>硬违规</th><th>不同公司</th><th>耗时</th></tr></thead><tbody>${result.rows.map(row=>`<tr class="${row.method === "hybrid" ? "best-row" : ""}"><td>${escape(row.label)}</td><td>${row.count}</td><td>${row.mean_skill_coverage == null ? "不适用" : row.mean_skill_coverage+"%"}</td><td>${row.hard_violations}</td><td>${row.company_count}</td><td>${number(row.latency_ms)} ms</td></tr>`).join("")}</tbody></table></div><p class="footnote">${escape(result.note)}</p><div class="detail-section"><h3>各方法前三项</h3>${result.rows.map(row=>`<p class="footnote"><strong>${escape(row.label)}</strong>：${row.titles.map(escape).join(" / ") || "未返回岗位，请检查必要信息"}</p>`).join("")}</div>`;
  }catch(error){$("comparison-results").innerHTML=`<div class="alert error">${escape(errorMessage(error))}</div>`;}finally{$("run-comparison").disabled=false;}
}
async function coach(){
  const button=$("coach-button");button.disabled=true;
  const revision=state.revision,target=state.target,request=++state.skillRequest;
  $("skill-output").innerHTML='<p class="loading">正在根据原文选择补证行动…</p>';
  try{const result=await api("coach",{...state.payload,job_id:target});if(revision!==state.revision||target!==state.target||request!==state.skillRequest||!$("detail-dialog").open)return;
    $("skill-output").innerHTML=`<div class="detail-section"><h3>优先行动</h3>${result.items.map(item=>`<div class="gap-row"><strong>${escape(item.skill)}</strong><p>${escape(item.action)}</p>${quoteBox(item.evidence,"岗位依据")}</div>`).join("")}<p class="footnote">${escape(result.mode)}。${escape(result.message)}</p></div>`;
  }catch(error){toast(errorMessage(error));}finally{button.disabled=false;}
}
function renderRequirementTree(node, depth=0) {
  if(!node || depth>7)return '<span class="tag warning">解析待核对</span>';
  const modality={required:"必需",preferred:"优先",negated:"否定",unknown:"模态待确认",mixed:"混合"}[node.modality]||"待确认";
  if(node.type)return `<div class="requirement-leaf"><span class="tag ${node.type==='task'?'blue':''}">${escape(node.text||node.key)}</span><small>${escape(modality)}</small></div>`;
  const label=node.op==='any'?'任选一个完整分支（OR）':'全部分支（AND）';
  return `<div class="requirement-node"><span class="logic-chip">${label} · ${escape(modality)}${node.parse_status==='unknown'?' · 作用域待确认':''}</span><div class="requirement-children">${(node.children||[]).map(child=>renderRequirementTree(child,depth+1)).join('')||'<p class="muted">尚未识别到要求，不能视为没有门槛。</p>'}</div></div>`;
}
async function showGraph(){
  const target=state.target,revision=state.revision,request=++state.skillRequest;
  try{const result=await api(`graph/${encodeURIComponent(target)}`);if(revision!==state.revision||target!==state.target||request!==state.skillRequest||!$("detail-dialog").open)return;
    $("skill-output").innerHTML=`<div class="detail-section"><h3>岗位需求关系</h3><div class="graph-root">${escape(result.title)}</div><div class="requirement-graph">${result.groups.map(group=>`<section>${renderRequirementTree(group.ast)}<small>原文：“${escape(group.evidence.quote)}”</small></section>`).join("")||'<p class="muted">没有可靠识别到要求，不能据此判断岗位没有要求。</p>'}</div><p class="footnote">${escape(result.notice)} 家族 ${escape((result.job_family_id||'').slice(0,12))} · ${escape(result.split)}。</p></div>`;
  }catch(error){toast(errorMessage(error));}
}
function renderTeacherRun(run) {
  const names={running:'正在标注',completed:'批次完成',incomplete:'待继续复核',failed:'执行失败',rate_limited:'接口限流，已暂停'};
  const wait=run.status==='rate_limited'?`<span class="tag warning">${Number(run.retry_wait_remaining_seconds)>0?`至少等待 ${Math.ceil(Number(run.retry_wait_remaining_seconds)/60)} 分钟后再恢复`:'冷却时间已到，可按运行手册低并发恢复'}</span> <span class="footnote">不会自动重复请求</span>`:'';
  return `<p>${run.task_kind==='relevance'?'人岗相关性':'需求抽取'}：${run.completed_tasks}/${run.tasks} · 规则通过 ${run.rule_passed} · ${escape(names[run.status]||'状态待核对')}<br><span class="footnote">${escape(run.model)} / ${escape(run.reasoning_effort)}</span> ${wait}</p>`;
}
async function loadResearch(){
  try{const data=await api("research");
    const teacher=data.teacher_study;$("teacher-study").innerHTML=teacher?.available?`<h3>接口标注与共同池研究</h3><p>${teacher.profiles}份合成画像 · ${teacher.pairs}个人岗对 · ${teacher.extraction_tasks}个需求片段</p><p class="footnote">${escape(teacher.scope)}</p>${teacher.annotation_runs.map(renderTeacherRun).join('')}${teacher.annotation_progress_stale?'<p class="alert">进度快照已过期，请先核对后台执行状态。</p>':''}<p class="footnote">人工评审：${teacher.human_reviewers}。模型双通道一致不等于人工一致，也不触发自动上线。</p>${Object.entries(teacher.metrics||{}).map(([name,value])=>`<details><summary>${escape(name)}研究指标</summary><pre>${escape(JSON.stringify(value,null,2))}</pre></details>`).join('')}`:`<p class="muted">${escape(teacher?.notice||'尚未配置接口标注研究材料。')}</p>`;
    $("research-metrics").innerHTML=[["岗位内容版本",data.jobs],["待审任务总数",data.task_count],["已标任务",data.annotated_tasks],["至少双人输入",data.double_annotated_tasks]].map(([title,value])=>`<div class="metric"><p class="metric-title">${escape(title)}</p><p class="metric-value">${number(value)}</p><p class="metric-foot">${escape(data.status)}</p></div>`).join("");
    $("research-runs").innerHTML=data.experiments.length?data.experiments.map(run=>`<details class="run-report"><summary>${escape(run.run)}</summary><pre>${escape(JSON.stringify(run.metrics,null,2))}</pre></details>`).join(""):'<p class="muted">尚无可读取的实验结果。训练完成后可刷新查看。</p>';
  }catch(error){toast(errorMessage(error));}
}
let activeTask=null,taskGeneration=0;
async function nextTask(){
  const annotator=$("annotator").value.trim(),kind=$("annotation-kind").value,generation=++taskGeneration;activeTask=null;
  if(!/^[\w\u4e00-\u9fff-]{2,40}$/.test(annotator)){toast("评审代号请使用2至40个汉字、字母、数字或下划线。");return;}
  $("annotation-task").innerHTML='<p class="loading">读取下一条…</p>';
  try{const result=await api(`research/task?annotator=${encodeURIComponent(annotator)}&kind=${encodeURIComponent(kind)}`);if(generation!==taskGeneration)return;
    if(!result.task){$("annotation-task").innerHTML='<p class="muted">当前没有未标任务，或评测材料尚未载入。</p>';return;}
    activeTask={task:result.task,annotator};const task=result.task;const relevant=kind==='relevance', extraction=kind==='requirement_extraction';
    $("annotation-task").innerHTML=`<p class="footnote">剩余 ${number(result.remaining)} 条 · ${escape(task.task_id)}</p>${task.query?`<h3>候选人材料</h3><pre class="annotation-text">${escape(task.query.text)}</pre><p class="footnote">条件：${escape(JSON.stringify(task.query.preferences))}</p>`:''}<h3>${escape(task.job?.title||'岗位片段')}</h3><p>${escape(task.job?.requirements||'')}</p><p class="footnote">薪资：${escape(task.job?.salary_raw||'未知')} · 地点：${escape(task.job?.address||'未知')}</p><pre class="annotation-text">${escape(relevant?task.job?.description:task.text||task.fragment||task.evidence?.quote||JSON.stringify(task,null,2))}</pre>${!relevant?`<details><summary>查看完整岗位上下文</summary><pre class="annotation-text">${escape(task.job?.description||'')}</pre></details>`:''}<div class="field"><label for="task-grade">${relevant?'相关性等级':extraction?'标注自检（正确表示已核对）':'核对结论'}</label><select id="task-grade"><option value="">请选择</option>${relevant?'<option value="0">0 · 明确冲突或职责无关</option><option value="1">1 · 方向相关，但关键能力无证据</option><option value="2">2 · 核心方向吻合，存在可解释缺口</option><option value="3">3 · 主要要求有证据且无已知硬冲突</option>':'<option>正确</option><option>错误</option><option>不确定</option>'}</select></div>${extraction?`<div class="field"><label for="task-extraction">需求结构标注（JSON）</label><textarea id="task-extraction" rows="7" maxlength="12000">{
  &quot;groups&quot;: []
}</textarea><p class="footnote">将每个独立要求放入groups数组，如 {&quot;logic&quot;:&quot;single&quot;,&quot;modality&quot;:&quot;required&quot;,&quot;skills&quot;:[技能项]}。混合要求使用 children 递归子组保留作用域（每个子组同结构），不要打平。逻辑：single 单项 / any 任选 / all 全部 / unknown 未明确。模态：required 必需 / preferred 优先 / negated 否定 / unknown 未明确。技能项示例：{&quot;skill&quot;:&quot;Python&quot;,&quot;quote&quot;:&quot;Python&quot;,&quot;occurrence&quot;:0}。quote 从本段复制，occurrence 为第几次出现（0 起）；服务端定位原字段跨度。无技能可保留空groups数组。</p></div>`:''}<div class="field"><label for="task-notes">判定依据 / 修正说明</label><textarea id="task-notes" maxlength="2000" rows="3" placeholder="引用支持或反对判断的原文，不填写联系方式。"></textarea></div><label class="consent-line"><input id="task-independent" type="checkbox"> 我已独立阅读材料，未查看其他评审或模型答案</label><button class="primary" id="save-task">保存并领取下一条</button><p class="footnote">未知条件不应自动判为满足；标注会保留修订记录，尚未仲裁为金标。</p>`;
    $("save-task").addEventListener("click",saveTask);
  }catch(error){$("annotation-task").innerHTML=`<div class="alert error">${escape(errorMessage(error))}</div>`;}
}
async function saveTask(){
  if(!activeTask)return;const value=$("task-grade").value;if(!value){toast("请选择判定结果。");return;}
  const button=$("save-task");button.disabled=true;const {task,annotator}=activeTask;
  try{await api("research/annotation",{task_id:task.task_id,task_hash:task.task_hash,independent:$("task-independent").checked,annotator,grade:$("annotation-kind").value==='relevance'?Number(value):null,decision:$("annotation-kind").value==='relevance'?'':value,notes:$("task-notes").value,extraction:$("task-extraction")?JSON.parse($("task-extraction").value):null});toast("已保存人工输入，尚未仲裁。");await nextTask();await loadResearch();}
  catch(error){toast(errorMessage(error));}finally{button.disabled=false;}
}
$("refresh-research").addEventListener("click",loadResearch);
$("next-task").addEventListener("click",nextTask);
for(const id of ["annotator","annotation-kind"])$(id).addEventListener("input",()=>{taskGeneration++;activeTask=null;$("annotation-task").innerHTML='<p class="muted">评审信息已变化，请重新领取任务。</p>';});
document.querySelectorAll("[data-view]").forEach(button=>button.addEventListener("click",()=>view(button.dataset.view)));
document.querySelectorAll("[data-go]").forEach(button=>button.addEventListener("click",()=>view(button.dataset.go)));
$("start-match").addEventListener("click",()=>{if($("category").value){$("intent").value=$("category").value;$("sample").value="";invalidate();toast("已带入当前职位方向，请选择或调整对应的简历经历。");}view("match");$("sample").focus();});
document.querySelector(".brand").addEventListener("click",event=>{event.preventDefault();view("demand");});
window.addEventListener("hashchange",()=>view(location.hash.slice(1)||"demand"));
$("choose-file").addEventListener("click",()=>$("resume-file").click());
$("category").addEventListener("change",loadOverview);
$("sample").addEventListener("change",()=>applySample($("sample").value));
$("resume-form").addEventListener("submit",recommend);
for(const id of ["resume-text",...fields.map(([id])=>id),"strict","query-mode","research-consent"]) $(id).addEventListener("input",()=>{if(id==="resume-text")clearSamplePreferences();const pair=fields.find(pair=>pair[0]===id);if(pair)state.origins[pair[1]]="user_input";$("sample").value="";invalidate();});
$("resume-file").addEventListener("change",async event=>{
  const file=event.target.files[0];if(!file)return;
  if(file.size>2*1024*1024||!/\.(txt|pdf|docx)$/i.test(file.name)){toast("请选择2MB以内的TXT、文字型PDF或DOCX简历。");return;}
  const revision=state.revision;const data=new FormData();data.append("file",file);
  $("choose-file").disabled=true;
  try{const response=await fetch("/api/v2/document",{method:"POST",body:data,credentials:"same-origin"});const result=await response.json();if(!response.ok)throw new Error(result.detail||"简历解析失败");if(revision!==state.revision){toast("编辑内容已变化，未覆盖当前简历，请重新导入。");return;}clearSamplePreferences();$("resume-text").value=result.text;$("sample").value="";invalidate();toast("已导入简历，样例条件已清除；请解析并确认画像。");}
  catch(error){toast(errorMessage(error));}finally{$("choose-file").disabled=false;event.target.value="";}
});
$("results").addEventListener("click",async event=>{const question=event.target.closest("[data-question-control]");if(question){const control=$(question.dataset.questionControl);if(control){control.scrollIntoView({behavior:"smooth",block:"center"});control.focus();}return;}const excluded=event.target.closest("[data-exclude]");if(excluded){try{await api("preferences/company",{job_id:excluded.dataset.exclude,exclude:true});await loadJourney();await recommend();toast("该公司已从后续推荐排除，可在偏好记录中撤销。");}catch(error){toast(errorMessage(error));}return;}const detail=event.target.closest("[data-detail]");if(detail){openDetail(detail.dataset.detail);return;}const button=event.target.closest("[data-feedback]");if(!button||!state.result)return;button.disabled=true;try{await api("feedback",{run_id:state.result.run_id,job_id:button.dataset.job,action:button.dataset.feedback});button.parentElement.querySelectorAll(".feedback").forEach(item=>item.classList.toggle("selected",item===button));toast("已记录本次反馈；明确排除公司可使用“不考虑该公司”，跳过不直接作为负例。");}catch(error){toast(errorMessage(error));}finally{button.disabled=false;}});
$("close-dialog").addEventListener("click",()=>$("detail-dialog").close());
$("detail-dialog").addEventListener("close",()=>{state.detailRequest+=1;state.skillRequest+=1;state.rewritten=null;});
$("detail-dialog").addEventListener("click",event=>{if(event.target===$("detail-dialog")){const bounds=$("detail-dialog").getBoundingClientRect();if(event.clientX<bounds.left||event.clientX>bounds.right||event.clientY<bounds.top||event.clientY>bounds.bottom)$("detail-dialog").close();}});
$("run-comparison").addEventListener("click",compare);
$("clear-feedback").addEventListener("click",async()=>{try{await api("feedback",null,"DELETE");invalidate();toast("本会话的曝光和偏好记录已清除，请重新匹配后继续反馈。");}catch(error){toast(errorMessage(error));}});
async function init() {
  try { const [overview,samples]=await Promise.all([api("overview"),api("samples")]);state.overview=overview;state.samples=samples.samples;
    $("category").innerHTML='<option value="">全部方向</option>'+overview.category_options.map(item=>`<option value="${escape(item.name)}">${escape(item.name)} · ${number(item.count)}</option>`).join("");renderOverview(overview);
    $("sample").innerHTML='<option value="">自定义简历</option>'+state.samples.map(sample=>`<option value="${sample.id}">${escape(sample.name)}</option>`).join("");$("sample").value="";invalidate();
    view(location.hash.slice(1)||"demand");
    registerAgentTools();
    await loadJourney();
  }catch(error){$("global-error").textContent=`岗位库读取失败：${errorMessage(error)} 请确认后端已启动并可读取数据文件，然后刷新页面。`;$("global-error").classList.remove("hidden");}
}
function registerAgentTools() {
  const context=document.modelContext;
  if(!context?.registerTool)return;
  const lifecycle=new AbortController();
  const validInput=(input,allowed)=>input!==null&&typeof input==="object"&&!Array.isArray(input)&&Object.keys(input).every(key=>allowed.includes(key));
  window.addEventListener("pagehide",()=>lifecycle.abort(),{once:true});
  const tools=[{
    name:"read_job_demand",title:"读取岗位需求",description:"读取指定职位方向的聚合需求，不修改简历或提交反馈。",
    inputSchema:{type:"object",properties:{category:{type:"string"}},additionalProperties:false},
    annotations:{readOnlyHint:true,untrustedContentHint:true},
    async execute(input){if(!validInput(input,["category"])||(input.category!==undefined&&typeof input.category!=="string"))throw new Error("仅接受可选字符串category");const result=await api(`overview?category=${encodeURIComponent(input.category||"")}`);return {total:result.total,skills:result.skills.slice(0,8),salary_quantiles:result.salary_quantiles,snapshot:result.snapshot};}
  },{
    name:"select_resume_sample",title:"选择匿名简历样例",description:"将指定的虚构样例载入可见简历编辑器；不会自动开始推荐或应用改写。",
    inputSchema:{type:"object",properties:{sample_id:{type:"string",enum:state.samples.map(sample=>sample.id)}},required:["sample_id"],additionalProperties:false},
    annotations:{readOnlyHint:false,untrustedContentHint:false},
    execute(input){if(!validInput(input,["sample_id"]))throw new Error("仅接受sample_id");const sample=state.samples.find(item=>item.id===input.sample_id);if(!sample)throw new Error("未知的简历样例");$("sample").value=sample.id;applySample(sample.id);view("match");return {sample_id:sample.id,name:sample.name,intent:sample.preferences.intent};}
  },{
    name:"run_resume_match",title:"匹配当前简历",description:"使用编辑器中的简历与偏好完成推荐，更新可见结果并在本机记录岗位曝光；不会保存简历正文。",
    inputSchema:{type:"object",properties:{},additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:true},
    async execute(input){if(!validInput(input,[]))throw new Error("此工具不接受参数");if($("recommend").disabled)throw new Error("已有推荐正在运行");if(!$("resume-form").reportValidity())throw new Error("请先检查表单输入");view("match");await recommend();if(!state.confirmation)return {action:"needs_confirmation",message:"请在可见画像卡中确认字段来源和冲突，再执行匹配。"};if(!state.result)throw new Error("匹配未完成，请检查输入或错误提示");return {action:state.result.action,run_id:state.result.run_id,questions:state.result.questions,jobs:state.result.jobs.map(job=>({id:job.id,title:job.title,score:job.score}))};}
  }];
  for(const tool of tools){try{Promise.resolve(context.registerTool(tool,{signal:lifecycle.signal})).catch(()=>{});}catch{/* 不支持注册时保留完整手动流程。 */}}
}
async function startAction(groupId, button){
  if(!state.payload||!state.target)return;
  const revision=state.revision,target=state.target,payload={...state.payload};button.disabled=true;
  try{
    const action=await api("journey/actions",{...payload,job_id:target,group_id:groupId});
    if(revision!==state.revision||target!==state.target||!$("detail-dialog").open)return;
    $("skill-output").innerHTML=`<div class="detail-section"><h3>补充真实经历：${escape(action.label)}</h3><p class="footnote">写明个人职责、实际操作与结果。如果没有做过，请先完成练习，暂不加入简历。</p><div class="field"><label for="action-evidence">准备加入简历的原文</label><textarea id="action-evidence" rows="5" minlength="15" maxlength="3000" placeholder="例如：我使用……完成……，负责……，通过……检查结果。">${escape(action.evidence||'')}</textarea></div><label class="checkbox"><input type="checkbox" id="save-evidence-consent">同意在本机保存这条证据30天，随时可清除</label><button class="primary" id="save-evidence">保存单条证据</button><div id="confirm-evidence-panel"></div></div>`;
    $("save-evidence").addEventListener("click",async()=>{
      if(revision!==state.revision)return;const save=$("save-evidence");save.disabled=true;
      try{
        const record=await api("journey/evidence",{action_id:action.id,evidence:$("action-evidence").value,consent_to_store:$("save-evidence-consent").checked});
        if(revision!==state.revision||!$("detail-dialog").open)return;
        $("confirm-evidence-panel").innerHTML=`<div class="quote-box"><strong>将追加的原文</strong><p>${escape(record.evidence)}</p><p class="footnote">此操作保留原简历，追加这一条经历。请确认内容确实来自自己的实践。</p><button class="secondary" id="confirm-evidence">确认属实，加入简历并复核</button></div>`;
        $("confirm-evidence").addEventListener("click",async()=>{
          const confirm=$("confirm-evidence");confirm.disabled=true;
          try{
            const result=await api("journey/confirm-evidence",{...payload,job_id:target,action_id:action.id,revised_text:payload.text.trimEnd()+"\n\n"+record.evidence,user_confirmed:true});
            if(revision!==state.revision)return;
            $("detail-dialog").close();$("resume-text").value=result.revised_text;$("sample").value="";invalidate();await loadJourney();await previewProfile();
            toast(result.closed_gaps.length?`新证据支持了 ${result.closed_gaps.length} 项原先待确认的要求，请确认新画像。`:"证据已记录，相关要求仍需继续核对，请确认新画像。");
          }catch(error){toast(errorMessage(error));confirm.disabled=false;}
        });
        await loadJourney();
      }catch(error){toast(errorMessage(error));}finally{save.disabled=false;}
    });
  }catch(error){toast(errorMessage(error));}finally{button.disabled=false;}
}
let journeyRevision=0;
async function loadJourney(){
  const revision=++journeyRevision;
  try{
    const [history,prefs]=await Promise.all([api("journey"),api("preferences")]);if(revision!==journeyRevision)return;
    $("company-preferences").innerHTML=prefs.excluded_companies.length?`<h3>暂不考虑的公司</h3>${prefs.excluded_companies.map(item=>`<div class="preference-row"><span>${escape(item.company)}</span><button class="text-button" data-undo-company="${escape(item.job_id)}">撤销排除</button></div>`).join("")}`:'<p class="footnote">尚未设置公司排除偏好。</p>';
    $("company-preferences").querySelectorAll("[data-undo-company]").forEach(button=>button.addEventListener("click",async()=>{try{await api("preferences/company",{job_id:button.dataset.undoCompany,exclude:false});await loadJourney();if(state.confirmation)await recommend();}catch(error){toast(errorMessage(error));}}));
    const names={pending:"待补证",evidence_submitted:"已保存，待确认",confirmed:"用户已确认"};
    $("journey-history").innerHTML=history.actions.length?history.actions.map(action=>`<div class="journey-item"><div><strong>${escape(action.label)}</strong><span class="tag ${action.status==='confirmed'?'positive':'warning'}">${escape(names[action.status]||action.status)}</span></div><p class="footnote">画像 ${escape(action.profile_version)} · ${new Date(action.updated*1000).toLocaleString('zh-CN')}</p>${action.evidence?`<details><summary>查看本人提交的证据</summary><p>${escape(action.evidence)}</p></details>`:''}<button class="text-button" data-journey-job="${escape(action.job_id)}">回到目标岗位核对</button></div>`).join(""):'<p class="muted">还没有补证行动。从岗位差距中选择一项开始。</p>';
    $("journey-history").innerHTML+=history.versions.map(version=>`<div class="version-change"><strong>版本复核</strong><p>${escape(version.summary.before_version)} → ${escape(version.summary.after_version)}</p><p>新增技能表述：${escape(version.summary.added_skill_statements.join('、')||'无')}；已获得支持的要求：${escape(version.summary.closed_gaps.join('、')||'无')}。</p><p class="footnote">${version.summary.remaining_gaps.length} 项仍待核对。用户确认不等于外部能力认证。</p></div>`).join("");
    $("journey-history").querySelectorAll("[data-journey-job]").forEach(button=>button.addEventListener("click",()=>{if(state.payload)openDetail(button.dataset.journeyJob);else{view("match");toast("先载入并确认对应的简历版本，再继续岗位补证。");}}));
  }catch(error){toast(errorMessage(error));}
}
$("refresh-journey").addEventListener("click",loadJourney);
$("delete-journey").addEventListener("click",async()=>{try{await api("journey",null,"DELETE");await loadJourney();toast("已清除本会话补证、版本摘要和研究记录。");}catch(error){toast(errorMessage(error));}});
init();
