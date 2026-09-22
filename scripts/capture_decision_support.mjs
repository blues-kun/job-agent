/** 使用本机 Chromium 调试接口验收真实页面；无需额外 npm 依赖。 */
import {mkdir, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';

const options = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, args) => {
  if (index % 2 === 0) pairs.push([value, args[index + 1]]);
  return pairs;
}, []));
const base = options['--url'] || 'http://127.0.0.1:8096';
const debug = options['--debug-url'] || 'http://127.0.0.1:9229';
for (const value of [base, debug]) {
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(new URL(value).hostname)) throw Error('仅验收本机页面');
}
if (typeof WebSocket !== 'function') throw Error('需要支持 WebSocket 的 Node；Node 20 请添加 --experimental-websocket');
const output = resolve(options['--output-dir'] || 'docs/screenshots/decision-support');
await mkdir(output, {recursive: true});
const target = await (await fetch(`${debug}/json/new?${encodeURIComponent('about:blank')}`, {method: 'PUT'})).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((accept, reject) => {ws.addEventListener('open', accept, {once: true}); ws.addEventListener('error', reject, {once: true});});
let sequence = 0;
const pending = new Map(), errors = [], steps = [], captures = [];
ws.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    const {accept, reject, timer} = pending.get(message.id); clearTimeout(timer); pending.delete(message.id);
    message.error ? reject(Error(message.error.message)) : accept(message.result);
  }
  if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
});
function send(method, params = {}) {
  const id = ++sequence;
  return new Promise((accept, reject) => {
    const timer = setTimeout(() => {pending.delete(id); reject(Error(`${method}超时`));}, 45000);
    pending.set(id, {accept, reject, timer}); ws.send(JSON.stringify({id, method, params}));
  });
}
async function run(expression) {
  const result = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
  if (result.exceptionDetails) throw Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  return result.result.value;
}
async function wait(expression) {
  return run(`(async()=>{const until=Date.now()+40000;while(!(${expression})){if(Date.now()>until)throw Error('页面状态未就绪');await new Promise(r=>setTimeout(r,150));}return true;})()`);
}
async function screenshot(name, caption) {
  await run('document.fonts.ready.then(()=>true)');
  const image = Buffer.from((await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false})).data, 'base64');
  await writeFile(resolve(output, name), image);
  captures.push({file: name, caption, sha256: createHash('sha256').update(image).digest('hex')});
}
try {
  await send('Page.enable'); await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false});
  await send('Page.navigate', {url: base});
  await wait("document.querySelector('#sample option[value=\"python-junior\"]')");
  await run(`document.querySelector('[data-view="match"]').click();document.getElementById('sample').value='python-junior';document.getElementById('sample').dispatchEvent(new Event('change'));document.getElementById('recommend').click();`);
  await wait("document.getElementById('confirm-profile')");
  await run(`const ack=document.getElementById('ack-conflicts');if(ack){ack.checked=true;ack.dispatchEvent(new Event('change'));}document.getElementById('confirm-profile').click();`);
  await wait("document.querySelector('.job-card') && document.querySelector('.decision-panel')");
  steps.push('画像解析与确认', '公开岗位库实际推荐', '关键追问与证据区间展示');
  await run(`document.querySelector('.decision-comparison').open=true;document.getElementById('results').scrollIntoView({block:'start'});`);
  await screenshot('08-decision.png', '关键追问与候选取舍比较');
  await run(`(async()=>{let selected=state.result.jobs[0].id;for(const job of state.result.jobs){const detail=await api('diagnose',{...state.payload,job_id:job.id});if(detail.action_plans.plans.some(plan=>plan.actions.length>0)){selected=job.id;break;}}document.querySelector('[data-detail="'+selected+'"]').click();})()`);
  await wait("document.querySelector('.action-plans')");
  await run(`document.querySelector('.action-plans').scrollIntoView({block:'center'});`);
  await screenshot('09-actions.png', '岗位要求的条件式补证路线');
  steps.push('真实诊断接口与补证路线展示');
  await run(`document.getElementById('close-dialog').click();document.getElementById('education-full-time').value='true';document.getElementById('education-full-time').dispatchEvent(new Event('input'));`);
  const cleared = await run(`state.confirmation==='' && state.origins.education_full_time==='user_input' && state.result===null`);
  if (!cleared) throw Error('全日制回填未撤销旧确认');
  await run(`document.getElementById('recommend').click();`);
  await wait("document.getElementById('confirm-profile')");
  if (!(await run(`state.preview.fields.some(field=>field.key==='education_full_time'&&field.value===true&&field.source==='user_input')`))) throw Error('全日制回答未进入确认画像');
  steps.push('追问回填后撤销旧凭据并重新确认');
  await send('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
  const overflow = await run(`document.documentElement.scrollWidth > innerWidth + 2`);
  if (overflow) throw Error('移动端页面横向溢出');
  steps.push('390像素移动端无横向溢出');
  if (errors.length) throw Error('页面出现未捕获异常');
  const report = {executed_at: new Date().toISOString(), scope: '内置合成简历与公开岗位库的浏览器工程验收',
    health: await (await fetch(`${base}/api/v2/health`)).json(), steps, errors, captures,
    human_outcome_evaluation: false};
  await writeFile(resolve(output, 'browser-verification.json'), JSON.stringify(report, null, 2)+'\n');
  console.log(JSON.stringify({steps, errors, captures: captures.length, output}));
} finally {
  // 清理本次浏览器会话留下的临时反馈，不影响其他会话或用户数据。
  try {await run(`Promise.all(['feedback','journey'].map(path=>fetch('/api/v2/'+path,{method:'DELETE'}))).then(results=>results.every(r=>r.ok))`);} catch {}
  ws.close();
  await fetch(`${debug}/json/close/${target.id}`).catch(() => {});
}
