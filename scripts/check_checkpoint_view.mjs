import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const [endpoint,out,...files]=process.argv.slice(2);
if(!files.length)throw new Error('Usage: node check_checkpoint_view.mjs CDP_URL report.json view.html ...');
const page=await(await fetch(endpoint+'/json/new?about:blank',{method:'PUT'})).json();
const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r));
let id=0;const pending=new Map(),errors=[];
ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);});
const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
await call('Page.enable');await call('Runtime.enable');
const reports=[];
for(const file of files){
 const expected=JSON.parse(await fs.readFile(file.replace(/\.html$/,'.audit.json'),'utf8'));
 await call('Emulation.setDeviceMetricsOverride',{width:1660,height:1080,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:pathToFileURL(file).href});
 for(let i=0;i<150;i++){if(await evaluate(`Boolean(window.DRREM_CHECKPOINT_VIEW)&&window.DRREM_CHECKPOINT_VIEW.state.checkpoint===${JSON.stringify(expected.provenance.checkpoint_sha256)}`))break;await new Promise(r=>setTimeout(r,200));}
 const state=await evaluate('window.DRREM_CHECKPOINT_VIEW.state');assert.equal(state.nodes,3072);
 assert.equal(state.checkpoint,expected.provenance.checkpoint_sha256);
 const audit=await evaluate(`(()=>{
  let max=0,cases=0;
  if(PK){for(const h of [1,H])for(const t of [0,T-1])for(let p=0;p<C.paths;p++)for(let d=0;d<C.width;d++){
   max=Math.max(max,Math.abs(at(TR.after,h-1,t,p,d)-at(TR.before,h-1,t,p,d)-S*(at(TR.attention,h-1,t,p,d)+at(TR.mlp,h-1,t,p,d))));cases++;
  }}else{for(const h of [1,H])for(const t of [0,T-1])for(let l=0;l<3;l++){
    hop=h;position=t;layer=l;neuron=17;
    for(const key of Object.keys(P.edges))if(+key.split('_')[0]===l){let sum=0;for(let j=0;j<N;j++)sum+=current(key,neuron,j);const bi=Object.keys(P.edges).indexOf(key);max=Math.max(max,Math.abs(sum-S*at(TR.messages,h-1,bi,t,neuron)));cases++;}
   }}
  hop=H;position=T-1;follow();update();return {max_current_error:max,cases};
 })()`);assert(audit.max_current_error<2e-5);
 const output=await evaluate("$('outputs').innerText");
 await evaluate("$('hop').value=1;$('hop').dispatchEvent(new Event('input'))");
 assert.equal(await evaluate("$('outputs').innerText"),output);
 await evaluate("$('position').value=0;$('position').dispatchEvent(new Event('input'))");
 assert.equal(await evaluate("$('attention').innerText.includes('BOS')"),true);
 assert.equal(await evaluate("$('phase').innerText.includes('Спайков, мембранного потенциала')"),true);
 await evaluate("$('position').value=T-1;$('position').dispatchEvent(new Event('input'));$('hop').value=H;$('hop').dispatchEvent(new Event('input'))");
 if(state.kind==='dense'){
  for(const metric of ['delta','spatial','attention','mlp','bridge','states'])await evaluate(`$('metric').value='${metric}';$('metric').dispatchEvent(new Event('change'))`);
  await evaluate("$('direction').value='out';$('direction').dispatchEvent(new Event('change'))");
  await evaluate("$('connections').querySelector('[data-edge]').click()");
  assert((await evaluate("$('edgeDetail').innerText")).length>30);
 }else{
  await evaluate("$('dimension').value=C.width-1;$('dimension').dispatchEvent(new Event('change'));$('path').value=C.paths-1;$('path').dispatchEvent(new Event('change'));$('historyPath').checked=false;$('historyPath').dispatchEvent(new Event('change'))");
  assert.equal(await evaluate('dimension'),127);
 }
 const box=await evaluate("(()=>{const b=canvas.getBoundingClientRect();return {x:b.x+b.width/2,y:b.y+b.height/2};})()");
 await call('Input.dispatchMouseEvent',{type:'mousePressed',...box,button:'left',clickCount:1});
 await call('Input.dispatchMouseEvent',{type:'mouseMoved',x:box.x+60,y:box.y+30,buttons:1});
 await call('Input.dispatchMouseEvent',{type:'mouseReleased',x:box.x+60,y:box.y+30,button:'left'});
 assert(Math.abs((await evaluate('yaw'))+.35)>.1);
 await evaluate("$('home').click()");
 const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(file.replace(/\.html$/,'.png'),Buffer.from(shot.data,'base64'));
 await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:false});await new Promise(r=>setTimeout(r,100));
 const mobile=await evaluate('({viewport:innerWidth,scroll:document.documentElement.scrollWidth})');assert(mobile.scroll<=mobile.viewport);
 reports.push({file,sha256:createHash('sha256').update(await fs.readFile(file)).digest('hex'),state,audit,mobile});
 console.log(JSON.stringify(reports.at(-1)));
}
assert.equal(errors.length,0);await fs.writeFile(out,JSON.stringify({reports,errors},null,2)+'\n');ws.close();
