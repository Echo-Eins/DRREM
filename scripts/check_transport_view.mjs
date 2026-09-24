import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const cdp=process.argv[2]||'http://127.0.0.1:19323';
// Start an isolated Chrome with --headless=new --remote-debugging-port=19323.
// Uses Node's built-in WebSocket (Node 22+), with no npm dependencies.

const page=await(await fetch(cdp+'/json/new?about:blank',{method:'PUT'})).json();
const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise(resolve=>ws.addEventListener('open',resolve));
let id=0;const pending=new Map(),errors=[];
ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);});
const call=(method,params={})=>new Promise((resolve,reject)=>{const mid=++id;pending.set(mid,{resolve,reject});ws.send(JSON.stringify({id:mid,method,params}));});
const evaluate=async(expression)=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
await call('Page.enable');await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:1660,height:1080,deviceScaleFactor:1,mobile:false});
await call('Page.navigate',{url:new URL('../docs/transport_3d.html',import.meta.url).href});
for(let i=0;i<100;i++){if(await evaluate('Boolean(window.DRREM_VIEW)'))break;await new Promise(r=>setTimeout(r,200));}
const numerical=await evaluate(`(()=>{
 const report={cases:0,max_current_sum_error:0,max_mlp_sum_error:0,variants:[]};
 for(const v of DATA.variants){
   $('variant').value=v.id;$('variant').dispatchEvent(new Event('change'));
   for(const h of [1,4,8]){
     $('hop').value=h;$('hop').dispatchEvent(new Event('input'));
     const sums=Object.fromEntries(DATA.edge_keys.map(k=>[k,new Float64Array(N)]));
     const bridgeSum=Array.from({length:L},()=>new Float64Array(N));
     for(const e of allEdges){if(e.kind==='bridge')bridgeSum[e.target][e.i]+=e.current;else sums[e.key][e.i]+=e.current;}
     for(let b=0;b<DATA.edge_keys.length;b++)for(let i=0;i<N;i++)report.max_current_sum_error=Math.max(report.max_current_sum_error,Math.abs(sums[DATA.edge_keys[b]][i]-S*at(v.trace.messages,h-1,b,position,i)));
     for(let l=0;l<L;l++)for(let i=0;i<N;i++)report.max_current_sum_error=Math.max(report.max_current_sum_error,Math.abs(bridgeSum[l][i]-at(v.trace.bridge,h-1,l,position,i)));
     for(let l=0;l<L;l++)for(let i=0;i<N;i++){
       let sum=0;for(let j=0;j<N*2;j++){const g=at(v.trace.gate,h-1,l,position,j),val=at(v.trace.value,h-1,l,position,j);sum+=S*at(DATA.shared.mlp_down,l,i,j)*g/(1+Math.exp(-g))*val;}
       report.max_mlp_sum_error=Math.max(report.max_mlp_sum_error,Math.abs(sum-S*at(v.trace.mlp,h-1,l,position,i)));
     }
     report.cases++;
   }
   $('zeros').checked=true;$('scope').value='all';$('scope').dispatchEvent(new Event('change'));
   report.variants.push({...window.DRREM_VIEW.state,expected:7*N*N+Object.keys(v.parameters.bridges).length*N});
 }
 return report;
})()`);
assert(numerical.max_current_sum_error<2e-6);assert(numerical.max_mlp_sum_error<2e-6);
for(const row of numerical.variants)assert.equal(row.renderedEdges,row.expected);
for(const row of numerical.variants){
 assert.equal(row.scales.mode,'shared');
 assert.deepEqual(row.scales,numerical.variants[0].scales);
}
await evaluate("$('scaleMode').value='auto';$('scaleMode').dispatchEvent(new Event('change'))");
assert.equal(await evaluate("$('legend').innerText.includes('несопоставима')"),true);
await evaluate("$('scaleMode').value='shared';$('scaleMode').dispatchEvent(new Event('change'))");
assert.deepEqual(await evaluate('window.DRREM_VIEW.state.scales'),numerical.variants[0].scales);
const routes=[];
for(const preset of ['input','bridge','mix','return']){
 await evaluate(`document.querySelector('[data-preset="${preset}"]').click()`);
 routes.push({preset,state:await evaluate('window.DRREM_VIEW.state'),detail:await evaluate("$('edgeDetail').innerText")});
}
assert.equal(routes[0].state.hop,0);assert.equal(routes[1].state.layer,2);assert.equal(routes[1].state.hop,1);assert(routes[1].detail.includes('L1[17] → L3[17]'));assert(routes[2].detail.includes('L3[17] → L3[73]'));assert(routes[3].detail.includes('L3[17] → L2[73]'));
await evaluate(`document.querySelector('[data-tab="attention"]').click();$('head').value=7;$('head').dispatchEvent(new Event('change'));document.querySelector('[data-position="0"]').click()`);
assert.equal(await evaluate("Array.from($('attentionBars').querySelectorAll('.barline')).every(e=>e.lastChild.textContent==='0.000')"),true);
await evaluate(`document.querySelector('[data-tab="output"]').click();document.querySelector('[data-position="13"]').click()`);
const decoded=await evaluate("$('outputLogits').innerText");await evaluate("$('hop').value=1;$('hop').dispatchEvent(new Event('input'))");assert.equal(await evaluate("$('outputLogits').innerText"),decoded);
await evaluate(`document.querySelector('[data-tab="code"]').click()`);assert.equal(await evaluate("document.querySelectorAll('#sources details').length"),7);
await evaluate(`$('variant').value='bridges_open';$('variant').dispatchEvent(new Event('change'));$('scope').value='star';$('scope').dispatchEvent(new Event('change'));$('intra').checked=true;$('adjacent').checked=true;$('bridges').checked=true;$('zeros').checked=false;$('intra').dispatchEvent(new Event('change'));$('hop').value=4;$('hop').dispatchEvent(new Event('input'));document.querySelector('[data-tab="state"]').click();$('home').click()`);
await new Promise(r=>setTimeout(r,100));
const box=await evaluate(`(()=>{const b=$('graph').getBoundingClientRect();return {x:b.x,y:b.y,width:b.width,height:b.height};})()`);
const x=box.x+box.width*.5,y=box.y+box.height*.5;
await call('Input.dispatchMouseEvent',{type:'mousePressed',x,y,button:'left',clickCount:1});
await call('Input.dispatchMouseEvent',{type:'mouseMoved',x:x+90,y:y+45,button:'left',buttons:1});
await call('Input.dispatchMouseEvent',{type:'mouseReleased',x:x+90,y:y+45,button:'left',clickCount:1});
const camera=await evaluate('window.DRREM_VIEW.state.camera');assert(Math.abs(camera.yaw+.38)>.2);
await call('Input.dispatchMouseEvent',{type:'mouseWheel',x,y,deltaX:0,deltaY:80});await new Promise(r=>setTimeout(r,100));assert((await evaluate('window.DRREM_VIEW.state.camera.zoom'))<1);
await evaluate("$('home').click();$('layer').value=2;$('layer').dispatchEvent(new Event('change'));$('coordinate').value=42;$('coordinate').dispatchEvent(new Event('change'))");
await evaluate("$('play').click()");const previous=await evaluate('window.DRREM_VIEW.state.hop');await new Promise(r=>setTimeout(r,1500));assert.notEqual(await evaluate('window.DRREM_VIEW.state.hop'),previous);await evaluate("$('play').click();$('hop').value=4;$('hop').dispatchEvent(new Event('input'))");
await new Promise(r=>setTimeout(r,200));
const screenshot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(root+'docs/transport_3d_preview.png',Buffer.from(screenshot.data,'base64'));
await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:false});await new Promise(r=>setTimeout(r,300));
const mobile=await evaluate('({viewport:innerWidth,scroll:document.documentElement.scrollWidth,state:window.DRREM_VIEW.state})');
const mobileShot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile('/tmp/drrem-transport-mobile.png',Buffer.from(mobileShot.data,'base64'));
await call('Emulation.setDeviceMetricsOverride',{width:1660,height:1080,deviceScaleFactor:1,mobile:false});
await new Promise(r=>setTimeout(r,200));
const point=await evaluate("(()=>{const b=cv.getBoundingClientRect(),p=projection[nodeId(1,80)];return {x:b.x+p.x,y:b.y+p.y};})()");
await call('Input.dispatchMouseEvent',{type:'mousePressed',...point,button:'left',clickCount:1});
await call('Input.dispatchMouseEvent',{type:'mouseReleased',...point,button:'left',clickCount:1});
assert.equal(await evaluate('window.DRREM_VIEW.state.layer'),1);assert.equal(await evaluate('window.DRREM_VIEW.state.neuron'),80);
// A fresh directory prevents an earlier screenshot from satisfying this test.
const download=await fs.mkdtemp('/tmp/drrem-transport-export-');
await call('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:download});
await evaluate("$('exportPng').click()");
let exported;
for(let i=0;i<100;i++){
 exported=(await fs.readdir(download)).find(f=>f.endsWith('.png'));
 if(exported)break;
 await new Promise(r=>setTimeout(r,100));
}
assert(exported);const png=await fs.readFile(download+'/'+exported);
const expectedPng=await evaluate('({width:cv.width,height:Math.floor(cv.height+118*cv.width/width)})');
assert.equal(png.readUInt32BE(16),expectedPng.width);assert.equal(png.readUInt32BE(20),expectedPng.height);
assert(png.length>10000);
const result={artifact_sha256:createHash('sha256').update(await fs.readFile(root+'docs/transport_3d.html')).digest('hex'),browser:await call('Browser.getVersion'),numerical,routes,interaction:{shared_brightness_scales:true,auto_scale_warning:true,pointer_neuron_selection:true,png_export:{...expectedPng,bytes:png.length,includes_provenance_and_brightness_band:true},rotation:true,zoom:true,hop_playback:true,attention_bos_zero:true,final_decoder_independent_of_viewed_hop:true},mobile,errors};
await fs.writeFile(root+'docs/transport_3d.browser.json',JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({numerical:result.numerical,interaction:result.interaction,mobile:{viewport:mobile.viewport,scroll:mobile.scroll},errors},null,2));assert.equal(errors.length,0);assert(mobile.scroll<=mobile.viewport);ws.close();
