
function cleanSummary(v){
  let s=String(v||'');
  const box=document.createElement('textarea');
  for(let i=0;i<3;i++){
    box.innerHTML=s;
    const decoded=box.value;
    if(decoded===s)break;
    s=decoded;
  }
  const tmp=document.createElement('div');
  tmp.innerHTML=s;
  s=(tmp.textContent||tmp.innerText||'').replace(/\s+/g,' ').trim();
  s=s.replace(/^\([A-Za-z]-?\d{3,}\)\s*/,'');
  return s;
}
function summarize3Lines(v){
  const s=cleanSummary(v);
  if(!s)return '';
  const parts=s.split(/(?<=[.!?])\s+|\s+[•|]\s+/).map(x=>x.trim()).filter(x=>x.length>=30);
  const boiler=/^(about (the )?(team|role|company)|who we are|what we do|equal opportunity|benefits|compensation|pay range)\b/i;
  const keys=['responsib','lead','build','manage','drive','develop','deliver','partner','design','architect','experience','skills','qualif','customer','platform','strategy','data','engineering','program'];
  const ranked=parts.map((x,i)=>({x,i,score:keys.reduce((n,k)=>n+(x.toLowerCase().includes(k)?1:0),0)-(boiler.test(x)?2:0)})).sort((a,b)=>b.score-a.score||a.i-b.i);
  const chosen=[];
  for(const r of ranked){if(chosen.length===3)break;if(chosen.some(c=>c.x===r.x))continue;chosen.push(r)}
  const out=(chosen.length?chosen.sort((a,b)=>a.i-b.i).map(r=>r.x):parts.slice(0,3)).slice(0,3).map(x=>x.length>155?x.slice(0,152).replace(/\s+\S*$/,'')+'…':x);
  return out.join('\n');
}
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];let jobs=[],picked=new Set();
async function api(url,opt={}){let r=await fetch(url,{headers:{'Content-Type':'application/json'},...opt}),d=await r.json();if(!r.ok)throw Error(d.detail||'Something went wrong');return d}function notice(x,b=0){let n=$('#notice');n.className=b?'bad':'good';n.id='notice';n.textContent=x;setTimeout(()=>n.className='',5000)}function esc(x=''){let d=document.createElement('div');d.textContent=x;return d.innerHTML}function count(){$('#selected').textContent=picked.size}
function cards(){let box=$('#jobs');box.innerHTML='';$('#count').textContent=`${jobs.length} opening${jobs.length==1?'':'s'}`;$('#no-jobs').style.display=jobs.length?'none':'block';jobs.forEach(j=>{let n=$('#card').content.cloneNode(true),l=n.querySelector('.logo'),b=n.querySelector('input');l.textContent=j.organization[0];if(j.logo_url)l.style.backgroundImage=`url('${j.logo_url}')`;n.querySelector('.org').textContent=j.organization;n.querySelector('h3').textContent=j.title;n.querySelector('.meta').textContent='Job ID: '+j.job_id;let loc=n.querySelector('.location');loc.textContent=j.location?'⌖ '+j.location:'⌖ Location not supplied';loc.classList.toggle('missing',!j.location);n.querySelector('.summary').textContent=j.summary||summarize3Lines(j.description)||'No description was supplied by the careers source.';let a=n.querySelector('a');a.href=j.source_url;n.querySelector('time').textContent=new Date(j.posted_at).toLocaleDateString();b.checked=picked.has(j.id);b.onchange=()=>{b.checked?picked.add(j.id):picked.delete(j.id);count()};box.append(n)})}
const SOURCE_LABELS={greenhouse:'Greenhouse',lever:'Lever',google_careers:'Google Careers',google:'Google Careers',generic:'Other careers site'};
const KNOWN_BOARDS={databricks:{source_type:'greenhouse',board_token:'databricks'}};
function localSourceDetection(name,url){
  const n=(name||'').trim().toLowerCase(),u=(url||'').trim(),l=u.toLowerCase();
  for(const [key,v] of Object.entries(KNOWN_BOARDS))if(n.includes(key)||l.includes(key))return {...v,detected_by:'known organization'};
  let m;
  if(l.includes('greenhouse.io')){
    m=u.match(/https?:\/\/(?:boards|job-boards)\.greenhouse\.io\/([^/?#]+)/i)||u.match(/boards-api\.greenhouse\.io\/v1\/boards\/([^/?#]+)/i);
    return {source_type:'greenhouse',board_token:m?m[1]:'',detected_by:'careers URL'};
  }
  if(l.includes('lever.co')){
    m=u.match(/https?:\/\/jobs\.lever\.co\/([^/?#]+)/i)||u.match(/api\.lever\.co\/v0\/postings\/([^/?#]+)/i);
    return {source_type:'lever',board_token:m?m[1]:'',detected_by:'careers URL'};
  }
  if(l.includes('google.com/about/careers'))return {source_type:'google_careers',board_token:'',detected_by:'careers URL'};
  return {source_type:'generic',board_token:'',detected_by:'automatic detection'};
}
function applyDetection(r,d){
  const type=r.querySelector('.type'),token=r.querySelector('.token');
  type.value=d.source_type||'generic';token.value=d.board_token||'';
}
async function detectOrgRow(r,remote=false){
  const name=r.querySelector('.name').value.trim(),url=r.querySelector('.url').value.trim();
  const local=localSourceDetection(name,url);applyDetection(r,local);
  if(!remote||!url)return;
  try{const d=await api('/api/source-detect',{method:'POST',body:JSON.stringify({name,careers_url:url})});applyDetection(r,d)}catch(e){applyDetection(r,local)}
}
let orgPage=1;
const ORGS_PER_PAGE=10;
function org(o={source_type:'generic'}){
  let r=document.createElement('div');
  r.className='orgrow';
  r.innerHTML=`<span class="org-index"></span><label class="orgfield"><span>Organization Name</span><input class="name" placeholder="Organization" value="${esc(o.name||'')}"></label><label class="orgfield careers-field"><span>Careers URL</span><input class="url" placeholder="https://..." value="${esc(o.careers_url||'')}"></label><label class="orgfield"><span>Source</span><select class="type auto-field" disabled title="Detected automatically from the careers website"><option value="greenhouse">Greenhouse</option><option value="lever">Lever</option><option value="google_careers">Google Careers</option><option value="generic">Other careers site</option></select></label><label class="orgfield"><span>Board Token</span><input class="token auto-field" placeholder="—" value="${esc(o.board_token||'')}" readonly title="Auto-detected if required"></label><label class="orgfield"><span>Location</span><input class="location-pref" placeholder="Bengaluru" value="${esc(o.location||'')}"></label><div class="remove-wrap"><span>Remove</span><button class="remove" title="Remove organization">×</button></div>`;
  r.querySelector('.remove').onclick=()=>{r.remove();const total=$$('.orgrow').length;const max=Math.max(1,Math.ceil(total/ORGS_PER_PAGE));orgPage=Math.min(orgPage,max);renderOrgPage()};
  r.querySelector('.url').addEventListener('input',()=>detectOrgRow(r,false));
  r.querySelector('.url').addEventListener('blur',()=>detectOrgRow(r,true));
  r.querySelector('.name').addEventListener('blur',()=>detectOrgRow(r,true));
  $('#orgs').append(r);
  applyDetection(r,{source_type:o.source_type||'generic',board_token:o.board_token||'',detected_by:o.source_detected_by||'saved preference'});
  if(o.careers_url&&!o.source_type)detectOrgRow(r,false);
  return r;
}
function renderOrgPage(){
  const rows=$$('.orgrow'),total=rows.length,max=Math.max(1,Math.ceil(total/ORGS_PER_PAGE));
  orgPage=Math.max(1,Math.min(orgPage,max));
  const start=(orgPage-1)*ORGS_PER_PAGE,end=Math.min(start+ORGS_PER_PAGE,total);
  rows.forEach((r,i)=>{r.hidden=!(i>=start&&i<end);const ix=r.querySelector('.org-index');if(ix)ix.textContent=i+1});
  const totalEl=$('#org-total'),range=$('#org-range'),pages=$('#org-pages'),prev=$('#org-prev'),next=$('#org-next');
  if(totalEl)totalEl.textContent=total;
  if(range)range.textContent=total?`Showing ${start+1}–${end} of ${total}`:'Showing 0–0 of 0';
  if(prev)prev.disabled=orgPage<=1;if(next)next.disabled=orgPage>=max;
  if(pages){pages.innerHTML='';for(let i=1;i<=max;i++){let b=document.createElement('button');b.className='page-number'+(i===orgPage?' active':'');b.textContent=i;b.type='button';b.onclick=()=>{orgPage=i;renderOrgPage()};pages.append(b)}}
}
function orgs(a=[]){$('#orgs').innerHTML='';orgPage=1;a.forEach(org);if(!a.length)org();renderOrgPage()}
function showToday(){let e=$('#today-date');if(e)e.textContent=new Intl.DateTimeFormat(undefined,{weekday:'long',year:'numeric',month:'long',day:'numeric'}).format(new Date())}
const PREF_STATE_KEY='careerCompass.jobPreferences.v1';
function savePreferenceBackup(organizations,roles){
  try{localStorage.setItem(PREF_STATE_KEY,JSON.stringify({organizations:organizations||[],roles:roles||[],saved_at:new Date().toISOString()}))}catch(_){}
}
function loadPreferenceBackup(){
  try{let x=JSON.parse(localStorage.getItem(PREF_STATE_KEY)||'null');return x&&Array.isArray(x.organizations)&&Array.isArray(x.roles)?x:null}catch(_){return null}
}
async function load(){
  showToday();resetRefreshProgress();
  let[j,c]=await Promise.all([api('/api/jobs'),api('/api/config')]);
  const backup=loadPreferenceBackup();
  // Browser-saved preferences are authoritative on launch. This protects the
  // user's settings when a newer application ZIP replaces data/portal.json
  // with the packaged seed/default configuration.
  if(backup&&backup.organizations&&backup.organizations.length){
    try{
      const restored=await api('/api/config',{method:'PUT',body:JSON.stringify({organizations:backup.organizations,roles:backup.roles})});
      c={...c,organizations:restored.organizations||backup.organizations,roles:restored.roles||backup.roles};
    }catch(_){
      c={...c,organizations:backup.organizations,roles:backup.roles};
    }
  }
  jobs=j.jobs;cards();$('#roles').value=(c.roles||[]).join('\n');$('#resume-status').textContent=c.resume?'Saved: '+c.resume.filename:'No resume uploaded yet.';orgs(c.organizations||[]);
  if(c.organizations&&c.organizations.length)savePreferenceBackup(c.organizations,c.roles||[]);
}
function resetRefreshProgress(){
  const box=$('#refresh-progress'),fill=$('#refresh-fill'),pct=$('#refresh-percent'),stage=$('#refresh-stage');
  box.classList.remove('complete','failed');
  box.classList.add('active');
  box.setAttribute('aria-hidden','false');
  fill.style.width='0%';
  pct.textContent='0%';
  stage.textContent='Ready to refresh';
}
function refreshProgress(){
  const box=$('#refresh-progress'),fill=$('#refresh-fill'),pct=$('#refresh-percent'),stage=$('#refresh-stage');
  const stages=[
    [8,'Reading your job preferences…'],
    [24,'Connecting to career sites…'],
    [46,'Collecting fresh openings…'],
    [66,'Matching roles and locations…'],
    [82,'Checking the last 7 days…'],
    [92,'Preparing your dashboard…']
  ];
  let value=3,index=0,timer;
  box.classList.remove('complete','failed');box.classList.add('active');box.setAttribute('aria-hidden','false');fill.style.width=value+'%';pct.textContent=value+'%';stage.textContent='Starting refresh…';
  timer=setInterval(()=>{
    const next=stages[index];
    if(next&&value>=next[0]){stage.textContent=next[1];index++}
    if(value<92){value=Math.min(92,value+(value<45?3:value<75?2:1));fill.style.width=value+'%';pct.textContent=value+'%'}
  },240);
  return {
    done(message='Dashboard updated'){
      clearInterval(timer);stage.textContent=message;fill.style.width='100%';pct.textContent='100%';box.classList.remove('failed');box.classList.add('active','complete');box.setAttribute('aria-hidden','false');
    },
    fail(message='Refresh failed'){
      clearInterval(timer);stage.textContent=message;box.classList.add('failed');
      setTimeout(()=>{box.classList.remove('active','failed');box.setAttribute('aria-hidden','true');fill.style.width='0%';pct.textContent='0%'},2200)
    }
  }
}
$('#refresh').onclick=async()=>{let b=$('#refresh'),progress=refreshProgress();b.disabled=1;b.classList.add('is-refreshing');b.textContent='↻ Refreshing';try{let x=await api('/api/refresh',{method:'POST'});jobs=x.jobs;picked.clear();cards();count();progress.done(`${jobs.length} matching job${jobs.length==1?'':'s'} ready`);notice(`Checked ${x.checked} source${x.checked==1?'':'s'}; found ${jobs.length} matching job${jobs.length==1?'':'s'}.`+(x.diagnostics?.length?' '+x.diagnostics.map(d=>`${d.organization}: raw ${d.raw??0}, matched ${d.matched??0}`).join(' · '):'')+(x.errors.length?' · '+x.errors.join(' · '):''),x.errors.length)}catch(e){progress.fail('Could not complete refresh');notice(e.message,1)}finally{b.disabled=0;b.classList.remove('is-refreshing');b.textContent='↻ Refresh jobs'}};
$('#add').onclick=()=>{org();orgPage=Math.max(1,Math.ceil($$('.orgrow').length/ORGS_PER_PAGE));renderOrgPage()};$('#org-prev').onclick=()=>{if(orgPage>1){orgPage--;renderOrgPage()}};$('#org-next').onclick=()=>{const max=Math.max(1,Math.ceil($$('.orgrow').length/ORGS_PER_PAGE));if(orgPage<max){orgPage++;renderOrgPage()}};$('#save-settings').onclick=async()=>{let b=$('#save-settings');b.disabled=1;b.textContent='Detecting & saving…';try{await Promise.all($$('.orgrow').map(r=>detectOrgRow(r,true)));let organizations=$$('.orgrow').map(r=>({name:r.querySelector('.name').value.trim(),source_type:r.querySelector('.type').value,board_token:r.querySelector('.token').value.trim(),careers_url:r.querySelector('.url').value.trim(),location:r.querySelector('.location-pref').value.trim()})).filter(x=>x.name);let roleList=$('#roles').value.split('\n').map(x=>x.trim()).filter(Boolean);let result=await api('/api/config',{method:'PUT',body:JSON.stringify({organizations,roles:roleList})});let savedOrgs=result.organizations||organizations;orgs(savedOrgs);savePreferenceBackup(savedOrgs,roleList);notice('Preferences saved and will be restored automatically next time you launch Career Compass.')}catch(e){notice(e.message,1)}finally{b.disabled=0;b.textContent='Save preferences'}};
$('#submit').onclick=async()=>{try{let x=await api('/api/submissions',{method:'POST',body:JSON.stringify({job_ids:[...picked]})});notice(`${x.added} role${x.added==1?'':'s'} added to submitted history.`)}catch(e){notice(e.message,1)}};
$('#file').onchange=async e=>{if(!e.target.files[0])return;let selected=e.target.files[0],f=new FormData();f.append('file',selected);try{let r=await fetch('/api/resume',{method:'POST',body:f}),d=await r.json();if(!r.ok)throw Error(d.detail);$('#resume-status').textContent=`Saved: ${d.filename} (${d.characters.toLocaleString()} readable characters)`;let indicator=$('#resume-uploaded-indicator'),name=$('#resume-uploaded-name');name.textContent=d.filename||selected.name;indicator.hidden=false;notice('Resume uploaded for tailored resume creation.')}catch(x){notice(x.message,1)}};
$('#tailor').onclick=async()=>{let b=$('#tailor');b.disabled=1;b.textContent='Tailoring…';try{let x=await api('/api/tailor',{method:'POST',body:JSON.stringify({job_ids:[...picked]})});x.files.forEach(f=>window.open(f.url,'_blank'));notice(`${x.files.length} tailored resume${x.files.length==1?'':'s'} ready for download.`)}catch(e){notice(e.message,1)}finally{b.disabled=0;b.textContent='Tailor selected resumes'}};
async function history(){let d=await api('/api/submissions'),b=$('#history-list');b.innerHTML='';$('#no-history').style.display=d.submissions.length?'none':'block';d.submissions.forEach(s=>{let j=s.job,e=document.createElement('article');e.innerHTML=`<h3>${esc(j.title)} · ${esc(j.organization)}</h3><p>Submitted ${new Date(s.submitted_at).toLocaleDateString()} · Job ID: ${esc(j.job_id)}</p>`;s.tailored_files.forEach(f=>{let a=document.createElement('a');a.href='/api/download/'+encodeURIComponent(f);a.textContent='Download tailored resume ↓';e.append(a)});b.append(e)})}
$$('.nav').forEach(n=>n.onclick=async()=>{$$('.nav').forEach(x=>x.classList.remove('active'));n.classList.add('active');$$('.view').forEach(x=>x.classList.remove('active'));$('#'+n.dataset.v).classList.add('active');$('#title').textContent={dashboard:'Your opportunities',history:'Your submitted roles',settings:'Job Preferences'}[n.dataset.v];$('#eyebrow').textContent=n.dataset.v==='dashboard'?'WELCOME BACK':n.dataset.v.toUpperCase();$('#today-date').style.display=n.dataset.v==='dashboard'?'block':'none';if(n.dataset.v==='history')history()});load().catch(e=>notice(e.message,1));
