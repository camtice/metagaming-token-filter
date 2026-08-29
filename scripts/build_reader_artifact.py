"""Assemble the rule-comparison token reader artifact."""
import json

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
data = json.load(open(f"{ROOT}/out/reader_data.json"))

HTML = r"""<title>Rule Reader</title>
<style>
:root{
  --bg:#F7F8F6; --card:#FFFFFF; --ink:#1C2321; --ink2:#566058; --ink3:#7C877F;
  --line:#DCE1DC; --accent:#1F6E5C; --accent-ink:#175243; --chip:#EEF1EC;
  --hlA:#2a78d6; --hlB:#eb6834; --both:#7a5fb0; --gtline:#1C2321;
  --agree1:rgba(31,110,92,.14); --agree2:rgba(31,110,92,.30); --agree3:rgba(31,110,92,.48);
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#121614; --card:#1B211E; --ink:#E8ECE9; --ink2:#A9B3AD; --ink3:#7C877F;
  --line:#2A322E; --accent:#4FA98A; --accent-ink:#7CC5AB; --chip:#232A26;
  --hlA:#3987e5; --hlB:#d95926; --both:#9c85d6; --gtline:#E8ECE9;
  --agree1:rgba(79,169,138,.16); --agree2:rgba(79,169,138,.32); --agree3:rgba(79,169,138,.52);
}}
:root[data-theme="dark"]{
  --bg:#121614; --card:#1B211E; --ink:#E8ECE9; --ink2:#A9B3AD; --ink3:#7C877F;
  --line:#2A322E; --accent:#4FA98A; --accent-ink:#7CC5AB; --chip:#232A26;
  --hlA:#3987e5; --hlB:#d95926; --both:#9c85d6; --gtline:#E8ECE9;
  --agree1:rgba(79,169,138,.16); --agree2:rgba(79,169,138,.32); --agree3:rgba(79,169,138,.52);
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:18px 18px 60px}
h1{font-family:Charter,Georgia,serif;font-size:22px;margin:2px 0 4px}
.kicker{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:600}
.sub{color:var(--ink2);font-size:13px;max-width:90ch;margin:2px 0 10px}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:.9em}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0;position:sticky;top:0;
  background:var(--bg);padding:8px 0;z-index:5;border-bottom:1px solid var(--line)}
.bar .lbl{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);font-weight:650}
button.chip{all:unset;cursor:pointer;font-size:12.5px;font-weight:600;color:var(--ink2);
  border:1px solid var(--line);border-radius:999px;padding:3px 11px;background:var(--card)}
button.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.chip[aria-pressed="true"]{background:var(--chip);color:var(--ink);border-color:var(--accent)}
button.chip.b[aria-pressed="true"]{border-color:var(--hlB)}
.cols{display:grid;grid-template-columns:300px 1fr;gap:14px;align-items:start}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px}
.doclist{max-height:80vh;overflow-y:auto;padding-bottom:8px}
.grp{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;font-weight:650;color:var(--ink3);
  padding:10px 12px 3px;position:sticky;top:0;background:var(--card)}
.docrow{display:block;width:100%;text-align:left;border:none;background:none;cursor:pointer;
  padding:5px 12px;font:12.5px ui-monospace,Menlo,monospace;color:var(--ink2);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;border-left:3px solid transparent}
.docrow:hover{background:var(--chip)}
.docrow.sel{border-left-color:var(--accent);color:var(--ink);background:var(--chip)}
.docrow .pct{color:var(--ink3);font-size:11px}
.reader{padding:14px 18px}
.reader h2{margin:0;font-size:15px;word-break:break-all;font-family:ui-monospace,Menlo,monospace}
.meta{color:var(--ink3);font-size:12px;margin:2px 0 8px}
.stats{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--ink2);margin:0 0 8px}
.stats b{color:var(--ink)}
.txt{border:1px solid var(--line);border-radius:8px;background:var(--bg);padding:12px 14px;
  font-size:13.5px;line-height:1.85;white-space:pre-wrap;overflow-wrap:break-word;
  max-height:64vh;overflow-y:auto;max-width:100ch}
.tA{background:color-mix(in srgb,var(--hlA) 30%,transparent);border-radius:2px}
.tB{background:color-mix(in srgb,var(--hlB) 32%,transparent);border-radius:2px}
.tAB{background:color-mix(in srgb,var(--both) 40%,transparent);border-radius:2px}
.g1{background:var(--agree1)} .g2{background:var(--agree2)} .g3{background:var(--agree3);border-radius:2px}
.gt{box-shadow:inset 0 -2.5px 0 var(--gtline)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--ink2);margin:8px 0 0}
.sw{display:inline-block;width:26px;border-radius:3px;text-align:center;font-size:11px}
.fpanel{margin-top:10px;font-size:12px;color:var(--ink2)}
.fpanel table{border-collapse:collapse;width:100%;font-size:12px}
.fpanel td{padding:2px 8px 2px 0;vertical-align:top}
details{margin-top:8px}
summary{cursor:pointer;color:var(--accent-ink);font-size:12.5px;font-weight:600}
</style>
<div class="wrap">
<div class="kicker">Metagaming filter · qualitative reader · G-TEST + kimi rollouts</div>
<h1>Rule Reader — token-level comparison</h1>
<p class="sub">Click a document, then pick one rule (highlight), two rules (diff: blue = first only,
orange = second only, purple = both), or <b>agreement</b> mode (darker = more of the six rules flag
the token). Underline = human-labeled span. Doc list shows rule-A (champion) flag share.
<b>Naming:</b> the <i>ratio screen</i> ρ selects features (fire ≥ρ× more on forget than retain
training text); the <i>vote k</i> is how many selected features must co-fire on a token to seed it
(paper uses k=2); labels then grow to adjacent active tokens (the paper's windowing). All rules were
selected on G-TRAIN only; everything shown here is held-out or unlabeled. Rules use the v8 judge pool
on the 65k SAE (the frozen mainline); rule A is the atlas champion (ρ≥10 · k=4, held-out F2 .821).</p>
<div class="bar" id="rulebar"><span class="lbl">rules</span></div>
<div class="cols">
  <div class="panel doclist" id="doclist"></div>
  <div class="panel reader" id="reader"></div>
</div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const GROUPS = [["span","Human-labeled rollouts (G-TEST)"],["c2","Held-out domain: dense c2 AI-safety"],
  ["kimi_trick","Kimi trick arms (unlabeled)"],["kimi_clean","Kimi clean arms (unlabeled)"],
  ["agentic_notice","Agentic — notice arm (unlabeled)"],["agentic_baseline","Agentic — baseline arm"],
  ["dense","Dense in-domain (G-TEST half)"],["clean","Reviewed-clean chat"],
  ["ctl_topfp","Controls — highest FP"],["ctl_random","Controls — random"]];
const state = {sel: [], mode: "rules", doc: null};
function unrle(r){
  const out=[]; let v=!!r[0];
  for(let i=1;i<r.length;i++){for(let j=0;j<r[i];j++)out.push(v); v=!v;}
  return out;
}
for(const d of D.docs){
  d.F={}; for(const c in d.flags) d.F[c]=unrle(d.flags[c]);
  d.GT = d.gt && d.gt.length ? unrle(d.gt) : null;
  d.rateA = d.F.A ? d.F.A.filter(Boolean).length/Math.max(d.F.A.length,1) : 0;
}
const bar=document.getElementById('rulebar');
for(const r of D.rules){
  const b=document.createElement('button');
  b.className='chip'; b.setAttribute('aria-pressed','false');
  b.textContent=`${r.code} · ${r.name}`;
  b.title=`width ${r.width}`;
  b.addEventListener('click',()=>{
    const i=state.sel.indexOf(r.code);
    if(i>=0) state.sel.splice(i,1);
    else { state.sel.push(r.code); if(state.sel.length>2) state.sel.shift(); }
    state.mode="rules"; sync();
  });
  b.dataset.code=r.code;
  bar.appendChild(b);
}
const agr=document.createElement('button');
agr.className='chip b'; agr.setAttribute('aria-pressed','false'); agr.textContent='agreement (all 6)';
agr.addEventListener('click',()=>{state.mode = state.mode==='agree' ? 'rules':'agree'; sync();});
bar.appendChild(agr);
const dl=document.getElementById('doclist');
for(const [g,label] of GROUPS){
  const docs=D.docs.filter(d=>d.group===g);
  if(!docs.length) continue;
  const h=document.createElement('div'); h.className='grp'; h.textContent=`${label} · ${docs.length}`;
  dl.appendChild(h);
  docs.sort((a,b)=>b.rateA-a.rateA);
  for(const d of docs){
    const b=document.createElement('button'); b.className='docrow'; b.dataset.id=d.id;
    b.innerHTML=`${d.id.replace(/^kimi:|^cotprobe:|^pr123:/,'').slice(0,34)} <span class="pct">${(d.rateA*100).toFixed(0)}%</span>`;
    b.title=d.id;
    b.addEventListener('click',()=>{state.doc=d.id; sync();});
    dl.appendChild(b);
  }
}
state.doc = (D.docs.find(d=>d.group==='span')||D.docs[0]).id;
state.sel = ['A'];
const reader=document.getElementById('reader');
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function sync(){
  for(const b of bar.querySelectorAll('button.chip[data-code]'))
    b.setAttribute('aria-pressed', String(state.mode==='rules' && state.sel.includes(b.dataset.code)));
  agr.setAttribute('aria-pressed', String(state.mode==='agree'));
  for(const b of dl.querySelectorAll('.docrow')) b.classList.toggle('sel', b.dataset.id===state.doc);
  const d=D.docs.find(x=>x.id===state.doc);
  if(!d){reader.innerHTML='';return;}
  const codes = state.sel.length?state.sel:['A'];
  let html='', prev=0;
  const N=d.toks.length;
  const parts=[];
  for(let i=0;i<N;i++){
    const [s,e]=d.toks[i];
    if(s>prev) parts.push(esc(d.text.slice(prev,s)));
    let cls='';
    if(state.mode==='agree'){
      let n=0; for(const c in d.F) if(d.F[c][i]) n++;
      cls = n>=5?'g3':n>=3?'g2':n>=1?'g1':'';
    } else if(codes.length===1){
      cls = d.F[codes[0]] && d.F[codes[0]][i] ? 'tA':'';
    } else {
      const a=d.F[codes[0]]&&d.F[codes[0]][i], b=d.F[codes[1]]&&d.F[codes[1]][i];
      cls = a&&b?'tAB':a?'tA':b?'tB':'';
    }
    if(d.GT && d.GT[i]) cls+=' gt';
    parts.push(cls.trim()?`<span class="${cls.trim()}">${esc(d.text.slice(s,e))}</span>`:esc(d.text.slice(s,e)));
    prev=e;
  }
  if(prev<d.text.length) parts.push(esc(d.text.slice(prev)));
  const stats = D.rules.map(r=>{
    const f=d.F[r.code]; if(!f) return '';
    const rate=f.filter(Boolean).length/Math.max(f.length,1);
    let rec='';
    if(d.GT){
      let tp=0,g=0; for(let i=0;i<N;i++){if(d.GT[i]){g++; if(f[i])tp++;}}
      if(g) rec=` · span recall <b>${(tp/g*100).toFixed(0)}%</b>`;
    }
    return `<span><b>${r.code}</b> ${(rate*100).toFixed(1)}%${rec}</span>`;
  }).join('');
  let legend='';
  if(state.mode==='agree') legend=`<span class="sw g1">1+</span> <span class="sw g2">3+</span> <span class="sw g3">5+</span> rules agree`;
  else if(codes.length===2) legend=`<span class="sw tA">${codes[0]}</span> only · <span class="sw tB">${codes[1]}</span> only · <span class="sw tAB">both</span>`;
  else legend=`<span class="sw tA">flagged by ${codes[0]}</span>`;
  const fp = D.rules.map(r=>`<tr><td><b>${r.code}</b></td><td>`+
    r.top_fp_feats.slice(0,4).map(f=>`<span class="mono">${f.lat}</span> ${esc(f.cap||'(uncaptioned)').slice(0,44)}`).join(' · ')+`</td></tr>`).join('');
  reader.innerHTML = `<h2>${esc(d.id)}</h2><div class="meta">${esc(d.meta||'')} · group ${d.group}</div>`+
    `<div class="stats">${stats}</div><div class="legend">${legend} · underline = human span</div>`+
    `<div class="txt">${parts.join('')}</div>`+
    `<details class="fpanel"><summary>top control-FP seeding features per rule</summary><table>${fp}</table></details>`;
}
sync();
</script>
"""
payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
open(f"{ROOT}/notes/rule_reader.html", "w").write(HTML.replace("__DATA__", payload))
print(f"wrote notes/rule_reader.html ({(len(HTML)+len(payload))/1e6:.2f} MB)")
