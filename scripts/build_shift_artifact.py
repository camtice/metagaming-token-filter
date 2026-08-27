"""Generate notes/feature_shift_2026-08-15.html — the D2-vs-P2 feature-shift artifact."""
import json

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
data = json.load(open(f"{ROOT}/out/artifact_shift_data.json"))

HTML = r"""<title>Forget-Set Feature Shift</title>
<style>
:root{
  --bg:#F7F8F5; --card:#FFFFFF; --ink:#1C2321; --ink2:#566058; --ink3:#7C877F;
  --line:#DCE1DC; --accent:#1F6E5C; --accent-ink:#175243;
  --added:#2a78d6; --dropped:#eb6834; --kept:#1baf7a; --pool:#98A29C;
  --chip:#EEF1EC; --warnbg:#FBF3E8; --warnline:#E8D9BF;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#14181A; --card:#1B211E; --ink:#E8ECE9; --ink2:#A9B3AD; --ink3:#7C877F;
  --line:#2A322E; --accent:#4FA98A; --accent-ink:#7CC5AB;
  --added:#3987e5; --dropped:#d95926; --kept:#199e70; --pool:#5F6A64;
  --chip:#232A26; --warnbg:#2A2620; --warnline:#4A4132;
}}
:root[data-theme="dark"]{
  --bg:#14181A; --card:#1B211E; --ink:#E8ECE9; --ink2:#A9B3AD; --ink3:#7C877F;
  --line:#2A322E; --accent:#4FA98A; --accent-ink:#7CC5AB;
  --added:#3987e5; --dropped:#d95926; --kept:#199e70; --pool:#5F6A64;
  --chip:#232A26; --warnbg:#2A2620; --warnline:#4A4132;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
  font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
main{max-width:78ch;margin:0 auto;padding:2.5rem 1.25rem 5rem}
h1,h2{font-family:Charter,Georgia,"Times New Roman",serif;line-height:1.15;text-wrap:balance}
h1{font-size:2.1rem;margin:0 0 .3rem}
h2{font-size:1.35rem;margin:2.6rem 0 .7rem;padding-top:1.2rem;border-top:1px solid var(--line)}
.kicker{font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;color:var(--accent);font-weight:600}
.sub{color:var(--ink2);margin:.2rem 0 0}
p{margin:.75rem 0}
a{color:var(--accent-ink)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.86em}
.tile-row{display:flex;gap:.8rem;flex-wrap:wrap;margin:1.4rem 0}
.tile{flex:1 1 10rem;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.8rem 1rem}
.tile .v{font-size:1.55rem;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.tile .k{font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:var(--ink2)}
.tile .d{font-size:.8rem;color:var(--ink3)}
.scroll{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th{font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2);
  text-align:left;padding:.55rem .65rem;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:.45rem .65rem;border-bottom:1px solid var(--line);vertical-align:top;
  font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right}
.hl td{background:color-mix(in srgb,var(--accent) 8%,transparent)}
.dot{display:inline-block;width:.62em;height:.62em;border-radius:50%;margin-right:.4em;vertical-align:baseline}
.tok{display:inline-block;background:var(--chip);border-radius:4px;padding:0 .35em;margin:.06em .12em .06em 0;
  font-family:ui-monospace,Menlo,monospace;font-size:.78em;color:var(--ink2);white-space:pre}
.cls-added{color:var(--added)} .cls-dropped{color:var(--dropped)} .cls-kept{color:var(--kept)}
.note{background:var(--warnbg);border:1px solid var(--warnline);border-radius:8px;padding:.7rem 1rem;font-size:.9rem}
figure{margin:1.2rem 0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:1rem}
figcaption{font-size:.8rem;color:var(--ink2);margin-top:.5rem}
.legend{display:flex;gap:1rem;flex-wrap:wrap;font-size:.82rem;margin:.4rem 0 .6rem}
.legend button{all:unset;cursor:pointer;display:inline-flex;align-items:center;gap:.35em;
  padding:.15em .5em;border-radius:6px;border:1px solid var(--line);color:var(--ink2)}
.legend button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.legend button[aria-pressed="false"]{opacity:.35}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);
  border-radius:6px;padding:.5rem .7rem;font-size:.78rem;max-width:22rem;display:none;
  box-shadow:0 4px 14px rgba(0,0,0,.12);z-index:9}
#tip .t{font-weight:650}
#tip .c{color:var(--ink2)}
svg text{fill:var(--ink2);font:11px system-ui}
details{margin:.8rem 0}
summary{cursor:pointer;color:var(--accent-ink);font-size:.9rem}
@media (prefers-reduced-motion:no-preference){ .legend button{transition:opacity .15s} }
</style>
<main>
<div class="kicker">Metagaming filter · TEST v4 · 2026-08-15</div>
<h1>Forget-Set Feature Shift</h1>
<p class="sub">What changes in the selected SAE latents when the main forget set moves from
<em>human spans only</em> (P2) to <em>human spans + dense c1&ndash;c4 documents</em> (D2) &mdash;
both maximized for F2 from the same 649-latent fable pool at the same control-FP budget (&le;.10).</p>

<div class="tile-row">
  <div class="tile"><div class="k">D2 main F2</div><div class="v">.882</div><div class="d">P .878 &middot; R .883 &middot; n=187</div></div>
  <div class="tile"><div class="k">Features swapped</div><div class="v">102 in / 49 out</div><div class="d">85 kept of P2&rsquo;s 134</div></div>
  <div class="tile"><div class="k">Span chat recall</div><div class="v">.527</div><div class="d">was .682 in P2 &mdash; the cost</div></div>
</div>

<p><strong>Verdict.</strong> The dense forget set does exactly what it was added to do: it
re-anchors selection on <em>topic</em> evidence and evicts the register features that the
span-only objective had learned to exploit. The 49 dropped features average 3&times; less
dense-forget evidence than the 102 added ones (mean fire rate .0038 vs .0125), and the
additions read like a glossary of the domain &mdash; <em>large language model, OpenAI generative
AI, compute costs, loss function, human feedback or judgment, standardized test
preparation</em>. Ten of the thirteen register suspects from the precision post-mortem are
now out. The price is paid in exactly one place: human-span recall in the chat register
(.682 &rarr; .527), where metagaming lives inside assistant-answer scaffolding that dense
prose does not teach.</p>

<h2>Scoreboard under both metrics</h2>
<div class="scroll"><table id="score"></table></div>
<p class="note"><strong>Read the two metric families separately.</strong> Main-metric precision is
mechanically higher for every set: dense docs are all-forget, so they contribute no
negative tokens to the precision denominator. Span metrics are the human-label check
(reported separately per the protocol); main is the headline as of protocol v5.
OG row uses the official scorer; dense c5 (SWE) is excluded from the forget side
everywhere, per the 2026-08-12 retain ruling.</p>

<h2>Where each feature&rsquo;s evidence lives</h2>
<figure>
  <div class="legend" id="legend" role="group" aria-label="toggle classes"></div>
  <div id="chart"></div>
  <figcaption>Each point is one of the 649 pool latents. x: fire rate inside human
  span tokens; y: fire rate on dense c1&ndash;c4 tokens (both above the mean+4SD threshold;
  log scale, zero pinned at the axis floor). The dashed line is equal evidence. Dropped
  features (orange) hug the x-axis &mdash; span evidence without dense corroboration; added
  features (blue) sit high on dense evidence. Hover a point for its caption and rates.
  The tables below carry the same data in text form.</figcaption>
</figure>

<h2>What was dropped &mdash; span-only carriers</h2>
<p>Top of the 49 by span evidence. Math/answer-format register dominates &mdash; these earned
recall inside umwp/mmlu rollout spans but have near-zero dense-forget footprint.</p>
<div class="scroll"><table id="dropped"></table></div>

<h2>What was added &mdash; dense-corroborated topic features</h2>
<p>Top of the 102 by dense evidence.</p>
<div class="scroll"><table id="added"></table></div>

<h2>The register suspects, tracked</h2>
<p>The thirteen features named in the precision post-mortem (2026-08-15) as
assistant-scaffold / filler / format detectors, and their fate under each objective.</p>
<div class="scroll"><table id="suspects"></table></div>

<h2>Caveats</h2>
<p>&bull; <strong>Chat-span recall is the open cost.</strong> D2 keeps 817 (assistant-scaffold) and
1419 (test-cases) because they carry dense evidence too, but chat recall still fell to
.527. If chat metagaming is the priority target, the register features were not all
spurious &mdash; some were the only carriers of real chat spans. More labeled chat data is
the fix, not re-adding filler features.<br>
&bull; <strong>Main-metric numbers are not comparable to span-metric numbers</strong> (see scoreboard
note).<br>
&bull; <strong>Selection ran on TEST</strong> (~10k evaluations); numbers are optimistic. Sealed
validation remains unspent and should be spent once, on one chosen set.<br>
&bull; D1 (recall-first variant) ended dominated by D2 and can be ignored.</p>

<p class="sub" style="font-size:.8rem">Sets: <span class="mono">data/candidate_sets/fable_trim_{p2,d1,d2}.json</span> &middot;
selection <span class="mono">scripts/select_latents_fable_v5_dense.py</span> &middot;
scorer <span class="mono">scripts/score_split.py</span> (protocol v5 main metric) &middot;
data <span class="mono">out/fable_trim_v5_dense.json</span>, <span class="mono">out/feature_shift_d2_p2.json</span></p>
</main>
<div id="tip" role="status"></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const CLS = {added:{c:'var(--added)',n:'added (D2 only)'}, dropped:{c:'var(--dropped)',n:'dropped (P2 only)'},
             kept:{c:'var(--kept)',n:'kept (both)'}, pool:{c:'var(--pool)',n:'pool (neither)'}};
const fmt = (x,d=3)=> x==null?'&mdash;':(+x).toFixed(d).replace(/^0\./,'.');
/* scoreboard */
{
  const t = document.getElementById('score');
  t.innerHTML = '<thead><tr><th>set</th><th class="num">n</th>'+
    '<th class="num">P<sub>main</sub></th><th class="num">R<sub>main</sub></th><th class="num">F2<sub>main</sub></th>'+
    '<th class="num">P<sub>span</sub></th><th class="num">R<sub>span</sub></th><th class="num">F2<sub>span</sub></th>'+
    '<th class="num">fp ctl</th><th class="num">fp clean-chat</th></tr></thead><tbody>'+
    D.rows.map(r=>`<tr${r.note==='recommended'?' class="hl"':''}><td>${r.name}${r.note?` <span style="color:var(--ink3);font-size:.75em">(${r.note})</span>`:''}</td>`+
      `<td class="num">${r.n}</td>`+
      r.main.map(v=>`<td class="num"><strong>${fmt(v)}</strong></td>`).join('')+
      r.span.map(v=>`<td class="num">${fmt(v)}</td>`).join('')+
      `<td class="num">${fmt(r.fp)}</td><td class="num">${fmt(r.fpc)}</td></tr>`).join('')+'</tbody>';
}
/* feature tables */
function featTable(el, rows){
  el.innerHTML = '<thead><tr><th>latent</th><th>caption</th><th class="num">q span</th>'+
    '<th class="num">q dense</th><th class="num">r ctl</th><th class="num">stop frac</th><th>fires on</th></tr></thead><tbody>'+
    rows.map(f=>`<tr><td class="mono"><span class="dot" style="background:${CLS[f.c].c}"></span>${f.l}</td>`+
      `<td>${f.cap||'<span style="color:var(--ink3)">(uncaptioned)</span>'}</td>`+
      `<td class="num">${fmt(f.qs,4)}</td><td class="num">${fmt(f.qd,4)}</td><td class="num">${fmt(f.rc,4)}</td>`+
      `<td class="num">${fmt(f.st,2)}</td>`+
      `<td>${(f.top||[]).map(t=>`<span class="tok">${t.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</span>`).join('')}</td></tr>`).join('')+'</tbody>';
}
featTable(document.getElementById('dropped'),
  D.feats.filter(f=>f.c==='dropped').sort((a,b)=>b.qs-a.qs).slice(0,18));
featTable(document.getElementById('added'),
  D.feats.filter(f=>f.c==='added').sort((a,b)=>b.qd-a.qd).slice(0,18));
/* suspects */
{
  const t = document.getElementById('suspects');
  const mark = b => b?'<strong>in</strong>':'<span style="color:var(--ink3)">out</span>';
  t.innerHTML = '<thead><tr><th>latent</th><th>caption</th><th>P2 (span)</th><th>D2 (dense)</th>'+
    '<th class="num">q dense</th><th class="num">stop frac</th><th>fires on</th></tr></thead><tbody>'+
    D.suspects.map(s=>`<tr><td class="mono">${s.l}</td><td>${s.cap}</td>`+
      `<td>${mark(s.p2)}</td><td>${mark(s.d2)}</td>`+
      `<td class="num">${fmt(s.qd,4)}</td><td class="num">${fmt(s.st,2)}</td>`+
      `<td>${(s.top||[]).map(t=>`<span class="tok">${t.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</span>`).join('')}</td></tr>`).join('')+'</tbody>';
}
/* scatter */
{
  const W=680, H=430, M={l:52,r:14,t:10,b:40}, EPS=1e-4;
  const lg=x=>Math.log10(x+EPS), x0=lg(0), x1=lg(0.2);
  const sx=v=>M.l+(lg(v)-x0)/(x1-x0)*(W-M.l-M.r);
  const sy=v=>H-M.b-(lg(v)-x0)/(x1-x0)*(H-M.t-M.b);
  const NS='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(NS,'svg');
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.setAttribute('style','width:100%;height:auto;display:block');
  svg.setAttribute('role','img');
  svg.setAttribute('aria-label','Scatter of span fire rate versus dense fire rate for 649 latents, colored by selection class');
  const mk=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e};
  const ticks=[0,0.001,0.01,0.1];
  for(const v of ticks){
    mk('line',{x1:sx(v),x2:sx(v),y1:M.t,y2:H-M.b,stroke:'var(--line)','stroke-width':1});
    mk('line',{x1:M.l,x2:W-M.r,y1:sy(v),y2:sy(v),stroke:'var(--line)','stroke-width':1});
    const tx=mk('text',{x:sx(v),y:H-M.b+16,'text-anchor':'middle'}); tx.textContent=v===0?'0':v;
    const ty=mk('text',{x:M.l-6,y:sy(v)+4,'text-anchor':'end'}); ty.textContent=v===0?'0':v;
  }
  mk('line',{x1:sx(0),y1:sy(0),x2:sx(0.2),y2:sy(0.2),stroke:'var(--ink3)','stroke-width':1,'stroke-dasharray':'4 4'});
  const xl=mk('text',{x:(M.l+W-M.r)/2,y:H-6,'text-anchor':'middle'}); xl.textContent='fire rate in human span tokens (q span)';
  const yl=mk('text',{x:14,y:(M.t+H-M.b)/2,'text-anchor':'middle',transform:`rotate(-90 14 ${(M.t+H-M.b)/2})`}); yl.textContent='fire rate on dense c1–c4 tokens (q dense)';
  const tip=document.getElementById('tip');
  const groups={};
  for(const c of ['pool','kept','dropped','added']){
    const g=document.createElementNS(NS,'g'); g.dataset.cls=c; svg.appendChild(g); groups[c]=g;
    for(const f of D.feats.filter(f=>f.c===c)){
      const cx=sx(f.qs), cy=sy(f.qd);
      const dot=document.createElementNS(NS,'circle');
      dot.setAttribute('cx',cx); dot.setAttribute('cy',cy);
      dot.setAttribute('r',c==='pool'?2.4:3.4);
      dot.setAttribute('fill',CLS[c].c);
      dot.setAttribute('fill-opacity',c==='pool'?0.35:0.85);
      g.appendChild(dot);
      const hit=document.createElementNS(NS,'circle');
      hit.setAttribute('cx',cx); hit.setAttribute('cy',cy); hit.setAttribute('r',8);
      hit.setAttribute('fill','transparent');
      hit.addEventListener('pointerenter',ev=>{
        dot.setAttribute('r',5.2);
        tip.innerHTML=`<div class="t"><span class="dot" style="background:${CLS[c].c}"></span>${f.l} &middot; ${CLS[c].n}</div>`+
          `<div class="c">${f.cap||'(uncaptioned)'}</div>`+
          `<div>q span ${fmt(f.qs,4)} &middot; q dense ${fmt(f.qd,4)} &middot; r ctl ${fmt(f.rc,4)} &middot; stop ${fmt(f.st,2)}</div>`;
        tip.style.display='block';
      });
      hit.addEventListener('pointermove',ev=>{
        const pad=14, w=tip.offsetWidth, h=tip.offsetHeight;
        tip.style.left=Math.min(ev.clientX+pad, innerWidth-w-8)+'px';
        tip.style.top=Math.min(ev.clientY+pad, innerHeight-h-8)+'px';
      });
      hit.addEventListener('pointerleave',()=>{dot.setAttribute('r',c==='pool'?2.4:3.4);tip.style.display='none';});
      g.appendChild(hit);
    }
  }
  document.getElementById('chart').appendChild(svg);
  const leg=document.getElementById('legend');
  for(const c of ['added','dropped','kept','pool']){
    const b=document.createElement('button');
    b.setAttribute('aria-pressed','true');
    b.innerHTML=`<span class="dot" style="background:${CLS[c].c}"></span>${CLS[c].n} &middot; ${D.counts[c]}`;
    b.addEventListener('click',()=>{
      const on=b.getAttribute('aria-pressed')!=='true';
      b.setAttribute('aria-pressed',String(on));
      groups[c].style.display=on?'':'none';
    });
    leg.appendChild(b);
  }
}
</script>
"""

payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
open(f"{ROOT}/notes/feature_shift_2026-08-15.html", "w").write(HTML.replace("__DATA__", payload))
print("wrote notes/feature_shift_2026-08-15.html,", len(payload)//1024, "KB data")
