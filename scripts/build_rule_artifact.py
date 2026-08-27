"""Generate notes/screening_rules_2026-08-15.html — the 3x-rule artifact."""
import json

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
data = json.load(open(f"{ROOT}/out/artifact_rule_data.json"))

HTML = r"""<title>The 3&times; Rule</title>
<style>
:root{
  --bg:#F7F8F6; --card:#FFFFFF; --ink:#1C2321; --ink2:#566058; --ink3:#7C877F;
  --line:#DCE1DC; --accent:#1F6E5C; --accent-ink:#175243; --chip:#EEF1EC;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
  --warnbg:#FBF3E8; --warnline:#E8D9BF;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#121614; --card:#1B211E; --ink:#E8ECE9; --ink2:#A9B3AD; --ink3:#7C877F;
  --line:#2A322E; --accent:#4FA98A; --accent-ink:#7CC5AB; --chip:#232A26;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
  --warnbg:#2A2620; --warnline:#4A4132;
}}
:root[data-theme="dark"]{
  --bg:#121614; --card:#1B211E; --ink:#E8ECE9; --ink2:#A9B3AD; --ink3:#7C877F;
  --line:#2A322E; --accent:#4FA98A; --accent-ink:#7CC5AB; --chip:#232A26;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
  --warnbg:#2A2620; --warnline:#4A4132;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
  font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:80ch;margin:0 auto;padding:2.5rem 1.25rem 5rem}
h1,h2{font-family:Charter,Georgia,serif;line-height:1.15;text-wrap:balance}
h1{font-size:2.1rem;margin:0 0 .3rem}
h2{font-size:1.3rem;margin:2.6rem 0 .7rem;padding-top:1.2rem;border-top:1px solid var(--line)}
.kicker{font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;color:var(--accent);font-weight:600}
.sub{color:var(--ink2);margin:.2rem 0 0}
p{margin:.75rem 0}
code,.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.86em}
.rulebox{background:var(--card);border:1px solid var(--accent);border-left-width:4px;
  border-radius:8px;padding:1rem 1.2rem;margin:1.3rem 0;font-size:1.05rem}
.rulebox b{color:var(--accent-ink)}
.scroll{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px;margin:.8rem 0}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th{font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2);
  text-align:left;padding:.5rem .6rem;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:.42rem .6rem;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right}
tr.hl td{background:color-mix(in srgb,var(--accent) 9%,transparent);font-weight:600}
.dot{display:inline-block;width:.6em;height:.6em;border-radius:50%;margin-right:.4em}
figure{margin:1.2rem 0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:1rem}
figcaption{font-size:.8rem;color:var(--ink2);margin-top:.5rem}
.legend{display:flex;gap:.5rem;flex-wrap:wrap;font-size:.8rem;color:var(--ink2);margin:0 0 .5rem;align-items:center}
.legend button{all:unset;cursor:pointer;display:inline-flex;align-items:center;gap:.35em;
  padding:.14em .55em;border-radius:999px;border:1px solid var(--line);color:var(--ink2)}
.legend button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.legend button[aria-pressed="false"]{opacity:.38}
.legend .hint{font-size:.72rem;color:var(--ink3);margin-left:.3rem}
g.series{transition:opacity .15s}
g.series.dim{opacity:.13}
.note{background:var(--warnbg);border:1px solid var(--warnline);border-radius:8px;padding:.7rem 1rem;font-size:.9rem}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);
  border-radius:6px;padding:.45rem .65rem;font-size:.78rem;max-width:20rem;display:none;
  box-shadow:0 4px 14px rgba(0,0,0,.14);z-index:9}
svg text{fill:var(--ink2);font:11px system-ui}
svg .dl{font-weight:600;fill:var(--ink)}
</style>
<main>
<div class="kicker">Metagaming filter · TEST v5 · 2026-08-15</div>
<h1>The 3&times; Rule</h1>
<p class="sub">A single pre-registered screening rule matches combinatorially optimized SAE
feature selection — across five judge pools and two SAE widths.</p>

<div class="rulebox">Keep a feature if its above-threshold firing rate on <b>forget</b> text is at
least <b>&rho;&nbsp;= 3&times;</b> its rate on <b>retain</b> text. One statistic per feature, one
universal constant, no search.</div>

<p>The question behind this page: our best feature sets came from hill-climbing over subsets
(~10<sup>4</sup> adaptive metric evaluations — real overfitting capacity). Can a simple rule with
<em>no fitted parameters</em> do the same job? Thresholds below were fixed in advance at round
numbers ({1,&nbsp;2,&nbsp;3,&nbsp;5,&nbsp;10} for the ratio; {1%,&nbsp;&hellip;,&nbsp;0.01%} for the
control cap) and evaluated once, exactly, under the frozen protocol (main forget = human spans +
dense c1&ndash;c4; Rathi firing rule; TEST&nbsp;v5, 2,820 docs).</p>

<h2>F2 is flat in &rho; — the knob sets your FP budget, not your score</h2>
<figure>
  <div class="legend" id="leg1"></div>
  <div id="chart1"></div>
  <figcaption>Each curve traces one pool through &rho; &isin; {1, 2, 3, 5, 10} (right to left:
  looser &rho; &rarr; higher control FP). Diamonds are the hill-climbed optimized sets re-scored on
  the same corpus. Across &rho; 1&rarr;5, F2 moves &le; .02 while control FP falls from ~.27&ndash;.36
  to ~.05&ndash;.08: tightening &rho; trades recall for precision almost exactly F2-neutrally.
  Choose the FP budget; &rho; follows (&rho;=3 &harr; ~10%, &rho;=5 &harr; ~6&ndash;8%,
  &rho;=10 &harr; ~2&ndash;3%). Hover points for exact values; the table below repeats them.</figcaption>
</figure>

<h2>The &rho;=3 and &rho;=5 grid</h2>
<div class="scroll"><table id="t1"></table></div>
<p>The pre-registered &rho;=3 lands within one point of every swept optimum and within ~.01 of full
optimization. Where it runs slightly hot (v8 pools), the next round number &rho;=5 is the
FP-conservative variant — <b>haiku v8 65k at &rho;=5 reaches F2 .904 with recall .924 at 8.2% FP</b>,
matching the searched 65k optimum with zero fitted parameters.</p>

<h2>R1: the control-cap baseline — the least test-coupled rule</h2>
<p>R1 (&ldquo;drop features firing on more than &tau; of retain-control tokens&rdquo;) deserves its
own reading: it is the <b>lightest-touch baseline in the study</b>. It never sees the forget labels
at all — only a corpus of ordinary text — so unlike R3 it cannot absorb any information from the
span annotations or dense docs, and unlike the optimized sets it involves no search whatsoever.
Its curve is the honest answer to &ldquo;what do you get from the judge pool plus a pile of clean
text?&rdquo;</p>
<figure>
  <div class="legend" id="leg2"></div>
  <div id="chart2"></div>
  <figcaption>R1 traced through twelve fixed &tau; values (2% down to 0.01%), same axes and
  optimized-set diamonds as above. Read at the 10% budget line: R1 reaches F2 ~.79&ndash;.81 at 16k
  and ~.74&ndash;.78 at 65k — 6&ndash;10 points under the &rho; curves, but far above the label-free
  rules, and a large share of the total signal considering it uses no forget data. Its two
  structural limits: it keeps quiet-but-useless features and discards moderate-firing recall
  carriers (the gap to R3), and &tau; is width-bound — the same &tau;=0.2% sits at ~10&ndash;15% FP
  on 16k but ~29&ndash;35% on 65k, so the cap must be re-tuned per dictionary while &rho; transfers
  unchanged. Caps stricter than shown (&tau; &lesssim; 0.05% at 16k, &lesssim; 0.02% at 65k) fall
  below the chart floor — F2 collapses toward .2&ndash;.6 as recall carriers are culled.</figcaption>
</figure>

<h2>All six rules, best feasible point (FP &le; 10%)</h2>
<div class="scroll"><table id="t2"></table></div>
<p>R2 (corpus-wide rate cap, fully label-free) fails badly — overall firing rate barely tracks
forget-specificity. R4 (stopword-share cap) is a passable register screen at 65k only. The
Rathi&nbsp;&amp;&nbsp;Radford caption screen (R5: Paulo-style embedding score &ge; 0.9; R6: that
screen then the control cap — their actual pipeline) underperforms the plain control cap here:
the score measures caption <em>faithfulness</em>, not task relevance — a perfectly captioned
&ldquo;list separators&rdquo; feature passes at 1.0 and still floods controls.
(Reproduction caveat: MiniLM embedder + AUC scoring; their embedder is unspecified.)</p>

<h2>Honesty notes</h2>
<p class="note"><b>What &ldquo;pre-registered&rdquo; does and doesn&rsquo;t buy.</b> The &rho; grid
was fixed before evaluation, but the per-feature fire rates are computed on the same TEST corpus the
metric uses, and TEST informed earlier iterations. The rule&rsquo;s ~1 effective parameter can barely
overfit — unlike subset search — which is exactly why its match to the optimized sets argues the
optimizers weren&rsquo;t extracting much signal beyond the ratio statistic. The clean next step:
recompute the ratios on disjoint selection-pool text and re-evaluate; sealed VAL-A/VAL-B remain
unspent and adjudicate finally. Also note the earlier sweep reported <em>count</em> ratios
(&times;23 smaller than rate ratios on this corpus mix); everything on this page is in per-token
<em>rate</em> form — the width-portable, paper-ready number.</p>

<p class="sub" style="font-size:.8rem">Data: <span class="mono">out/fixed_points_*.json</span>,
<span class="mono">out/screen_rules_*.json</span> · report
<span class="mono">notes/reports/screening_rules_2026-08-15.md</span> · scripts
<span class="mono">fixed_ratio_points.py</span>, <span class="mono">screen_rules_sweep.py</span>,
<span class="mono">embed_score_captions.py</span> · protocol v5, manifest sha
<span class="mono">0f6f83c7</span>.</p>
</main>
<div id="tip"></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const COLS = ['var(--s1)','var(--s2)','var(--s3)','var(--s4)','var(--s5)'];
const fmt = (x,d=3)=> x==null?'—':(+x).toFixed(d).replace(/^0\./,'.');
const tip = document.getElementById('tip');
function mkTip(html, ev){
  tip.innerHTML = html; tip.style.display='block';
  const pad=14, w=tip.offsetWidth, h=tip.offsetHeight;
  tip.style.left=Math.min(ev.clientX+pad, innerWidth-w-8)+'px';
  tip.style.top=Math.min(ev.clientY+pad, innerHeight-h-8)+'px';
}
/* pool visibility state shared by both charts */
const poolOn = D.pools.map(()=>true);
const seriesGroups = D.pools.map(()=>[]);   // [poolIdx] -> [<g> in each chart]
const legendBtns = D.pools.map(()=>[]);
function applyState(){
  D.pools.forEach((_,i)=>{
    seriesGroups[i].forEach(g=>g.classList.toggle('dim', !poolOn[i]));
    legendBtns[i].forEach(b=>b.setAttribute('aria-pressed', String(poolOn[i])));
  });
}
function legend(el){
  D.pools.forEach((p,i)=>{
    const b=document.createElement('button');
    b.setAttribute('aria-pressed','true');
    b.innerHTML=`<span class="dot" style="background:${COLS[i]}"></span>${p.name}`;
    b.title='click to dim/undim this family in both charts';
    b.addEventListener('click',()=>{
      const anyOff = poolOn.some(v=>!v);
      if(!anyOff){ poolOn.forEach((_,j)=>poolOn[j] = (j===i)); }   // first click isolates
      else { poolOn[i] = !poolOn[i];
             if(poolOn.every(v=>v===false)) poolOn.forEach((_,j)=>poolOn[j]=true); }
      applyState();
    });
    legendBtns[i].push(b);
    el.appendChild(b);
  });
  const d=document.createElement('span');
  d.innerHTML=`<span style="display:inline-block;width:.7em;height:.7em;background:var(--ink3);transform:rotate(45deg);margin-right:.4em"></span>optimized sets`;
  el.appendChild(d);
  const h=document.createElement('span');
  h.className='hint'; h.textContent='click a family to isolate it; click again to add/remove others';
  el.appendChild(h);
}
legend(document.getElementById('leg1'));
legend(document.getElementById('leg2'));
function curveChart(el, series, refs, xmax){
  const W=680,H=400,M={l:52,r:16,t:12,b:42};
  const sx=v=>M.l+(v/xmax)*(W-M.l-M.r);
  const y0=0.55, y1=0.95;
  const sy=v=>H-M.b-((v-y0)/(y1-y0))*(H-M.t-M.b);
  const NS='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(NS,'svg');
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.setAttribute('style','width:100%;height:auto;display:block');
  svg.setAttribute('role','img');
  const mk=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e};
  for(let v=y0;v<=y1+1e-9;v+=0.1){
    mk('line',{x1:M.l,x2:W-M.r,y1:sy(v),y2:sy(v),stroke:'var(--line)','stroke-width':1});
    const t=mk('text',{x:M.l-6,y:sy(v)+4,'text-anchor':'end'}); t.textContent=v.toFixed(2).replace(/^0/,'');
  }
  for(let v=0;v<=xmax+1e-9;v+=0.1){
    mk('line',{x1:sx(v),x2:sx(v),y1:M.t,y2:H-M.b,stroke:'var(--line)','stroke-width':1});
    const t=mk('text',{x:sx(v),y:H-M.b+16,'text-anchor':'middle'}); t.textContent=(v*100).toFixed(0)+'%';
  }
  mk('line',{x1:sx(0.10),x2:sx(0.10),y1:M.t,y2:H-M.b,stroke:'var(--accent)','stroke-width':1.5,'stroke-dasharray':'5 4'});
  const bl=mk('text',{x:sx(0.10)+4,y:M.t+12}); bl.textContent='10% budget';
  const xt=mk('text',{x:(M.l+W-M.r)/2,y:H-6,'text-anchor':'middle'}); xt.textContent='false-positive rate on retain controls';
  const yt=mk('text',{x:14,y:(M.t+H-M.b)/2,'text-anchor':'middle',transform:`rotate(-90 14 ${(M.t+H-M.b)/2})`}); yt.textContent='main F2 (spans + dense c1–c4)';
  series.forEach((s,i)=>{
    const g=document.createElementNS(NS,'g');
    g.setAttribute('class','series'); svg.appendChild(g);
    const mkg=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);g.appendChild(e);return e};
    const pts=s.pts.filter(p=>p.fp<=xmax+0.02 && p.F2m>=y0);
    mkg('path',{d:pts.map((p,j)=>`${j?'L':'M'}${sx(Math.min(p.fp,xmax))},${sy(p.F2m)}`).join(''),
      fill:'none',stroke:COLS[i],'stroke-width':2,'stroke-linecap':'round'});
    pts.forEach(p=>{
      const c=mkg('circle',{cx:sx(Math.min(p.fp,xmax)),cy:sy(p.F2m),r:4.4,fill:COLS[i],
        stroke:'var(--card)','stroke-width':2});
      const hit=mkg('circle',{cx:sx(Math.min(p.fp,xmax)),cy:sy(p.F2m),r:10,fill:'transparent'});
      hit.addEventListener('pointerenter',ev=>{if(g.classList.contains('dim'))return;
        c.setAttribute('r',6.2);
        mkTip(`<b>${s.name}</b> · ${p.lab}<br>F2 ${fmt(p.F2m)} · R ${fmt(p.Rm)} · fp ${(p.fp*100).toFixed(1)}% · n=${p.n}`,ev);});
      hit.addEventListener('pointermove',ev=>{if(!g.classList.contains('dim'))mkTip(tip.innerHTML,ev);});
      hit.addEventListener('pointerleave',()=>{c.setAttribute('r',4.4);tip.style.display='none';});
    });
    const last=pts[pts.length-1];
    if(last){const dl=mkg('text',{x:sx(Math.min(last.fp,xmax))-8,y:sy(last.F2m)-8,'text-anchor':'end','class':'dl'});
      dl.textContent=s.short;}
    seriesGroups[i].push(g);
  });
  for(const [name,r] of Object.entries(refs)){
    if(r.fp>xmax) continue;
    const x=sx(r.fp), y=sy(r.F2m);
    const d=mk('rect',{x:x-5,y:y-5,width:10,height:10,fill:'var(--ink3)',transform:`rotate(45 ${x} ${y})`,
      stroke:'var(--card)','stroke-width':1.5});
    const hit=mk('circle',{cx:x,cy:y,r:10,fill:'transparent'});
    hit.addEventListener('pointerenter',ev=>mkTip(`<b>${name}</b><br>F2 ${fmt(r.F2m)} · R ${fmt(r.Rm)} · fp ${(r.fp*100).toFixed(1)}%`,ev));
    hit.addEventListener('pointermove',ev=>mkTip(tip.innerHTML,ev));
    hit.addEventListener('pointerleave',()=>tip.style.display='none');
  }
  el.appendChild(svg);
}
const RHOL={"1":"ρ=1","2":"ρ=2","3":"ρ=3","5":"ρ=5","10":"ρ=10"};
const short = n => n.replace('haiku ','').replace(' 16k','·16k').replace(' 65k','·65k');
curveChart(document.getElementById('chart1'),
  D.pools.map(p=>({name:p.name, short:short(p.name),
    pts:Object.entries(p.r3).map(([r,m])=>({lab:RHOL[r], fp:m.fp_ctl, ...m})).sort((a,b)=>a.fp-b.fp)})),
  D.refs, 0.40);
curveChart(document.getElementById('chart2'),
  D.pools.map(p=>({name:p.name, short:short(p.name),
    pts:Object.entries(p.r1).map(([t,m])=>({lab:'τ='+(+t*100)+'%', fp:m.fp_ctl, ...m})).sort((a,b)=>a.fp-b.fp)})),
  D.refs, 0.40);
/* t1: rho grid */
{
  const t=document.getElementById('t1');
  t.innerHTML='<thead><tr><th>pool</th><th class="num">n kept (ρ=3)</th>'+
    '<th class="num">F2 ρ=3</th><th class="num">R ρ=3</th><th class="num">fp ρ=3</th>'+
    '<th class="num">F2 ρ=5</th><th class="num">R ρ=5</th><th class="num">fp ρ=5</th>'+
    '<th class="num">optimized F2 / fp</th></tr></thead><tbody>'+
    D.pools.map((p,i)=>{
      const a=p.r3["3"],b=p.r3["5"];
      const ref={fable_16k:'D2 (optimized, fable)',h6_16k:'h6-16k optimized',h6_65k:'h6-65k optimized'}[p.tag];
      const r=ref?D.refs[ref]:null;
      return `<tr${p.tag==='h8_65k'?' class="hl"':''}><td><span class="dot" style="background:${COLS[i]}"></span>${p.name}</td>`+
        `<td class="num">${a.n}</td><td class="num">${fmt(a.F2m)}</td><td class="num">${fmt(a.Rm)}</td><td class="num">${(a.fp_ctl*100).toFixed(1)}%</td>`+
        `<td class="num">${fmt(b.F2m)}</td><td class="num">${fmt(b.Rm)}</td><td class="num">${(b.fp_ctl*100).toFixed(1)}%</td>`+
        `<td class="num">${r?fmt(r.F2m)+' / '+(r.fp*100).toFixed(1)+'%':'—'}</td></tr>`;}).join('')+'</tbody>';
}
/* t2: rules table */
{
  const t=document.getElementById('t2');
  const RN={R3_ratio:"R3 ratio (swept, for reference)",R1_ctl_rate:"R1 control cap",
    R4_stop_frac:"R4 stopword share",R6_embed09_then_ctl:"R6 embed≥0.9 → ctl cap (Rathi)",
    R5_embed:"R5 embed score alone",R2_corpus_rate:"R2 corpus rate (label-free)"};
  t.innerHTML='<thead><tr><th>rule</th>'+D.pools.map((p,i)=>`<th class="num"><span class="dot" style="background:${COLS[i]}"></span>${p.name}</th>`).join('')+'</tr></thead><tbody>'+
    Object.entries(RN).map(([k,n])=>`<tr${k==='R3_ratio'?' class="hl"':''}><td>${n}</td>`+
      D.pools.map(p=>{const b=p.rules_best_fp10[k];return `<td class="num">${b?fmt(b.F2m):'—'}</td>`;}).join('')+'</tr>').join('')+'</tbody>';
}
</script>
"""

payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
open(f"{ROOT}/notes/screening_rules_2026-08-15.html", "w").write(HTML.replace("__DATA__", payload))
print("wrote notes/screening_rules_2026-08-15.html")
