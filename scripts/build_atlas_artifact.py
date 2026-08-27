"""Assemble the Generalization Atlas artifact."""
import json

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
data = json.load(open(f"{ROOT}/out/atlas_data.json"))
data["gsplit"] = json.load(open(f"{ROOT}/out/gsplit_stats.json"))

HTML = r"""<title>Generalization Atlas</title>
<style>
:root{
  --bg:#F7F8F6; --card:#FFFFFF; --ink:#1C2321; --ink2:#566058; --ink3:#7C877F;
  --line:#DCE1DC; --accent:#1F6E5C; --accent-ink:#175243; --chip:#EEF1EC;
  --f1:#2a78d6; --f2:#eb6834; --f3:#1baf7a; --f4:#eda100; --f5:#e87ba4; --f6:#008300; --f7:#4a3aa7; --f8:#e34948;
  --barA:rgba(42,120,214,.25); --barB:rgba(235,104,52,.25);
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#121614; --card:#1B211E; --ink:#E8ECE9; --ink2:#A9B3AD; --ink3:#7C877F;
  --line:#2A322E; --accent:#4FA98A; --accent-ink:#7CC5AB; --chip:#232A26;
  --f1:#3987e5; --f2:#d95926; --f3:#199e70; --f4:#c98500; --f5:#d55181; --f6:#00a300; --f7:#9085e9; --f8:#e66767;
  --barA:rgba(57,135,229,.35); --barB:rgba(217,89,38,.35);
}}
:root[data-theme="dark"]{
  --bg:#121614; --card:#1B211E; --ink:#E8ECE9; --ink2:#A9B3AD; --ink3:#7C877F;
  --line:#2A322E; --accent:#4FA98A; --accent-ink:#7CC5AB; --chip:#232A26;
  --f1:#3987e5; --f2:#d95926; --f3:#199e70; --f4:#c98500; --f5:#d55181; --f6:#00a300; --f7:#9085e9; --f8:#e66767;
  --barA:rgba(57,135,229,.35); --barB:rgba(217,89,38,.35);
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:100ch;margin:0 auto;padding:2.2rem 1.2rem 5rem}
h1,h2{font-family:Charter,Georgia,serif;line-height:1.15;text-wrap:balance}
h1{font-size:2rem;margin:0 0 .3rem}
h2{font-size:1.25rem;margin:2.4rem 0 .6rem;padding-top:1.1rem;border-top:1px solid var(--line)}
.kicker{font-size:.7rem;letter-spacing:.13em;text-transform:uppercase;color:var(--accent);font-weight:600}
.sub{color:var(--ink2);font-size:.92rem;max-width:88ch}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:.88em}
.scroll{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px;margin:.7rem 0}
table{border-collapse:collapse;width:100%;font-size:.8rem}
th{font-size:.64rem;letter-spacing:.07em;text-transform:uppercase;color:var(--ink2);
  text-align:left;padding:.45rem .55rem;border-bottom:1px solid var(--line);white-space:nowrap;
  position:sticky;top:0;background:var(--card)}
th.sortable{cursor:pointer}
th.sortable:hover{color:var(--ink)}
td{padding:.32rem .55rem;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right}
.dot{display:inline-block;width:.6em;height:.6em;border-radius:50%;margin-right:.35em}
.cellbar{position:relative}
.cellbar .bar{position:absolute;left:0;top:15%;bottom:15%;background:var(--barA);border-radius:2px;z-index:0}
.cellbar .bar.b{background:var(--barB)}
.cellbar span{position:relative;z-index:1}
.chips{display:flex;flex-wrap:wrap;gap:.4rem;margin:.5rem 0;align-items:center}
.chips .lbl{font-size:.65rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);font-weight:650;margin-right:.2rem}
button.chip{all:unset;cursor:pointer;font-size:.78rem;font-weight:600;color:var(--ink2);
  border:1px solid var(--line);border-radius:999px;padding:.16em .6em;background:var(--card)}
button.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.chip[aria-pressed="true"]{background:var(--chip);color:var(--ink);border-color:var(--accent)}
figure{margin:.9rem 0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.9rem}
figcaption{font-size:.78rem;color:var(--ink2);margin-top:.45rem;max-width:88ch}
.legend{display:flex;flex-wrap:wrap;gap:.8rem;font-size:.78rem;color:var(--ink2);margin:0 0 .4rem}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);
  border-radius:6px;padding:.4rem .6rem;font-size:.75rem;max-width:22rem;display:none;
  box-shadow:0 4px 14px rgba(0,0,0,.14);z-index:9}
svg text{fill:var(--ink2);font:10.5px system-ui}
.note{background:var(--card);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;
  padding:.6rem 1rem;font-size:.85rem;margin:.8rem 0;color:var(--ink2)}
.sizedots{letter-spacing:.1em;color:var(--accent-ink);font-weight:700}
p{max-width:88ch}
</style>
<main>
<div class="kicker">Metagaming filter · quantitative compendium · G-split (seed 20260815) · companion pages: Frontrunner Sets (the short list, F2 in-dist vs held-out) · Rule Reader (token browsing) · The 3× Rule (threshold history)</div>
<h1>Generalization Atlas</h1>
<p class="sub">Every experiment from the qualitative program in one place, organized around a single
question: <b>which design choices changed generalization?</b> Unless marked otherwise, all numbers
are on <b>G-TEST</b> — data the selection step never saw: held-out dense halves, the fully held-out
c2 (AI-safety) domain, all human-labeled rollout spans, and held-out controls. Selection statistics
come from G-TRAIN only (dense c1/c3/c4 halves + control half).
The two central metrics: <b>F2 in-distribution</b> (F2<sub>in</sub> — held-out documents of
trained-on kinds: rollout spans + unseen dense halves vs held-out controls) and
<b>F2 held-out-domain</b> (F2<sub>all</sub> — the same plus all of never-trained c2 counted as
forget; the headline). <b>F2<sub>train</sub></b> anchors the train→test comparison and
<b>gap</b> = F2<sub>train</sub> − F2<sub>in</sub> (bigger = more overfit). None are comparable to
the protocol-v5 F2<sub>main</sub> on earlier scoreboards. For just the short list of best sets on
these two metrics, see the companion <b>Frontrunner Sets</b> page.</p>

<div class="note"><b>The rule's anatomy (all names on this page use it).</b>
<b>Screen</b> — which SAE features are in the set: <i>ratio</i> keeps features firing ≥ρ× more per
token on forget than retain training text; <i>control-cap</i> keeps features firing on ≤τ of control
tokens. <b>Vote k</b> — how many kept features must fire above threshold on the <i>same token</i>
for it to become a seed (the paper uses k=2; the vote does <i>not</i> change which features are
selected). <b>Windowing</b> — the paper's adjacency growth: labels spread from seeds across
neighboring tokens where any kept feature is active. Same-corpus-statistics rows are retained only
as an audit of the old practice and should not be reported as results; haiku-v6 rows are hidden by
default (indistinguishable from v8 — toggle the chips to see them).</div>

<h2>The findings, as F2 changes on held-out data</h2>
<p class="sub">Each finding shown as a matched pair on the same pool: the held-out generalization metric <b>F2<sub>all</sub></b> (spans + in-domain dense + the never-trained c2 domain)
before and after the design choice. Bars run between the two values; the delta is printed.
These are the numbers to quote in the selection write-up.</p>
<div class="scroll"><table id="pairs"></table></div>

<h2>Current best candidates for the final set</h2>
<div class="scroll"><table id="cands"></table></div>
<p class="sub">All with train-only statistics (the honest regime), vote k≥4, windowing on,
mean+4SD thresholds; ranked by F2<sub>all</sub>. Recipe for each: judge pool → ratio screen ρ against
G-TRAIN → Rathi rule with the stated vote. External AUCs for these v8/fable cells are not yet
computed (the v6 twins measured wAUC .74–.76, kAUC .61–.68 — expected to carry over given pool
similarity); run them before the final freeze.</p>

<h2>The choice ledger — what moved generalization</h2>
<div class="scroll"><table id="ledger"></table></div>

<h2>What is in G-TEST (and G-TRAIN)</h2>
<div class="scroll"><table id="gsplit"></table></div>
<p class="sub">All held-out metrics in this atlas are computed on the G-TEST rows: <b>R_span</b> over
the 6,577 human-labeled tokens inside 25 rollout docs (spans never appear in training),
<b>R_c2</b> over the 30 fully held-out AI-safety dense docs, <b>R_in</b> over the held-out halves of
the trained-on dense classes, and <b>fp</b> over the 1,313 held-out controls. Dense c5 (SWE) is
excluded from both sides per the retain-scope ruling. Split is doc-level, seeded (20260815),
inside TEST v5; the sealed VAL-A/VAL-B sets are untouched by all of this.</p>

<h2>Train → test: does training performance predict held-out performance?</h2>
<figure>
  <div class="legend" id="leg1"></div>
  <div id="chart1"></div>
  <figcaption>Each point is one method×pool cell (138 total, hover for identity). The dashed line is
  F2<sub>train</sub> = F2<sub>in</sub>; distance below the line is the generalization gap. Ratio-screen
  points (blue) hug the line; hill-climb (aqua) sits farther below it at 65k; the same-corpus-statistics
  audit points (yellow) shift right of their clean twins — the measured inflation of the old practice.</figcaption>
</figure>

<h2>Does in-domain coverage predict the held-out domain?</h2>
<figure>
  <div class="legend" id="leg2"></div>
  <div id="chart2"></div>
  <figcaption>x = recall on held-out halves of the <i>training</i> dense classes (c1/c3/c4);
  y = recall on the never-seen c2 domain. The ratio screen tracks the diagonal (topical selection
  transfers across domains); the control-cap (orange) falls far below it — capping control firing alone
  keeps domain-arbitrary features. This is the single clearest picture of why the ratio matters.</figcaption>
</figure>

<h2>The recall–FP frontier on held-out data</h2>
<div class="chips" id="poolchips"><span class="lbl">pool</span></div>
<figure>
  <div class="legend" id="leg3"></div>
  <div id="chart3"></div>
  <figcaption>Span recall (human rollout labels, never trained on) vs control FP, per rule family,
  connected along each family's threshold grid. The green vote-k≥4 series shows the
  loosen-screen/tighten-vote frontier sitting above the paper's k=2 rule. Toggle pools above.</figcaption>
</figure>

<h2>Master table — every cell</h2>
<div class="chips" id="famchips"><span class="lbl">family</span></div>
<div class="chips" id="poolchips2"><span class="lbl">pool</span></div>
<div class="scroll" style="max-height:70vh;overflow-y:auto"><table id="master"></table></div>
<p class="sub"><b>Caution on the "all-features" rows:</b> their leading F2_all is an internal-metric
artifact — on real external text they flag 83–90% of all documents and their westover/kimi AUC drops
below the pool rules (see the judge-pool ledger row). Read them as the no-curation baseline, not a
candidate.</p>
<p class="sub">Click a column header to sort. ★ = the threshold the train split itself would select.
All columns except <b>F2_train</b> are G-TEST (held-out) statistics: <b>F2_in</b> (forget = spans +
in-domain dense), <b>F2_all</b> (c2 included as forget), <b>R_span</b> (6,577 human-labeled tokens),
<b>R_c2</b> (held-out domain), <b>R_in</b> (in-domain dense halves), <b>fp_ctl</b> (1,313 held-out
controls), <b>fp_clean</b> (reviewed-clean chat). Bars: blue = F2<sub>all</sub> (the headline generalization metric), orange = fp. Default sort: F2_all.</p>

<h2>Rathi-rule ablations (fixed ρ=5 sets)</h2>
<div class="scroll"><table id="abl"></table></div>
<p class="sub">Threshold SD is inert; windowing carries recall (esp. held-out domain at 16k); the
seed vote is the FP lever. Paper default is sd4 · seed2 · windowed.</p>

<h2>Real-world transfer (fresh corpora, doc-level, no token labels)</h2>
<div class="scroll"><table id="real"></table></div>
<p class="sub">AUC of doc flag-rate separating positives from negatives. Kimi = mean over six
trick-vs-clean arm pairs (umwp / falseqa / if-conflict columns shown; falseqa's inversion is
explained by in-scope c3 education vocabulary in the clean arm). Needham ≈ .5 is the correct
behavior for an <i>awareness</i> filter applied to docs that merely <i>are</i> eval transcripts.</p>

<h2>Split-seed robustness</h2>
<div class="scroll"><table id="seeds"></table></div>

<div class="note"><b>Caveats.</b> One G-split design (three seeds for the headline rules only);
judge pools and threshold grids carry history from earlier TEST work; hill-climb was budget-limited
(~12k evaluations) — the honest claim is that search finds nothing beyond the ratio statistic at
realistic budgets, not that no subset beats it. Sealed VAL-A/VAL-B remain unspent and adjudicate
finally. Data: <span class="mono">out/gen_study_*.json, out/quali_*.json</span>; companion pages:
The 3× Rule (thresholds), Rule Reader (token-level browsing).</div>
</main>
<div id="tip"></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const FAM = {"ratio screen (paper vote k=2)":"--f1","control-cap screen":"--f2",
  "hill-climb search":"--f3","audit: same-corpus stats":"--f4","no-training baselines":"--f5",
  "ratio screen · vote k≥4":"--f6","ratio, ALL features (no judge)":"--f7",
  "control-cap, ALL features":"--f8"};
const FAMS = Object.keys(FAM);
const POOLS = ["fable_16k","haiku_v6_16k","haiku_v8_16k","haiku_v6_65k","haiku_v8_65k",
  "haiku_v8_262k","ALL 16k dictionary","ALL 65k dictionary"];
const col = f => `var(${FAM[f]})`;
const fmt = (x,d=3)=> (x==null||x!==x)?'—':(+x).toFixed(d).replace(/^0\./,'.').replace(/^-0\./,'-.');
const tip = document.getElementById('tip');
function mkTip(html, ev){
  tip.innerHTML = html; tip.style.display='block';
  const pad=14, w=tip.offsetWidth, h=tip.offsetHeight;
  tip.style.left=Math.min(ev.clientX+pad, innerWidth-w-8)+'px';
  tip.style.top=Math.min(ev.clientY+pad, innerHeight-h-8)+'px';
}
const hideTip=()=>tip.style.display='none';

/* findings pairs */
{
  const t=document.getElementById('pairs');
  const lo=0.45, hi=0.85;
  const pct=v=>Math.max(0,Math.min(100,(v-lo)/(hi-lo)*100));
  t.innerHTML='<thead><tr><th>finding</th><th>pool</th><th>from</th><th>to</th>'+
    '<th style="min-width:220px">F2_all (held-out)</th><th class="num">Δ</th></tr></thead><tbody>'+
    D.pairs.map(p=>{
      const l=Math.min(p.a,p.b), r=Math.max(p.a,p.b);
      const up = p.b>=p.a;
      return `<tr><td style="white-space:normal"><b>${p.finding}</b>${p.note?`<br><span style="color:var(--ink3);font-size:.72rem">${p.note}</span>`:''}</td>`+
      `<td>${p.pool.replace('haiku_','h')}</td>`+
      `<td>${p.a_lab} <span class="mono">${fmt(p.a)}</span></td>`+
      `<td>${p.b_lab} <span class="mono">${fmt(p.b)}</span></td>`+
      `<td><div style="position:relative;height:14px;background:var(--chip);border-radius:4px">`+
      `<div style="position:absolute;left:${pct(l)}%;width:${Math.max(pct(r)-pct(l),1)}%;top:2px;bottom:2px;`+
      `background:${up?'var(--f1)':'var(--f2)'};border-radius:3px;opacity:.75"></div>`+
      `<div style="position:absolute;left:${pct(p.b)}%;top:0;bottom:0;width:3px;background:var(--ink);border-radius:2px"></div>`+
      `</div></td>`+
      `<td class="num" style="font-weight:650;color:${up?'var(--accent-ink)':'var(--f2)'}">${p.b>p.a?'+':''}${(p.b-p.a).toFixed(3).replace(/^(-?)0\./,'$1.')}</td></tr>`;}).join('')+'</tbody>';
}
/* candidates */
{
  const t=document.getElementById('cands');
  t.innerHTML='<thead><tr><th>candidate</th><th class="num">n</th><th class="num">F2_all</th>'+
    '<th class="num">F2_in</th><th class="num">R_span</th><th class="num">R_c2</th>'+
    '<th class="num">fp</th><th class="num">westover AUC</th><th class="num">kimi AUC</th></tr></thead><tbody>'+
    D.candidates.map((c,i)=>`<tr${i===0?' class="hl"':''}><td style="white-space:normal">${c.name}</td>`+
      `<td class="num">${c.n}</td><td class="num" style="font-weight:650">${fmt(c.F2all)}</td><td class="num">${fmt(c.F2in)}</td>`+
      `<td class="num">${fmt(c.Rspan)}</td><td class="num">${fmt(c.Rc2)}</td><td class="num">${fmt(c.fp,4)}</td>`+
      `<td class="num">${c.wauc?fmt(c.wauc):'—'}</td><td class="num">${c.kauc?fmt(c.kauc):'—'}</td></tr>`).join('')+'</tbody>';
}
/* ledger */
{
  const t=document.getElementById('ledger');
  t.innerHTML='<thead><tr><th>design choice</th><th>measured effect on generalization</th><th>verdict</th><th>size</th></tr></thead><tbody>'+
    D.ledger.map(l=>`<tr><td style="white-space:normal"><b>${l.choice}</b></td>`+
      `<td style="white-space:normal">${l.effect}</td><td>${l.verdict}</td>`+
      `<td><span class="sizedots">${'●'.repeat(l.size)||'·'}</span></td></tr>`).join('')+'</tbody>';
}

/* g-split table */
{
  const t=document.getElementById('gsplit');
  const ORDER=[["train_dense","G-TRAIN · dense c1/c3/c4 halves","selection forget signal"],
    ["train_ctl","G-TRAIN · control half","selection retain signal (rates, FP objective)"],
    ["gtest_span","G-TEST · human-labeled rollouts","span recall (R_span) + precision"],
    ["gtest_c2","G-TEST · dense c2 (AI-safety)","held-out DOMAIN (R_c2) — never trained on"],
    ["gtest_dense","G-TEST · dense c1/c3/c4 halves","in-domain generalization (R_in)"],
    ["gtest_clean","G-TEST · reviewed-clean chat","register-matched FP (fp_clean)"],
    ["gtest_ctl","G-TEST · control half","held-out FP (fp_ctl)"],
    ["excluded_c5","excluded · dense c5 (SWE)","retain-scope ruling — in neither side"]];
  t.innerHTML='<thead><tr><th>component</th><th class="num">docs</th><th class="num">tokens</th>'+
    '<th class="num">labeled span tokens</th><th>role in the metrics</th></tr></thead><tbody>'+
    ORDER.map(([k,name,use])=>{const g=D.gsplit[k]||{docs:0,tokens:0,gt_tokens:0};
      return `<tr${k.startsWith('gtest')?'':' style="color:var(--ink2)"'}><td>${name}</td>`+
      `<td class="num">${g.docs.toLocaleString()}</td><td class="num">${g.tokens.toLocaleString()}</td>`+
      `<td class="num">${g.gt_tokens?g.gt_tokens.toLocaleString():'—'}</td>`+
      `<td style="white-space:normal">${use}</td></tr>`;}).join('')+'</tbody>';
}

/* scatter helper */
function scatter(el, pts, xk, yk, xlab, ylab, lo, hi){
  const W=660,H=420,M={l:50,r:14,t:10,b:40};
  const sx=v=>M.l+(v-lo)/(hi-lo)*(W-M.l-M.r);
  const sy=v=>H-M.b-(v-lo)/(hi-lo)*(H-M.t-M.b);
  const NS='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(NS,'svg');
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.setAttribute('style','width:100%;height:auto;display:block');
  svg.setAttribute('role','img');
  const mk=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e};
  for(let v=Math.ceil(lo*10)/10; v<=hi+1e-9; v+=0.1){
    mk('line',{x1:sx(v),x2:sx(v),y1:M.t,y2:H-M.b,stroke:'var(--line)','stroke-width':1});
    mk('line',{x1:M.l,x2:W-M.r,y1:sy(v),y2:sy(v),stroke:'var(--line)','stroke-width':1});
    const tx=mk('text',{x:sx(v),y:H-M.b+14,'text-anchor':'middle'}); tx.textContent=v.toFixed(1).replace(/^0/,'');
    const ty=mk('text',{x:M.l-5,y:sy(v)+4,'text-anchor':'end'}); ty.textContent=v.toFixed(1).replace(/^0/,'');
  }
  mk('line',{x1:sx(lo),y1:sy(lo),x2:sx(hi),y2:sy(hi),stroke:'var(--ink3)','stroke-width':1,'stroke-dasharray':'4 4'});
  const xt=mk('text',{x:(M.l+W-M.r)/2,y:H-5,'text-anchor':'middle'}); xt.textContent=xlab;
  const yt=mk('text',{x:13,y:(M.t+H-M.b)/2,'text-anchor':'middle',transform:`rotate(-90 13 ${(M.t+H-M.b)/2})`}); yt.textContent=ylab;
  for(const p of pts){
    if(p[xk]==null||p[yk]==null) continue;
    const c=mk('circle',{cx:sx(p[xk]),cy:sy(p[yk]),r:4.2,fill:col(p.family),
      stroke:'var(--card)','stroke-width':1.5,'fill-opacity':.9});
    const hit=mk('circle',{cx:sx(p[xk]),cy:sy(p[yk]),r:9,fill:'transparent'});
    hit.addEventListener('pointerenter',ev=>{c.setAttribute('r',6);
      mkTip(`<b>${p.method}</b> · ${p.pool}<br>${xlab}: ${fmt(p[xk])} · ${ylab}: ${fmt(p[yk])}`+
        `<br>R_span ${fmt(p.R_span)} · fp ${fmt(p.fp_ctl)}`,ev);});
    hit.addEventListener('pointermove',ev=>mkTip(tip.innerHTML,ev));
    hit.addEventListener('pointerleave',()=>{c.setAttribute('r',4.2);hideTip();});
  }
  el.appendChild(svg);
}
function legend(el, fams){
  el.innerHTML = fams.map(f=>`<span><span class="dot" style="background:${col(f)}"></span>${f}</span>`).join('');
}
legend(document.getElementById('leg1'), FAMS.filter(f=>f!=="ratio screen · vote k≥4"));
scatter(document.getElementById('chart1'),
        D.rows.filter(r=>r.F2_train!=null&&r.F2_in!=null&&r.F2_in>=0.2),
        'F2_train','F2_in','F2 on G-TRAIN','F2 on G-TEST (held out)',0.2,0.85);
legend(document.getElementById('leg2'), ["ratio screen (paper vote k=2)","control-cap screen","hill-climb search","no-training baselines"]);
scatter(document.getElementById('chart2'),
        D.rows.filter(r=>r.R_in!=null&&r.R_c2!=null&&r.R_in>=0.2&&["ratio screen (paper vote k=2)","control-cap screen","hill-climb search","no-training baselines"].includes(r.family)),
        'R_in','R_c2','recall, in-domain dense (held-out halves)','recall, held-out c2 domain',0.2,1.0);

/* frontier chart with pool chips */
const fstate={pools:new Set(["haiku_v8_65k"])};
const pc=document.getElementById('poolchips');
for(const p of POOLS){
  const b=document.createElement('button'); b.className='chip';
  b.setAttribute('aria-pressed', String(fstate.pools.has(p)));
  b.textContent=p.replace('haiku_','h').replace('fable_','fable ');
  b.addEventListener('click',()=>{
    fstate.pools.has(p)?fstate.pools.delete(p):fstate.pools.add(p);
    if(!fstate.pools.size) fstate.pools.add(p);
    b.setAttribute('aria-pressed', String(fstate.pools.has(p)));
    drawFrontier();
  });
  pc.appendChild(b);
}
legend(document.getElementById('leg3'), ["ratio screen (paper vote k=2)","control-cap screen","ratio screen · vote k≥4","ratio, ALL features (no judge)","hill-climb search","no-training baselines"]);
function drawFrontier(){
  const el=document.getElementById('chart3'); el.innerHTML='';
  const W=660,H=420,M={l:50,r:14,t:10,b:40},xmax=0.45;
  const sx=v=>M.l+Math.min(v,xmax)/xmax*(W-M.l-M.r);
  const y0=0.3,y1=1.0;
  const sy=v=>H-M.b-(v-y0)/(y1-y0)*(H-M.t-M.b);
  const NS='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(NS,'svg');
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.setAttribute('style','width:100%;height:auto;display:block');
  const mk=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e};
  for(let v=0;v<=xmax+1e-9;v+=0.1){
    mk('line',{x1:sx(v),x2:sx(v),y1:M.t,y2:H-M.b,stroke:'var(--line)','stroke-width':1});
    const t=mk('text',{x:sx(v),y:H-M.b+14,'text-anchor':'middle'}); t.textContent=(v*100).toFixed(0)+'%';
  }
  for(let v=y0;v<=y1+1e-9;v+=0.1){
    mk('line',{x1:M.l,x2:W-M.r,y1:sy(v),y2:sy(v),stroke:'var(--line)','stroke-width':1});
    const t=mk('text',{x:M.l-5,y:sy(v)+4,'text-anchor':'end'}); t.textContent=v.toFixed(1).replace(/^0/,'');
  }
  const xt=mk('text',{x:(M.l+W-M.r)/2,y:H-5,'text-anchor':'middle'}); xt.textContent='false-positive rate on held-out controls';
  const yt=mk('text',{x:13,y:(M.t+H-M.b)/2,'text-anchor':'middle',transform:`rotate(-90 13 ${(M.t+H-M.b)/2})`}); yt.textContent='span recall (held-out human labels)';
  for(const fam of ["ratio screen (paper vote k=2)","control-cap screen","ratio screen · vote k≥4","ratio, ALL features (no judge)","control-cap, ALL features","hill-climb search","no-training baselines"]){
    for(const pool of fstate.pools){
      const pts=D.rows.filter(r=>r.family===fam&&r.pool===pool&&r.R_span!=null&&r.fp_ctl!=null
        &&r.fp_ctl<=xmax&&r.R_span>=y0).sort((a,b)=>a.fp_ctl-b.fp_ctl);
      if(!pts.length) continue;
      if(pts.length>1&&fam!=="hill-climb"&&fam!=="no-training baseline")
        mk('path',{d:pts.map((p,j)=>`${j?'L':'M'}${sx(p.fp_ctl)},${sy(p.R_span)}`).join(''),
          fill:'none',stroke:col(fam),'stroke-width':1.8,'stroke-opacity':.75});
      for(const p of pts){
        const c=mk('circle',{cx:sx(p.fp_ctl),cy:sy(p.R_span),r:4,fill:col(fam),
          stroke:'var(--card)','stroke-width':1.5});
        const hit=mk('circle',{cx:sx(p.fp_ctl),cy:sy(p.R_span),r:9,fill:'transparent'});
        hit.addEventListener('pointerenter',ev=>{c.setAttribute('r',6);
          mkTip(`<b>${p.method}</b> · ${p.pool}<br>R_span ${fmt(p.R_span)} · fp ${fmt(p.fp_ctl)} · Rc2 ${fmt(p.R_c2)} · n=${p.n||'—'}`,ev);});
        hit.addEventListener('pointermove',ev=>mkTip(tip.innerHTML,ev));
        hit.addEventListener('pointerleave',()=>{c.setAttribute('r',4);hideTip();});
      }
    }
  }
  el.appendChild(svg);
}
drawFrontier();

/* master table */
const mstate={fams:new Set(FAMS), pools:new Set(POOLS.filter(p=>!p.includes('_v6_'))), sort:'F2_all', dir:-1};
function chips(el, items, set, redraw){
  for(const it of items){
    const b=document.createElement('button'); b.className='chip';
    b.setAttribute('aria-pressed', String(set.has(it)));
    b.innerHTML = FAM[it]?`<span class="dot" style="background:${col(it)}"></span>${it}`:it.replace('haiku_','h');
    b.addEventListener('click',()=>{
      set.has(it)?set.delete(it):set.add(it);
      if(!set.size) items.forEach(x=>set.add(x));
      b.setAttribute('aria-pressed', String(set.has(it)));
      redraw();
    });
    el.appendChild(b);
  }
}
chips(document.getElementById('famchips'), FAMS, mstate.fams, drawMaster);
chips(document.getElementById('poolchips2'), POOLS, mstate.pools, drawMaster);
const COLS=[["pool",0],["method",0],["n",1],["F2_train",1],["F2_in",1],["F2_all",1],["gap (train−test)",1],
  ["R_span",1],["R_c2",1],["R_in",1],["fp_ctl",1],["fp_clean",1]];
function drawMaster(){
  const t=document.getElementById('master');
  const rows=D.rows.filter(r=>mstate.fams.has(r.family)&&mstate.pools.has(r.pool));
  rows.sort((a,b)=>{
    const key=mstate.sort==='gap (train−test)'?'gap':mstate.sort;const va=a[key], vb=b[key];
    if(va==null) return 1; if(vb==null) return -1;
    return (va<vb?-1:va>vb?1:0)*mstate.dir;
  });
  t.innerHTML='<thead><tr>'+COLS.map(([c,num])=>
    `<th class="sortable${num?' num':''}" data-c="${c}">${c.replace('_','&#8202;')}${mstate.sort===c?(mstate.dir<0?' ▾':' ▴'):''}</th>`).join('')+'</tr></thead><tbody>'+
    rows.map(r=>{
      const f2w=r.F2_all==null?0:Math.max(0,(r.F2_all-0.2)/0.7*100);
      const fpw=r.fp_ctl==null?0:Math.min(100,r.fp_ctl/0.45*100);
      return `<tr><td><span class="dot" style="background:${col(r.family)}"></span>${r.pool.replace('haiku_','h')}</td>`+
      `<td class="mono">${r.method}${r.sel||''}</td><td class="num">${r.n??'—'}</td>`+
      `<td class="num">${fmt(r.F2_train)}</td>`+
      `<td class="num">${fmt(r.F2_in)}</td>`+
      `<td class="num cellbar"><div class="bar" style="width:${f2w}%"></div><span>${fmt(r.F2_all)}</span></td>`+
      `<td class="num">${fmt(r.gap)}</td><td class="num">${fmt(r.R_span)}</td>`+
      `<td class="num">${fmt(r.R_c2)}</td><td class="num">${fmt(r.R_in)}</td>`+
      `<td class="num cellbar"><div class="bar b" style="width:${fpw}%"></div><span>${fmt(r.fp_ctl)}</span></td>`+
      `<td class="num">${fmt(r.fp_clean)}</td></tr>`;}).join('')+'</tbody>';
  t.querySelectorAll('th.sortable').forEach(h=>h.addEventListener('click',()=>{
    const c=h.dataset.c;
    if(mstate.sort===c) mstate.dir*=-1; else {mstate.sort=c; mstate.dir=-1;}
    drawMaster();
  }));
}
drawMaster();

/* ablation table */
{
  const t=document.getElementById('abl');
  const rows=D.ablation.sort((a,b)=>a.width.localeCompare(b.width)||a.sd-b.sd||a.seed-b.seed||(b.win-a.win));
  t.innerHTML='<thead><tr><th>width</th><th>threshold</th><th>seed vote</th><th>window</th>'+
    '<th class="num">R_span</th><th class="num">R_c2</th><th class="num">fp</th><th class="num">F2_in</th></tr></thead><tbody>'+
    rows.map(r=>{
      const def = r.sd===4&&r.seed===2&&r.win;
      return `<tr${def?' style="font-weight:650"':''}><td>${r.width}</td><td>mean+${r.sd}SD</td>`+
      `<td>≥${r.seed}</td><td>${r.win?'on':'off'}</td>`+
      `<td class="num">${fmt(r.R_span)}</td><td class="num">${fmt(r.R_c2)}</td>`+
      `<td class="num">${fmt(r.fp_ctl,4)}</td><td class="num cellbar"><div class="bar" style="width:${Math.max(0,(r.F2_in-0.4)/0.45*100)}%"></div><span>${fmt(r.F2_in)}</span></td></tr>`;}).join('')+'</tbody>';
}
/* real-world table */
{
  const t=document.getElementById('real');
  t.innerHTML='<thead><tr><th>rule</th><th class="num">kimi mean AUC</th><th class="num">umwp</th>'+
    '<th class="num">falseqa</th><th class="num">if-conflict</th><th class="num">westover AUC</th>'+
    '<th class="num">needham AUC</th><th class="num">agentic notice AUC</th></tr></thead><tbody>'+
    D.real.map(r=>`<tr><td><b>${r.code}</b> ${r.name}</td>`+
      `<td class="num">${fmt(r.kimi_mean_auc)}</td><td class="num">${fmt(r.kimi_umwp,2)}</td>`+
      `<td class="num">${fmt(r.kimi_falseqa,2)}</td><td class="num">${fmt(r.kimi_ifc,2)}</td>`+
      `<td class="num"><b>${fmt(r.westover_auc)}</b></td><td class="num">${fmt(r.needham_auc)}</td>`+
      `<td class="num">${fmt(r.agentic_notice_auc)}</td></tr>`).join('')+'</tbody>';
}
/* seeds table */
{
  const t=document.getElementById('seeds');
  t.innerHTML='<thead><tr><th>pool</th><th>split seed</th><th class="num">ratio ρ≥10 F2</th>'+
    '<th class="num">R3 ρ=10 R_span</th><th class="num">R3 ρ=5 F2</th><th class="num">R1-best F2</th></tr></thead><tbody>'+
    D.seeds.map(s=>`<tr><td>${s.pool.replace('haiku_','h')}</td><td class="mono">${s.seed}</td>`+
      `<td class="num">${fmt(s.rho10_F2)}</td><td class="num">${fmt(s.rho10_Rspan)}</td>`+
      `<td class="num">${fmt(s.rho5_F2)}</td><td class="num">${fmt(s.R1_F2)}</td></tr>`).join('')+'</tbody>';
}
</script>
"""
payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
open(f"{ROOT}/notes/gen_atlas.html", "w").write(HTML.replace("__DATA__", payload))
print(f"wrote notes/gen_atlas.html ({(len(HTML)+len(payload))//1024} KB)")
