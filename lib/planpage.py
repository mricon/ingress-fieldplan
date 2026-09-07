# -*- coding: utf-8 -*-
"""
The page template for htmlout.

Kept apart from the logic only because it is long. Two rules govern edits
here: nothing may be fetched from the network (the plan has to work in a
park with no signal), and every tap target has to survive being poked with
a thumb while walking.

Placeholders: __TITLE__, __ACCENT__, __GLOW__, __PLAN__.
"""

TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<meta name="mobile-web-app-capable" content="yes">
<title>__TITLE__</title>
<style>
:root{
  --accent:__ACCENT__; --glow:__GLOW__;
  --bg:#f4f5f7; --card:#ffffff; --ink:#15171a; --dim:#6a7178;
  --line:#e2e5e9; --sunk:#eef0f3; --shadow:0 1px 3px rgba(0,0,0,.10),0 8px 24px rgba(0,0,0,.06);
  --warn:#c2410c;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0d0f12; --card:#171a1f; --ink:#eef1f4; --dim:#949ca4;
    --line:#272c33; --sunk:#111418; --shadow:0 1px 2px rgba(0,0,0,.5);
    --warn:#fb923c;
  }
}
:root[data-theme="dark"]{
  --bg:#0d0f12; --card:#171a1f; --ink:#eef1f4; --dim:#949ca4;
  --line:#272c33; --sunk:#111418; --shadow:0 1px 2px rgba(0,0,0,.5);
  --warn:#fb923c;
}
*{box-sizing:border-box; -webkit-tap-highlight-color:transparent}
html,body{margin:0; height:100%; overscroll-behavior:none}
body{
  background:var(--bg); color:var(--ink);
  font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  display:flex; flex-direction:column;
  padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left);
}
button{font:inherit; color:inherit; background:none; border:0; cursor:pointer}

/* ---------- header ---------- */
header{flex:0 0 auto; padding:6px 12px 8px; background:var(--bg)}
.bar{height:4px; border-radius:4px; background:var(--sunk); overflow:hidden}
.bar i{display:block; height:100%; width:0; background:var(--accent); transition:width .35s ease}
.hud{display:flex; align-items:center; gap:8px; margin-top:7px; font-size:13px}
.hud .where{font-weight:650; font-size:15px}
.hud .of{color:var(--dim)}
.hud .left{margin-left:auto; color:var(--dim); font-variant-numeric:tabular-nums}
.icon{width:36px; height:36px; border-radius:11px; font-size:17px; line-height:1;
      display:grid; place-items:center; color:var(--dim); flex:0 0 auto; opacity:.75}
.icon.on{opacity:1}
.icon.on{background:var(--glow); color:var(--accent)}

/* ---------- the deck ---------- */
.deck{
  /* positioned so a card's offsetLeft is its position in the scroll content;
     without this the offsetParent is the body and the centring maths is off */
  position:relative;
  flex:1 1 auto; min-height:0;
  display:flex; gap:12px; align-items:flex-start;
  overflow-x:auto; overflow-y:hidden;
  scroll-snap-type:x mandatory; scroll-behavior:smooth;
  padding:2px 9vw 10px; scrollbar-width:none;
}
.deck::-webkit-scrollbar{display:none}
.card{
  flex:0 0 min(82vw, 440px); max-height:100%; scroll-snap-align:center; scroll-snap-stop:always;
  background:var(--card); border-radius:18px; box-shadow:var(--shadow);
  overflow-y:auto; overscroll-behavior:contain; padding:14px 15px 18px;
  border:1px solid transparent; transition:border-color .2s, opacity .2s;
}
.card.done{border-color:var(--accent)}
.card.peek{opacity:.55}

/* travel leg */
.leg{display:flex; align-items:center; gap:10px; margin:-2px 0 12px}
.leg .txt{font-size:13.5px; color:var(--dim); line-height:1.3}
.leg .txt b{color:var(--ink); font-weight:650}
.arrow{display:inline-block; transform:rotate(var(--b,0deg)); color:var(--accent)}
.go{
  margin-left:auto; flex:0 0 auto; text-decoration:none;
  background:var(--accent); color:#06130a; font-weight:700; font-size:14px;
  padding:9px 14px; border-radius:11px; white-space:nowrap;
}

/* who */
.who{display:flex; gap:10px; align-items:flex-start; margin-bottom:14px}
.num{
  flex:0 0 auto; width:30px; height:30px; border-radius:9px; background:var(--sunk);
  display:grid; place-items:center; font-size:13px; font-weight:700; color:var(--dim);
}
.card.done .num{background:var(--accent); color:#06130a}
.who h2{margin:0; font-size:22px; line-height:1.2; font-weight:700; letter-spacing:-.01em}
.kind{display:inline-block; margin-top:4px; font-size:11.5px; font-weight:700;
      letter-spacing:.06em; text-transform:uppercase; color:var(--dim)}
.kind.blocker{color:var(--warn)}

/* checklist */
.acts{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:6px}
.act{display:flex; gap:11px; align-items:flex-start; padding:11px 10px; border-radius:12px;
     min-height:48px; background:var(--sunk); transition:background .15s, opacity .15s}
.act:active{background:var(--line)}
.act .box{
  flex:0 0 auto; width:23px; height:23px; margin-top:1px; border-radius:7px;
  border:2px solid var(--line); display:grid; place-items:center;
  font-size:14px; color:transparent; transition:all .15s;
}
.act.on .box{background:var(--accent); border-color:var(--accent); color:#06130a}
.act.on{opacity:.45}
.act.on .t{text-decoration:line-through}
.act .t{font-size:15.5px; font-weight:600; line-height:1.3}
.act .sub{font-size:12.5px; color:var(--dim); margin-top:2px}
.act .glyph{opacity:.85; margin-right:5px}
.act.note{background:transparent; border:1px dashed var(--line); min-height:0; padding:9px 10px}
.act.note .box{display:none}
.act.note .t{font-weight:600; color:var(--dim); font-size:14px}
.chip{display:inline-block; min-width:19px; font-size:12px; font-weight:800;
      text-align:center; padding:1px 5px; border-radius:7px;
      background:var(--glow); color:var(--accent); margin-left:7px; vertical-align:1px}

/* finish card */
.fin{text-align:center; display:flex; flex-direction:column; justify-content:center; gap:6px}
.fin h2{font-size:26px; margin:0}
.fin .big{font-size:40px; font-weight:800; color:var(--accent); letter-spacing:-.02em}
.fin dl{display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:14px 0 0; text-align:left}
.fin dt{font-size:11.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim)}
.fin dd{margin:0; font-size:17px; font-weight:700; font-variant-numeric:tabular-nums}

/* ---------- footer ---------- */
footer{flex:0 0 auto; padding:0 12px 10px; background:var(--bg)}
.upnext{
  display:flex; width:100%; text-align:left; gap:10px; align-items:center;
  padding:10px 12px; border-radius:13px; background:var(--card); box-shadow:var(--shadow);
  margin-bottom:8px; border:1.5px solid transparent; transition:border-color .2s;
}
.upnext.ready{border-color:var(--accent)}
.upnext .lbl{font-size:10.5px; text-transform:uppercase; letter-spacing:.07em; color:var(--dim)}
.upnext.ready .lbl{color:var(--accent)}
.upnext .nm{font-size:15.5px; font-weight:650; line-height:1.25;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.upnext .meta{font-size:12.5px; color:var(--dim)}
.upnext .caret{margin-left:auto; font-size:22px; color:var(--dim)}
.upnext.ready .caret{color:var(--accent)}
.nav{display:flex; gap:8px}
.nav button{
  flex:1; min-height:46px; border-radius:13px; background:var(--card);
  box-shadow:var(--shadow); font-size:14.5px; font-weight:650;
}
.nav button:disabled{opacity:.35}

/* ---------- overview sheet ---------- */
.sheet{
  position:fixed; inset:0; z-index:20; background:var(--bg);
  display:flex; flex-direction:column;
  padding:env(safe-area-inset-top) 0 env(safe-area-inset-bottom);
}
.sheet[hidden]{display:none}
.sh-top{display:flex; align-items:center; gap:10px; padding:12px 14px 8px}
.sh-top h2{margin:0; font-size:19px; flex:1}
.sh-body{flex:1; overflow-y:auto; padding:0 14px 20px}
.map{width:100%; height:190px; background:var(--card); border-radius:14px;
     box-shadow:var(--shadow); display:block}
.mapbar{display:flex; align-items:center; gap:10px; margin:8px 0 14px}
.play{flex:0 0 auto; padding:9px 14px; border-radius:11px; background:var(--card);
      box-shadow:var(--shadow); font-size:13.5px; font-weight:650; min-height:40px}
.play.on{background:var(--accent); color:#06130a}
.cap{font-size:12.5px; color:var(--dim); overflow:hidden;
     text-overflow:ellipsis; white-space:nowrap}
.cap b{color:var(--ink); font-weight:650}
.tot{display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:14px}
.tot div{background:var(--card); border-radius:11px; padding:8px 6px; text-align:center;
         box-shadow:var(--shadow)}
.tot b{display:block; font-size:15px; font-variant-numeric:tabular-nums}
.tot span{font-size:10.5px; text-transform:uppercase; letter-spacing:.05em; color:var(--dim)}
.olist{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:6px}
.olist li{display:flex; gap:11px; align-items:center; padding:11px 12px; min-height:50px;
          background:var(--card); border-radius:12px; box-shadow:var(--shadow)}
.olist li.at{outline:2px solid var(--accent)}
.olist .n{flex:0 0 auto; width:26px; height:26px; border-radius:8px; background:var(--sunk);
          display:grid; place-items:center; font-size:12px; font-weight:700; color:var(--dim)}
.olist li.ok .n{background:var(--accent); color:#06130a}
.olist .nm{flex:1; font-weight:600; font-size:15px; overflow:hidden;
           text-overflow:ellipsis; white-space:nowrap}
.olist .mt{font-size:12px; color:var(--dim); white-space:nowrap}
.reset{width:100%; margin-top:16px; min-height:46px; border-radius:12px;
       border:1.5px solid var(--line); color:var(--warn); font-weight:650}
.hint{font-size:12px; color:var(--dim); text-align:center; margin-top:14px; line-height:1.5}
</style>
</head>
<body>

<header>
  <div class="bar"><i id="fill"></i></div>
  <div class="hud">
    <span class="where" id="hwhere">Stop 1</span>
    <span class="of" id="hof"></span>
    <span class="left" id="hleft"></span>
    <button class="icon" id="awake" title="Keep the screen awake" hidden>&#9749;</button>
    <button class="icon" id="theme" title="Light or dark">&#127765;</button>
  </div>
</header>

<div class="deck" id="deck"></div>

<footer>
  <button class="upnext" id="upnext" hidden></button>
  <nav class="nav">
    <button id="prev">&#8249;&ensp;Back</button>
    <button id="list">All stops</button>
    <button id="next">Next&ensp;&#8250;</button>
  </nav>
</footer>

<div class="sheet" id="sheet" hidden>
  <div class="sh-top">
    <h2>All stops</h2>
    <button class="icon" id="shclose">&#10005;</button>
  </div>
  <div class="sh-body">
    <svg class="map" id="map" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="mapbar">
      <button class="play" id="playbtn">&#9654;&ensp;Preview the run</button>
      <span class="cap" id="mapcap"></span>
    </div>
    <div class="tot" id="tot"></div>
    <ul class="olist" id="olist"></ul>
    <button class="reset" id="reset">Clear all progress</button>
    <p class="hint" id="hint"></p>
  </div>
</div>

<script type="application/json" id="plandata">__PLAN__</script>
<script>
(function(){
"use strict";
var PLAN = JSON.parse(document.getElementById('plandata').textContent);
var STOPS = PLAN.stops, KEY = 'fieldplan:' + PLAN.id;

/* ---------- progress ---------- */
var done = new Set();
var stored = null;
try { stored = localStorage.getItem(KEY); } catch(e){}
if (stored){
  try { done = new Set(JSON.parse(stored)); } catch(e){}
} else {
  // First time this plan is opened: anything already done -- keys you are
  // carrying -- starts ticked. Only ever seeded once, so unticking one and
  // coming back later leaves it unticked.
  STOPS.forEach(function(s){
    s.acts.forEach(function(a){ if (a.pre) done.add(a.id); });
  });
  save();
}
function save(){
  try { localStorage.setItem(KEY, JSON.stringify(Array.from(done))); }
  catch(e){ /* private mode: the plan still works, it just will not remember */ }
}
function todo(s){ return s.acts.filter(function(a){ return !a.note; }); }
function isDone(s){ return todo(s).every(function(a){ return done.has(a.id); }); }

/* ---------- small helpers ---------- */
var CMP = ['N','NE','E','SE','S','SW','W','NW'];
function compass(b){ return CMP[Math.round(b/45) % 8]; }
function metres(m){ return m >= 1000 ? (m/1000).toFixed(1)+' km' : m+' m'; }
/* The travel model rounds to whole minutes, so short hops come back as 0 */
function minutes(t, mode){ return t >= 1 ? (t + ' min' + (mode ? ' ' + mode : ''))
                                         : 'under a minute'; }
function el(tag, cls, txt){
  var e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
}
var GLYPH = {capture:'⬡', keys:'🔑', link:'➤',
             shields:'◉', blocker:'✗', arrive:'⚑'};

/* ---------- build the deck ---------- */
var deck = document.getElementById('deck');
var cards = [];

function buildLeg(s){
  var leg = el('div','leg');
  var t = el('div','txt');
  if (!s.travel){
    t.appendChild(el('b', null, 'Start here'));
  } else if (!s.travel.moved){
    t.appendChild(el('b', null, 'Same spot'));
    t.appendChild(document.createTextNode(' · no need to move'));
  } else {
    var arw = el('span','arrow','➤');
    arw.style.setProperty('--b', (s.travel.b - 90) + 'deg');
    t.appendChild(arw);
    t.appendChild(document.createTextNode(' '));
    t.appendChild(el('b', null, s.travel.nice + ' ' + compass(s.travel.b)));
    t.appendChild(document.createTextNode(' · ' + minutes(s.travel.t, PLAN.modename)));
  }
  leg.appendChild(t);
  if (!s.travel || s.travel.moved){
    var a = el('a','go','Navigate');
    a.href = s.map; a.target = '_blank'; a.rel = 'noopener';
    leg.appendChild(a);
  }
  return leg;
}

function buildAct(act){
  var li = el('li', 'act' + (act.note ? ' note' : ''));
  li.dataset.id = act.id;
  li.dataset.kind = act.k;
  li.appendChild(el('span','box','✓'));
  var body = el('div','body');
  var t = el('div','t');
  t.appendChild(el('span','glyph', GLYPH[act.k] || '•'));
  // No "Link to" prefix: the arrow glyph on the left already says link
  t.appendChild(document.createTextNode(act.txt));
  if (act.k === 'link' && act.link !== 'L'){
    // Just the count -- the green pill already says these are fields
    t.appendChild(el('span','chip', String(act.fields)));
  }
  body.appendChild(t);
  if (act.sub) body.appendChild(el('div','sub', act.sub));
  li.appendChild(body);
  if (!act.note) li.addEventListener('click', function(){ toggle(act.id); });
  return li;
}

function buildCard(s){
  var c = el('article','card');
  c.dataset.i = s.i;
  c.appendChild(buildLeg(s));
  var who = el('div','who');
  who.appendChild(el('div','num', String(s.num)));
  var box = el('div');
  box.appendChild(el('h2', null, s.name));
  var kindtxt = s.kind === 'waypoint' ? 'Waypoint'
              : s.kind === 'blocker' ? 'Blocker — take it down'
              : (s.first ? 'Capture' : 'Return visit');
  box.appendChild(el('div','kind' + (s.kind === 'blocker' ? ' blocker' : ''), kindtxt));
  who.appendChild(box);
  c.appendChild(who);
  var ul = el('ul','acts');
  s.acts.forEach(function(a){ ul.appendChild(buildAct(a)); });
  c.appendChild(ul);
  return c;
}

function buildFinish(){
  var c = el('article','card fin');
  c.dataset.i = STOPS.length;
  c.appendChild(el('div','kind','Plan complete'));
  c.appendChild(el('h2', null, 'Nice run.'));
  c.appendChild(el('div','big', PLAN.stats.ap.toLocaleString() + ' AP'));
  var dl = el('dl');
  [['Distance', PLAN.stats.km + ' km'],
   ['Time', PLAN.stats.time],
   ['Links', String(PLAN.stats.links)],
   ['Fields', String(PLAN.stats.fields)],
   ['Covered', PLAN.stats.sqkm + ' km²'],
   ['AP / min', String(PLAN.stats.appmin)]].forEach(function(row){
    dl.appendChild(el('dt', null, row[0]));
    dl.appendChild(el('dd', null, row[1]));
  });
  c.appendChild(dl);
  return c;
}

STOPS.forEach(function(s){ deck.appendChild(buildCard(s)); });
deck.appendChild(buildFinish());
cards = Array.prototype.slice.call(deck.children);

/* ---------- navigation ---------- */
var at = 0;
function goTo(i, smooth){
  i = Math.max(0, Math.min(cards.length - 1, i));
  var c = cards[i];
  var left = c.offsetLeft - (deck.clientWidth - c.clientWidth) / 2;
  if (smooth === false){
    var prevBehavior = deck.style.scrollBehavior;
    deck.style.scrollBehavior = 'auto';
    deck.scrollLeft = left;
    deck.style.scrollBehavior = prevBehavior;
  } else {
    deck.scrollTo({left: left, behavior: 'smooth'});
  }
  at = i; paint();
}
function nearest(){
  var mid = deck.scrollLeft + deck.clientWidth / 2, best = 0, bd = Infinity;
  for (var i = 0; i < cards.length; i++){
    var d = Math.abs(cards[i].offsetLeft + cards[i].clientWidth / 2 - mid);
    if (d < bd){ bd = d; best = i; }
  }
  return best;
}
var scrollTimer;
deck.addEventListener('scroll', function(){
  clearTimeout(scrollTimer);
  scrollTimer = setTimeout(function(){
    var i = nearest();
    if (i !== at){ at = i; paint(); }
  }, 90);
}, {passive:true});

document.getElementById('prev').addEventListener('click', function(){ goTo(at - 1); });
document.getElementById('next').addEventListener('click', function(){ goTo(at + 1); });
document.addEventListener('keydown', function(e){
  if (e.key === 'ArrowRight') goTo(at + 1);
  else if (e.key === 'ArrowLeft') goTo(at - 1);
  else if (e.key === 'Escape') closeSheet();
});

/* ---------- toggling ---------- */
function toggle(id){
  if (done.has(id)) done.delete(id); else done.add(id);
  save(); paint();
}

/* ---------- painting ---------- */
var fill = document.getElementById('fill'),
    hwhere = document.getElementById('hwhere'), hof = document.getElementById('hof'),
    hleft = document.getElementById('hleft'), upnext = document.getElementById('upnext');

function firstOpen(){
  for (var i = 0; i < STOPS.length; i++){ if (!isDone(STOPS[i])) return i; }
  return STOPS.length;
}

function paint(){
  var total = 0, hit = 0;
  STOPS.forEach(function(s){
    todo(s).forEach(function(a){ total++; if (done.has(a.id)) hit++; });
  });
  fill.style.width = (total ? (hit / total * 100) : 100) + '%';

  cards.forEach(function(c, i){
    c.classList.toggle('peek', i !== at);
    if (i >= STOPS.length){ c.classList.toggle('done', hit === total); return; }
    var s = STOPS[i];
    c.classList.toggle('done', isDone(s));
    Array.prototype.forEach.call(c.querySelectorAll('.act'), function(li){
      li.classList.toggle('on', done.has(li.dataset.id));
    });
  });

  if (at >= STOPS.length){
    hwhere.textContent = 'Finish';
    hof.textContent = '';
  } else {
    hwhere.textContent = 'Stop ' + STOPS[at].num;
    hof.textContent = 'of ' + STOPS.length;
  }

  // What is left to travel from here on
  var far = 0, mins = 0;
  for (var j = at + 1; j < STOPS.length; j++){
    if (STOPS[j].travel){ far += STOPS[j].travel.d; mins += STOPS[j].travel.t; }
  }
  hleft.textContent = far ? (metres(far) + ' · ' + mins + ' min left') : 'last stop';

  paintNext();
  if (!sheet.hidden) paintSheet();
}

/* What is waiting at a stop, in the order it matters while walking there */
function summary(s){
  var bits = [];
  if (s.travel && s.travel.moved) bits.push(s.travel.nice + ' · ' + minutes(s.travel.t));
  if (s.kind === 'blocker') bits.push('take the blocker down');
  s.acts.forEach(function(a){
    if (a.k === 'keys' && !a.note) bits.push(a.txt.toLowerCase());
  });
  var nl = s.acts.filter(function(a){ return a.k === 'link'; }).length;
  if (nl) bits.push(nl + (nl === 1 ? ' link' : ' links'));
  if (!bits.length) bits.push(s.first ? 'capture it' : 'nothing to do');
  return bits.join(' · ');
}

function paintNext(){
  var n = at + 1;
  if (n > STOPS.length){ upnext.hidden = true; return; }
  upnext.hidden = false;
  upnext.textContent = '';
  var ready = at < STOPS.length && isDone(STOPS[at]);
  upnext.classList.toggle('ready', ready);
  var box = el('div');
  box.style.minWidth = '0';
  box.appendChild(el('div','lbl', ready ? 'Done here — go to' : 'Next up'));
  if (n === STOPS.length){
    box.appendChild(el('div','nm','Finish'));
    box.appendChild(el('div','meta', PLAN.stats.ap.toLocaleString() + ' AP total'));
  } else {
    box.appendChild(el('div','nm', STOPS[n].name));
    box.appendChild(el('div','meta', summary(STOPS[n])));
  }
  upnext.appendChild(box);
  upnext.appendChild(el('div','caret','›'));
}
upnext.addEventListener('click', function(){ goTo(at + 1); });

/* ---------- overview ---------- */
var sheet = document.getElementById('sheet'),
    olist = document.getElementById('olist'), tot = document.getElementById('tot');

function paintSheet(){
  tot.textContent = '';
  [[PLAN.stats.ap.toLocaleString(),'AP'], [PLAN.stats.km + ' km','walk'],
   [PLAN.stats.time,'time'], [String(PLAN.stats.fields),'fields']].forEach(function(r){
    var d = el('div'); d.appendChild(el('b', null, r[0]));
    d.appendChild(el('span', null, r[1])); tot.appendChild(d);
  });

  olist.textContent = '';
  STOPS.forEach(function(s, i){
    var li = el('li', (isDone(s) ? 'ok ' : '') + (i === at ? 'at' : ''));
    li.appendChild(el('div','n', String(s.num)));
    li.appendChild(el('div','nm', s.name));
    var n = todo(s).filter(function(a){ return done.has(a.id); }).length;
    li.appendChild(el('div','mt', todo(s).length ? n + '/' + todo(s).length : '—'));
    li.addEventListener('click', function(){ closeSheet(); goTo(i); });
    olist.appendChild(li);
  });
  drawMap();
}

function openSheet(){ sheet.hidden = false; paintSheet(); }
function closeSheet(){ stopPreview(); sheet.hidden = true; }
document.getElementById('list').addEventListener('click', openSheet);
document.getElementById('shclose').addEventListener('click', closeSheet);
document.getElementById('reset').addEventListener('click', function(){
  if (!confirm('Clear every check on this plan?')) return;
  done.clear(); save(); paint(); goTo(0);
});

/* ---------- the map ---------- */
var SVGNS = 'http://www.w3.org/2000/svg';
/* While the preview runs this holds the frame being drawn: which links are
   up yet, which stop we have reached, and how far along the leg out of it
   we are. Null the rest of the time, when the map shows your real progress. */
var anim = null;

function drawMap(){
  var svg = document.getElementById('map');
  var P = PLAN.portals;
  if (!P.length) return;
  var W = svg.clientWidth || 340, H = svg.clientHeight || 190, pad = 14;
  var latm = P.reduce(function(a,p){ return a + p.lat; }, 0) / P.length;
  var kx = Math.cos(latm * Math.PI / 180);
  var xs = P.map(function(p){ return p.lng * kx; }), ys = P.map(function(p){ return -p.lat; });
  var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
  var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
  var sc = Math.min((W - 2*pad) / ((x1 - x0) || 1e-9), (H - 2*pad) / ((y1 - y0) || 1e-9));
  var ox = (W - (x1 - x0) * sc) / 2, oy = (H - (y1 - y0) * sc) / 2;
  function X(i){ return (xs[i] - x0) * sc + ox; }
  function Y(i){ return (ys[i] - y0) * sc + oy; }

  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  svg.textContent = '';
  function add(tag, attrs){
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    svg.appendChild(e);
    return e;
  }
  var accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
  var dimc = getComputedStyle(document.documentElement).getPropertyValue('--line').trim();

  var shown = anim ? anim.done : done;
  var here = anim ? anim.at : at;

  // the route you walk, under everything else: whole thing faint, walked part lit
  function route(from, to, stroke, width, op, tail){
    var pts = [];
    for (var k = from; k <= to && k < STOPS.length; k++) pts.push(X(STOPS[k].node) + ',' + Y(STOPS[k].node));
    if (tail) pts.push(tail[0] + ',' + tail[1]);
    if (pts.length < 2) return;
    add('polyline', {points: pts.join(' '), fill:'none', stroke: stroke,
                     'stroke-width': width, 'stroke-opacity': op,
                     'stroke-linejoin':'round', 'stroke-linecap':'round'});
  }
  // Where the walker is: at a stop, or partway along the leg out of it
  var spot = [X(STOPS[Math.min(here, STOPS.length - 1)].node),
              Y(STOPS[Math.min(here, STOPS.length - 1)].node)];
  if (anim && anim.t > 0 && here + 1 < STOPS.length){
    var nx = X(STOPS[here + 1].node), ny = Y(STOPS[here + 1].node);
    spot = [spot[0] + (nx - spot[0]) * anim.t, spot[1] + (ny - spot[1]) * anim.t];
  }
  route(0, STOPS.length - 1, dimc, 5, 1);
  route(0, here, accent, 5, .55, anim && anim.t > 0 ? spot : null);

  // fields next so links and dots sit on top
  PLAN.links.forEach(function(l){
    if (!shown.has(l.id)) return;
    l.tri.forEach(function(t){
      add('polygon', {points: t.map(function(v){ return X(v) + ',' + Y(v); }).join(' '),
                      fill: accent, 'fill-opacity': .16, stroke: 'none'});
    });
  });
  PLAN.links.forEach(function(l){
    var got = shown.has(l.id);
    add('line', {x1:X(l.a), y1:Y(l.a), x2:X(l.b), y2:Y(l.b),
                 stroke: got ? accent : dimc, 'stroke-width': got ? 1.6 : 1,
                 'stroke-dasharray': got ? '' : '3 3'});
  });
  P.forEach(function(p, i){
    add('circle', {cx:X(i), cy:Y(i), r:p.w ? 2.6 : 3.2,
                   fill: p.w ? dimc : accent, 'fill-opacity': p.w ? 1 : .85});
  });
  if (anim){
    add('circle', {cx:spot[0], cy:spot[1], r:5, fill:accent});
    add('circle', {cx:spot[0], cy:spot[1], r:9, fill:'none',
                   stroke:accent, 'stroke-width':2, 'stroke-opacity':.5});
  } else if (at < STOPS.length){
    add('circle', {cx:spot[0], cy:spot[1], r:8, fill:'none',
                   stroke:accent, 'stroke-width':2});
  }
}

/* ---------- previewing the run ---------- */
var playbtn = document.getElementById('playbtn'), mapcap = document.getElementById('mapcap');
var raf = null;

function previewSteps(){
  // One step per leg walked and per link made, in the order you play them
  var steps = [];
  STOPS.forEach(function(s, i){
    if (i > 0) steps.push({kind:'move', to:i});
    s.acts.forEach(function(a){
      if (a.k === 'link') steps.push({kind:'link', to:i, id:a.id});
    });
  });
  return steps;
}

function caption(text, sub){
  mapcap.textContent = '';
  if (!text) return;
  mapcap.appendChild(el('b', null, text));
  if (sub) mapcap.appendChild(document.createTextNode(' · ' + sub));
}

function stopPreview(){
  if (raf) cancelAnimationFrame(raf);
  raf = null; anim = null;
  playbtn.classList.remove('on');
  playbtn.innerHTML = '&#9654;&ensp;Preview the run';
  caption('');
  if (!sheet.hidden) drawMap();
}

function startPreview(){
  var steps = previewSteps();
  if (!steps.length) return;
  playbtn.classList.add('on');
  playbtn.innerHTML = '&#9632;&ensp;Stop';

  // Whole thing in about nine seconds however long the plan is, but never
  // so fast that a step cannot be seen
  var base = Math.max(70, Math.min(420, 9000 / steps.length));
  var seen = new Set();
  anim = {done: seen, at: 0, t: 0};
  var k = 0, began = null, painted = 0;

  function frame(ts){
    if (!anim) return;
    if (began === null) began = ts;
    var step = steps[k];
    var span = step.kind === 'move' ? base : base * 0.55;
    var p = Math.min(1, (ts - began) / span);

    if (step.kind === 'move'){
      anim.at = step.to - 1; anim.t = p;
    } else {
      anim.at = step.to; anim.t = 0;
      if (p >= 0.4) seen.add(step.id);
    }

    // Rebuilding the whole map every frame is wasteful; 30 fps is plenty
    if (ts - painted > 33){ drawMap(); painted = ts; }

    if (p >= 1){
      if (step.kind === 'move'){ anim.at = step.to; anim.t = 0; }
      var s = STOPS[Math.min(anim.at, STOPS.length - 1)];
      caption(s.name, 'stop ' + s.num + ' of ' + STOPS.length);
      k++; began = ts;
      if (k >= steps.length){
        drawMap();
        caption('Done', PLAN.stats.ap.toLocaleString() + ' AP, ' + PLAN.stats.km + ' km');
        // Leave the finished map up for a moment before handing it back
        raf = null;
        setTimeout(function(){ if (anim) stopPreview(); }, 1400);
        return;
      }
    }
    raf = requestAnimationFrame(frame);
  }
  caption(STOPS[0].name, 'stop 1 of ' + STOPS.length);
  raf = requestAnimationFrame(frame);
}

playbtn.addEventListener('click', function(){
  if (anim) stopPreview(); else startPreview();
});

/* ---------- screen wake lock ---------- */
var awake = document.getElementById('awake'), lock = null, want = false;
if ('wakeLock' in navigator){
  awake.hidden = false;
  awake.addEventListener('click', function(){
    want = !want;
    awake.classList.toggle('on', want);
    if (want) grab(); else release();
  });
  document.addEventListener('visibilitychange', function(){
    if (want && document.visibilityState === 'visible') grab();
  });
}
function grab(){
  if (lock) return;
  navigator.wakeLock.request('screen').then(function(l){
    lock = l;
    l.addEventListener('release', function(){ lock = null; });
  }).catch(function(){ want = false; awake.classList.remove('on'); });
}
function release(){ if (lock){ lock.release(); lock = null; } }

/* ---------- theme ---------- */
var theme = document.getElementById('theme');
try {
  var saved = localStorage.getItem('fieldplan:theme');
  if (saved) document.documentElement.dataset.theme = saved;
} catch(e){}
theme.addEventListener('click', function(){
  var now = document.documentElement.dataset.theme;
  var dark = now ? now === 'dark'
                 : matchMedia('(prefers-color-scheme: dark)').matches;
  var next = dark ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('fieldplan:theme', next); } catch(e){}
  if (!sheet.hidden) drawMap();
});

/* ---------- go ---------- */
document.getElementById('hint').textContent =
  'Progress is kept in this browser only. Plan ' + PLAN.id + '.';
window.addEventListener('resize', function(){ if (!sheet.hidden) drawMap(); });
paint();
goTo(firstOpen(), false);
})();
</script>
</body>
</html>
'''
