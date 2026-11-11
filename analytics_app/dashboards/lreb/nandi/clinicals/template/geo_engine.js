/* ===========================================================================
   Afyanalytics · Nandi OPD · geo map engine
   Loaded inside an iframe by template/geo_map.py. Expects:
     window.KP       county outlines {county: svgPathData}
     window.KLL      the same outlines as lon/lat rings (for the Leaflet basemap)
     window.PAYLOAD  data prepared by views/views_geo.py
     window.KIND     one of: catchment | rings | disease | shift | female | ncd
   County outlines are real; sub-county and ward shapes come from P.bounds
   (BOUNDARY_GEOJSON in the geo lookup table) when present, otherwise they are
   a schematic subdivision of the county outline.
   =========================================================================== */
(function(){
const NS='http://www.w3.org/2000/svg';
const C={primary:'#0F6E56',blue:'#0C447C',amber:'#D97706',crit:'#A32D2D',neutral:'#888780',text:'#111827',muted:'#6B7280',border:'#E5E7EB',s2:'#F0F1F5',grey:'#E5E7EB'};
const el=(t,a,p)=>{const e=document.createElementNS(NS,t);for(const k in (a||{}))e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e};
const tx=(p,s,a)=>{const t=el('text',a,p);t.textContent=s;return t};
const H=(tag,cls,html,p)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(html!=null)e.innerHTML=html;if(p)p.appendChild(e);return e};
const fmt=n=>Math.round(n).toLocaleString('en-US');
const nk=s=>String(s==null?'':s).toLowerCase().replace(/[^a-z]/g,'');
const tip=H('div','tip','',document.body);
function hover(n,html){n.addEventListener('mousemove',e=>{tip.innerHTML=html;tip.style.display='block';tip.style.left=Math.min(e.clientX+12,innerWidth-190)+'px';tip.style.top=(e.clientY+12)+'px'});n.addEventListener('mouseleave',()=>tip.style.display='none')}
const rnd=s=>{let h=2166136261;for(const c of String(s)){h^=c.charCodeAt(0);h=Math.imul(h,16777619)}h^=h>>>15;h=Math.imul(h,2246822507);h^=h>>>13;h=Math.imul(h,3266489909);h^=h>>>16;return((h>>>0)%10000)/10000};
const sc=(t,col)=>({color:col,op:+(0.12+0.88*Math.sqrt(Math.max(0,Math.min(1,t)))).toFixed(2)});
const clampPct=x=>Math.max(0,Math.min(1,x));

/* ---------- UI controls ---------- */
function seg(parent,opts,val,cb){const d=H('span','seg','',parent);opts.forEach(([v,l])=>{const b=H('button',v===val?'on':'',l,d);b.onclick=()=>{d.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');cb(v)}});return d}
function pills(parent,opts,val,cb){const d=H('div','pills','',parent);const draw=v=>d.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.v===v));opts.forEach(o=>{const b=H('button','',o,d);b.dataset.v=o;b.onclick=()=>{draw(o);cb(o)}});draw(val);return d}
function sel(parent,label,opts,val,cb){const l=H('label','',label+' ',parent);const s=H('select','','',l);opts.forEach(([v,t])=>{const o=H('option','',t,s);o.value=v});s.value=val;s.onchange=()=>cb(s.value);return s}

/* ---------- geometry ---------- */
const KP=window.KP,P=window.PAYLOAD;
const KN=Object.keys(KP),KC={};KN.forEach(k=>KC[nk(k)]=k);
const ALIAS={nairobicity:'Nairobi',taitataveta:'Taita Taveta',muranga:"Murang'a",elgeyomarakwet:'Elgeyo-Marakwet',tharakanithi:'Tharaka-Nithi',transnzoia:'Trans Nzoia',uasingishu:'Uasin Gishu',homabay:'Homa Bay',tanariver:'Tana River',westpokot:'West Pokot'};
const canon=n=>KC[nk(n)]||ALIAS[nk(n)]||null;
const meas=el('svg',{style:'position:absolute;width:0;height:0;visibility:hidden'},document.body);
const KE={};KN.forEach(n=>{KE[n]=el('path',{d:KP[n]},meas)});
const BB={};KN.forEach(n=>{const b=KE[n].getBBox();BB[n]={x:b.x,y:b.y,w:b.width,h:b.height}});
const KBB=(()=>{let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;KN.forEach(n=>{const b=BB[n];x0=Math.min(x0,b.x);y0=Math.min(y0,b.y);x1=Math.max(x1,b.x+b.w);y1=Math.max(y1,b.y+b.h)});return{x:x0,y:y0,w:x1-x0,h:y1-y0}})();
const KMU=1033/KBB.h; /* km per svg unit (Kenya spans ~9.7 degrees of latitude) */
const inside=(n,x,y)=>KE[n].isPointInFill(new DOMPoint(x,y));
function samples(n,k,seed,filter){const b=BB[n],out=[];let t=0;while(out.length<k&&t<k*60){t++;const x=b.x+rnd(seed+'x'+t)*b.w,y=b.y+rnd(seed+'y'+t)*b.h;if(inside(n,x,y)&&(!filter||filter(x,y)))out.push([x,y])}return out}
const CEN={};KN.forEach(n=>{const s=samples(n,200,'c'+n);CEN[n]=[s.reduce((a,p)=>a+p[0],0)/s.length,s.reduce((a,p)=>a+p[1],0)/s.length]});
const pos=(lat,lon)=>[KBB.x+(lon-33.91)/(41.91-33.91)*KBB.w,KBB.y+(5.03-lat)/(5.03+4.68)*KBB.h];
const HOSP=(()=>{const h=P.hospital||{lat:0.2030,lon:35.1050};const p=pos(h.lat,h.lon);return inside('Nandi',p[0],p[1])?p:CEN.Nandi})();
const half=(poly,a,b)=>{const mx=(a[0]+b[0])/2,my=(a[1]+b[1])/2,dx=b[0]-a[0],dy=b[1]-a[1];const f=p=>(p[0]-mx)*dx+(p[1]-my)*dy;const out=[];for(let i=0;i<poly.length;i++){const p=poly[i],q=poly[(i+1)%poly.length],fp=f(p),fq=f(q);if(fp<=0)out.push(p);if((fp<0&&fq>0)||(fp>0&&fq<0)){const t=fp/(fp-fq);out.push([p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])])}}return out};
const clipConvex=(poly,clip)=>{let out=poly;for(let i=0;i<clip.length&&out.length;i++){const a=clip[i],b=clip[(i+1)%clip.length];const nx=b[1]-a[1],ny=a[0]-b[0];const side=p=>(p[0]-a[0])*nx+(p[1]-a[1])*ny;const cx=clip.reduce((s,p)=>s+p[0],0)/clip.length,cy=clip.reduce((s,p)=>s+p[1],0)/clip.length;const sg=side([cx,cy])>=0?1:-1;const inp=out;out=[];for(let j=0;j<inp.length;j++){const p=inp[j],q=inp[(j+1)%inp.length],fp=side(p)*sg,fq=side(q)*sg;if(fp>=0)out.push(p);if((fp<0&&fq>0)||(fp>0&&fq<0)){const t=fp/(fp-fq);out.push([p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])])}}}return out};
const boxPoly=b=>{const m=b.w*.05;return[[b.x-m,b.y-m],[b.x+b.w+m,b.y-m],[b.x+b.w+m,b.y+b.h+m],[b.x-m,b.y+b.h+m]]};
const voronoi=(seeds,bound)=>seeds.map((s,i)=>{let p=bound;seeds.forEach((o,j)=>{if(i!==j&&p.length)p=half(p,s,o)});return p});
const pip=(poly,x,y)=>{let c=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const a=poly[i],b=poly[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])c=!c}return c};
const area=p=>Math.abs(p.reduce((s,a,i)=>{const b=p[(i+1)%p.length];return s+a[0]*b[1]-b[0]*a[1]},0))/2;
const pcen=p=>{let A=0,x=0,y=0;p.forEach((a,i)=>{const b=p[(i+1)%p.length],c=a[0]*b[1]-b[0]*a[1];A+=c;x+=(a[0]+b[0])*c;y+=(a[1]+b[1])*c});A/=2;return A?[x/(6*A),y/(6*A)]:p[0]};
const ptsd=p=>'M'+p.map(q=>q[0].toFixed(2)+','+q[1].toFixed(2)).join('L')+'Z';
function place(n,k,seed,filter,init){const pts=samples(n,500,seed,filter);if(!pts.length)return[];let S=init?init.slice():pts.slice(0,k).map((_,i)=>pts[Math.floor(i*pts.length/k)]);while(S.length<k)S.push(pts[S.length%pts.length]);
 for(let it=0;it<(init?2:7);it++){const sum=S.map(()=>[0,0,0]);pts.forEach(p=>{let b=0,bd=1e18;S.forEach((s,i)=>{const d=(s[0]-p[0])**2+(s[1]-p[1])**2;if(d<bd){bd=d;b=i}});sum[b][0]+=p[0];sum[b][1]+=p[1];sum[b][2]++});S=S.map((s,i)=>sum[i][2]?[sum[i][0]/sum[i][2],sum[i][1]/sum[i][2]]:s)}return S}
const star=(p,x,y,s)=>el('polygon',{points:`${x},${y-7*s} ${x+2*s},${y-2*s} ${x+7*s},${y-2*s} ${x+3*s},${y+1*s} ${x+4.5*s},${y+6*s} ${x},${y+3*s} ${x-4.5*s},${y+6*s} ${x-3*s},${y+1*s} ${x-7*s},${y-2*s} ${x-2*s},${y-2*s}`,fill:C.crit,stroke:'#fff','stroke-width':s,'pointer-events':'none'},p);

/* ---------- data indexes ---------- */
const CD={},SD={},WD={};
(P.counties||[]).forEach(r=>{const k=canon(r.name);if(k)CD[k]=r});
(P.subs||[]).forEach(r=>{const k=canon(r.county);if(!k)return;(SD[k]=SD[k]||[]).push(r)});
(P.wards||[]).forEach(r=>{const k=canon(r.county);if(!k)return;((WD[k]=WD[k]||{})[r.sub]=WD[k][r.sub]||[]).push(r)});
Object.keys(SD).forEach(k=>SD[k].sort((a,b)=>b.v-a.v));
const subRow=(k,name)=>(SD[k]||[]).find(s=>s.n===name);
const wardRow=(k,sub,name)=>((WD[k]||{})[sub]||[]).find(w=>w.n===name);
const NSEED={emgwen:[.5,.45],chesumei:[.68,.18],mosop:[.22,.2],aldai:[.2,.62],nandihills:[.78,.62],tinderetTindiret:[.5,.85],tinderet:[.5,.85],tindiret:[.5,.85]};
const NSEEDK=n=>{const k=nk(n);if(k.indexOf('tinderet')>=0||k.indexOf('tindiret')>=0)return NSEED.tinderet;return NSEED[k]};

/* ---------- lon/lat helpers (the county outlines were fitted to lat/lon, so the fit can be inverted) ---------- */
const inv=(x,y)=>[5.03-(y-KBB.y)/KBB.h*(5.03+4.68),33.91+(x-KBB.x)/KBB.w*(41.91-33.91)]; /* -> [lat,lon] */
const KLL=window.KLL;
const kll=k=>KLL[k].map(r=>r.map(p=>[p[1],p[0]]));
const KB={};KN.forEach(k=>{KB[k]=L.latLngBounds(kll(k).reduce((a,r)=>a.concat(r),[]))});
const KENYA_B=L.latLngBounds([[-4.75,33.85],[5.1,41.95]]);
const REGION=['Siaya','Kisumu','Kakamega','Vihiga','Nandi','Uasin Gishu','Kericho','Nakuru','Nairobi','Bungoma','Trans Nzoia','Kisii'];
const regionB=()=>{let b=null;REGION.forEach(k=>{b=b?b.extend(KB[k]):L.latLngBounds(KB[k].getSouthWest(),KB[k].getNorthEast())});return b};
const HLL=[(P.hospital||{lat:0.203}).lat,(P.hospital||{lon:35.105}).lon];
const CENLL=k=>inv(CEN[k][0],CEN[k][1]);

/* real boundaries (BOUNDARY_GEOJSON) arrive as lon/lat rings */
function boundRings(level,county,sub,name){const B=P.bounds&&P.bounds[level];if(!B)return null;const set=B[nk(county)];if(!set)return null;
  const names=String(name).split(' / ');let all=[];const sk=sub?nk(sub):'';names.forEach(nm=>{const rec=level==='sub'?set[nk(nm)]:(set[sk]||{})[nk(nm)];if(rec&&rec.rings)all=all.concat(rec.rings)});return all.length?all:null}
const toLL=rings=>rings.map(r=>r.map(q=>[q[1],q[0]]));
const ringCenLL=rings=>{let best=rings[0],ba=0;rings.forEach(r=>{const a=area(r);if(a>ba){ba=a;best=r}});const c=pcen(best);return[c[1],c[0]]};

const CS={};
const cty=n=>{if(CS[n])return CS[n];const rs=KLL[n].slice().sort((a,b)=>b.length-a.length);return CS[n]=rs[0].map(q=>pos(q[1],q[0]))};
const cut=(n,cell)=>{const o=clipConvex(cty(n),cell);return o.length>2?[o.map(p=>inv(p[0],p[1]))]:[cell.map(p=>inv(p[0],p[1]))]};
const LAY={};
function layout(n){if(LAY[n]!==undefined)return LAY[n];const subs=SD[n];if(!subs||!subs.length)return LAY[n]=null;
 const names=subs.map(s=>s.n),b=BB[n];const realRings=names.map(nm=>boundRings('sub',n,null,nm));const realOk=realRings.filter(Boolean).length>=Math.max(2,names.length*.7);
 let cells;
 if(realOk){cells=names.map((nm,i)=>{const r=realRings[i];return r?{name:nm,ll:toLL(r),cen:ringCenLL(r),real:true}:{name:nm,ll:[],cen:CENLL(n),real:true,missing:true}})}
 else{let init=null;if(n==='Nandi'){init=names.map(nm=>{const p=NSEEDK(nm)||[.5,.5];let x=b.x+p[0]*b.w,y=b.y+p[1]*b.h;if(!inside(n,x,y)){const s=samples(n,1,'n'+nm,(a,c)=>Math.hypot(a-x,c-y)<b.w*.3);if(s.length){x=s[0][0];y=s[0][1]}}return[x,y]})}
  const seeds=place(n,names.length,'s'+n,null,init),polys=voronoi(seeds,boxPoly(b));
  cells=names.map((nm,i)=>{const sp=samples(n,300,'q'+nm,(x,y)=>pip(polys[i],x,y));const c=sp.length?[sp.reduce((a,p)=>a+p[0],0)/sp.length,sp.reduce((a,p)=>a+p[1],0)/sp.length]:pcen(polys[i]);
    return{name:nm,poly:polys[i],ll:cut(n,polys[i]),cen:inv(c[0],c[1])}})}
 cells.forEach(c=>{const ws=(WD[n]||{})[c.name];if(!ws||!ws.length)return;const wn=ws.map(w=>w.n);const wr=wn.map(w=>boundRings('ward',n,c.name,w));
   if(wr.filter(Boolean).length>=Math.max(2,wn.length*.7)){c.wards=wn.map((w,i)=>wr[i]?{name:w,ll:toLL(wr[i]),cen:ringCenLL(wr[i]),real:true}:{name:w,ll:[],cen:c.cen,missing:true})}
   else if(c.poly){const sd=place(n,wn.length,'w'+c.name,(x,y)=>pip(c.poly,x,y));if(n==='Nandi'&&nk(c.name)==='emgwen'){const kap=wn.findIndex(w=>nk(w)==='kapsabet');if(kap>=0){let j=0,bd=1e18;sd.forEach((q,k)=>{const d=Math.hypot(q[0]-HOSP[0],q[1]-HOSP[1]);if(d<bd){bd=d;j=k}});const t=sd[kap];sd[kap]=sd[j];sd[j]=t}}
     const wp=voronoi(sd,c.poly).map(p=>clipConvex(p,c.poly));c.wards=wn.map((w,i)=>{const sp=samples(n,120,'z'+w,(x,y)=>pip(wp[i],x,y));const cc=sp.length?[sp.reduce((a,p)=>a+p[0],0)/sp.length,sp.reduce((a,p)=>a+p[1],0)/sp.length]:pcen(wp[i]);return{name:w,ll:cut(n,wp[i]),cen:inv(cc[0],cc[1])}})}});
 return LAY[n]={cells,real:realOk}}
const hasWards=n=>{const l=layout(n);return !!(l&&l.cells.some(c=>c.wards))};

/* ---------- the map: Leaflet with a normal street basemap, free zoom and pan ---------- */
const TILE=P.tile||{url:'https://tile.openstreetmap.org/{z}/{x}/{y}.png',attr:'&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors'};
function newMap(parent,h,small){
  const div=H('div','lmap','',parent);div.style.height=h+'px';
  const map=L.map(div,{zoomSnap:.5,zoomDelta:1,wheelPxPerZoomLevel:70,minZoom:3,maxZoom:15,scrollWheelZoom:false,attributionControl:true});
  L.tileLayer(TILE.url,{attribution:TILE.attr,subdomains:TILE.subdomains||'abc',maxZoom:19,referrerPolicy:'strict-origin-when-cross-origin'}).addTo(map);
  const M={map,div,lg:L.layerGroup().addTo(map),lab:[],vkey:null,bounds:KENYA_B};
  const Reset=L.Control.extend({options:{position:'topleft'},onAdd:()=>{const b=L.DomUtil.create('div','leaflet-bar');const a=L.DomUtil.create('a','',b);a.href='#';a.title='Reset view';a.innerHTML='⤢';a.style.cssText='font-size:16px;line-height:26px;text-align:center;cursor:pointer';
    L.DomEvent.on(a,'click',e=>{L.DomEvent.stop(e);fit(M)});return b}});
  if(!small)new Reset().addTo(map);
  const hint=H('div','maphint','Click the map, then scroll to zoom. Drag to pan.',div);if(small){hint.style.display='none';map.attributionControl.setPrefix(false)}
  map.on('click',()=>{map.scrollWheelZoom.enable();hint.style.display='none'});
  div.addEventListener('mouseleave',()=>{map.scrollWheelZoom.disable();if(!small)hint.style.display=''});
  map.on('zoomend moveend',()=>relabel(M));
  map.fitBounds(KENYA_B,{animate:false});return M}
const fit=M=>{M.map.invalidateSize();M.map.fitBounds(M.bounds,{padding:[14,14],animate:false});relabel(M)};
function relabel(M){const m=M.map,sz=m.getSize();M.lab.forEach(l=>{const e=l.m.getElement();if(!e)return;const sw=m.latLngToContainerPoint(l.b.getSouthWest()),ne=m.latLngToContainerPoint(l.b.getNorthEast());
  const w=Math.abs(ne.x-sw.x),c=m.latLngToContainerPoint(l.m.getLatLng()),on=(!l.need||w>=l.need)&&c.x>0&&c.y>0&&c.x<sz.x&&c.y<sz.y;e.style.display=on?'':'none'})}
const starIcon=L.divIcon({className:'',html:'<svg width="22" height="22" viewBox="0 0 22 22"><polygon points="11,1 14,8 21,8.4 15.6,13 17.4,20 11,16 4.6,20 6.4,13 1,8.4 8,8" fill="#A32D2D" stroke="#fff" stroke-width="1.5"/></svg>',iconSize:[22,22],iconAnchor:[11,11]});
const TIPO={sticky:true,className:'lt',direction:'top',opacity:1};
const bnd=ll=>L.latLngBounds(ll.reduce((a,r)=>a.concat(r),[]));
function addPoly(M,ll,style,tip,click){const p=L.polygon(ll,style).addTo(M.lg);if(tip)p.bindTooltip(tip,TIPO);
  p.on('mouseover',()=>p.setStyle({weight:(style.weight||1)+1.4}));p.on('mouseout',()=>p.setStyle({weight:style.weight||1}));if(click)p.on('click',e=>{L.DomEvent.stopPropagation(e);click()});return p}
function addLabel(M,ll,html,cls,need,b){const m=L.marker(ll,{icon:L.divIcon({className:'',html:`<div class="lb ${cls||''}">${html}</div>`,iconSize:[0,0]}),interactive:false,keyboard:false}).addTo(M.lg);if(b)M.lab.push({m,b,need});return m}
const fo=f=>f.color===C.grey?.3:Math.max(.2,Math.min(.9,f.op*.88));
const nameLines=nm=>nm.indexOf(' / ')>0?nm.split(' / '):[nm];

function geoMap(M,st,cfg){
  M.lg.clearLayers();M.lab=[];const n=st.sel,vkey=n?'c:'+n:(st.region?'region':'kenya');
  let bounds=n?KB[n]:(st.region?regionB():KENYA_B);
  if(!n){KN.forEach(k=>{const f=cfg.county(k);addPoly(M,kll(k),{color:f.stroke||'#fff',weight:f.sw||1,fillColor:f.color,fillOpacity:fo(f)},f.tip||k,cfg.click?()=>cfg.click(k):null)});
    if(!cfg.noLabels)KN.forEach(k=>{const f=cfg.county(k);if(f.l1)addLabel(M,CENLL(k),f.l1,f.dark?'dk':'',0)});
    L.polygon(kll('Nandi'),{color:C.crit,weight:2,fill:false,interactive:false}).addTo(M.lg);
    if(cfg.overlay)cfg.overlay(M);L.marker(HLL,{icon:starIcon,zIndexOffset:900}).bindTooltip('Kapsabet CRH',TIPO).addTo(M.lg)}
  else{
    KN.forEach(k=>{if(k===n)return;const f=cfg.county?cfg.county(k):null;/* neighbours keep their own colours so the surrounding picture is not lost */
      addPoly(M,kll(k),f?{color:'#fff',weight:1,fillColor:f.color,fillOpacity:fo(f)*.75}:{color:'#8B95A5',weight:1,fillColor:'#fff',fillOpacity:.04},f?(f.tip||k):k,cfg.click?()=>cfg.click(k):null)});
    const l=layout(n);
    if(!l){const f=cfg.county(n);addPoly(M,kll(n),{color:'#fff',weight:2,fillColor:f.color,fillOpacity:fo(f)},f.tip||n);addLabel(M,CENLL(n),n+(f.l2?`<small>${f.l2}</small>`:''),f.dark?'dk':'',0)}
    else{
      const wardMode=st.lvl==='ward'&&hasWards(n);
      if(l.real){const all=l.cells.filter(c=>!c.missing).map(c=>bnd(c.ll));if(all.length){bounds=all[0];all.forEach(b=>bounds=bounds.extend(b))}}
      l.cells.forEach((c,i)=>{if(c.missing)return;const f=cfg.sub(n,c.name,i);
        addPoly(M,c.ll,{color:'#fff',weight:wardMode?2.4:1.6,fillColor:f.color,fillOpacity:wardMode?.04:fo(f)},wardMode?null:(f.tip||c.name))});
      if(wardMode){l.cells.forEach(c=>(c.wards||[]).forEach(wd=>{if(wd.missing)return;const f=cfg.ward(n,c.name,wd.name);addPoly(M,wd.ll,{color:'#fff',weight:.9,fillColor:f.color,fillOpacity:fo(f)},f.tip||wd.name)}));
        l.cells.forEach(c=>{if(c.missing)return;L.polygon(c.ll,{color:'#fff',weight:2.4,fill:false,interactive:false}).addTo(M.lg)})}
      if(!cfg.noLabels){
        if(wardMode)l.cells.forEach(c=>(c.wards||[]).forEach(wd=>{if(wd.missing)return;const f=cfg.ward(n,c.name,wd.name);addLabel(M,wd.cen,wd.name+(f.l2?`<small>${f.l2}</small>`:''),f.dark?'dk':'',wd.name.length*5.6,bnd(wd.ll))}));
        else l.cells.forEach((c,i)=>{if(c.missing)return;const f=cfg.sub(n,c.name,i);addLabel(M,c.cen,nameLines(c.name).join('<br>')+(f.l2?`<small>${f.l2}</small>`:''),f.dark?'dk':'',Math.min(60,c.name.length*5),bnd(c.ll))})}
      if(!l.real)L.polygon(kll(n),{color:'#374151',weight:2,fill:false,interactive:false}).addTo(M.lg)}
    if(n==='Nandi')L.marker(HLL,{icon:starIcon,zIndexOffset:900}).bindTooltip('Kapsabet CRH',TIPO).addTo(M.lg);
    if(cfg.overlay)cfg.overlay(M)}
  if(M.vkey!==vkey){M.vkey=vkey;M.bounds=bounds;fit(M)}else{M.map.invalidateSize();relabel(M)}}

/* ---------- shared bits ---------- */
function crumb(parent,st,redraw,suffix,extra){const d=H('div','crumb','',parent);d._set=()=>{const ex=extra?extra():'';if(!st.sel){d.innerHTML='Kenya <span> · click a county to zoom in · </span>';const a=H('a','back',st.region?'Show whole Kenya':'Zoom to western Kenya, where most patients live',d);a.onclick=()=>{st.region=!st.region;redraw()};if(ex)d.insertAdjacentHTML('beforeend',ex)}
  else{d.innerHTML=`<a class="back">‹ Kenya</a> › <b>${st.sel}</b>${suffix?suffix():''}${ex}`;d.querySelector('.back').onclick=()=>{st.sel=null;redraw()}}};return d}
function legend(parent){return H('div','legend','',parent)}
const swatch=(c,t,o)=>`<span><i style="background:${c};${o?'opacity:'+o:''}"></i>${t}</span>`;
const rgba=(hex,a)=>{const h=hex.replace('#','');return`rgba(${parseInt(h.substr(0,2),16)},${parseInt(h.substr(2,2),16)},${parseInt(h.substr(4,2),16)},${a})`};
/* one colour per disease group, used everywhere a group is shown */
const CATCOL={'All visits':'#888780','Infectious disease':'#D97706','Respiratory':'#1D6FA8','NCD':'#6B3FA0','Injury & trauma':'#A32D2D','Maternal & obstetric':'#D6336C','Mental health':'#0F6E56','Malaria':'#8A7A12'};
const fmtp=v=>(Math.round(v*10)/10)+'%';
function pieSvg(r,parts){const tot=parts.reduce((a,p)=>a+p[0],0)||1;let a0=-Math.PI/2,out='';
  parts.forEach(([v,c])=>{if(v<=0)return;const a1=a0+v/tot*2*Math.PI;
    out+=v/tot>.999?`<circle cx="${r}" cy="${r}" r="${r-.5}" fill="${c}" stroke="#fff" stroke-width="1"/>`:`<path d="M${r},${r}L${r+(r-.5)*Math.cos(a0)},${r+(r-.5)*Math.sin(a0)}A${r-.5},${r-.5} 0 ${a1-a0>Math.PI?1:0} 1 ${r+(r-.5)*Math.cos(a1)},${r+(r-.5)*Math.sin(a1)}Z" fill="${c}" stroke="#fff" stroke-width="1"/>`;a0=a1});
  return`<svg width="${2*r}" height="${2*r}" viewBox="0 0 ${2*r} ${2*r}">${out}</svg>`}
const showSoon=M=>setTimeout(()=>{M.map.invalidateSize();relabel(M)},60);

/* ============================== A · catchment ============================== */
function mountCatchment(root){
  const A={sel:null,lvl:'sub',met:'v',region:false},card=H('div','card','',root);
  const head=H('div','card-head','',card);H('div','card-title','Catchment map: Kenya → county → sub-county → ward',head);
  const ctl=H('div','ctl','',card);seg(ctl,[['v','Visits'],['p','Unique patients'],['r','Main reason']],A.met,v=>{A.met=v;if(v==='r'&&!A.sel)A.region=true;draw()});
  const lv=seg(ctl,[['sub','Sub-county'],['ward','Ward']],A.lvl,v=>{A.lvl=v;draw()});
  const goTo=sel(ctl,'Go to',[['','Whole of Kenya']].concat(Object.keys(SD).map(canon).filter(Boolean).sort((a,b)=>(a==='Nandi'?-1:b==='Nandi'?1:CD[b]?CD[b].v-(CD[a]?CD[a].v:0):0)).map(n=>[n,n+' (sub-counties'+(hasWards(n)?' and wards':'')+')'])),'',v=>{A.sel=v||null;draw()});
  const cr=crumb(card,A,draw,()=>` › ${A.lvl==='ward'&&hasWards(A.sel)?'Wards':'Sub-counties'}`);
  const M=newMap(card,540),leg=legend(card),note=H('div','card-note','',card);
  const key=()=>A.met==='p'?'p':'v';
  function draw(){
    const m=A.met,k=key();const others=Object.keys(CD).filter(x=>x!=='Nandi'),omax=Math.max(1,...others.map(x=>CD[x][k]));
    const cfg={
      county:n=>{const d=CD[n];const tp=d?`<b>${n}</b><br>${fmt(d.v)} visits · ${fmt(d.p)} patients`:`<b>${n}</b><br>no visits`;
        if(n==='Nandi')return{color:C.primary,op:1,tip:tp};if(m==='r')return{color:'#C9CED8',op:.35,tip:tp};if(!d||!d[k])return{color:C.grey,op:1,tip:tp};return{...sc(d[k]/omax,C.blue),tip:tp}},
      sub:(n,name,i)=>{const rows=SD[n]||[],mx=Math.max(1,...rows.map(r=>r[k])),r=subRow(n,name);if(!r)return{color:C.grey,op:1,tip:name};const col=n==='Nandi'?C.primary:C.blue,f=sc(r[k]/mx,col);
        return{...f,dark:f.op>.5,l2:fmt(r[k]),tip:`<b>${name}</b> · ${n}<br>${fmt(r.v)} visits · ${fmt(r.p)} patients`}},
      ward:(n,sub,name)=>{const all=Object.values(WD[n]||{}).flat(),mx=Math.max(1,...all.map(r=>r[k])),r=wardRow(n,sub,name);if(!r)return{color:C.grey,op:1,tip:name};const op=+(0.15+0.85*Math.sqrt(r[k]/mx)).toFixed(2);
        return{color:n==='Nandi'?C.primary:C.blue,op,dark:op>.55,l2:r[k]>=mx*.04?fmt(r[k]):'',tip:`<b>${name}</b> · ${sub}<br>${fmt(r.v)} visits · ${fmt(r.p)} patients`}},
      click:n=>{A.sel=n;draw()},
      overlay:M=>{if(A.sel||m!=='r')return;const rows=others.map(x=>CD[x]).filter(d=>d&&d.v>=5&&canon(d.name)),mx=Math.max(1,...rows.map(d=>d.v));
        rows.forEach(d=>L.polyline([CENLL(canon(d.name)),HLL],{color:C.blue,opacity:.3,weight:.6+Math.sqrt(d.v/mx)*3.2,interactive:false}).addTo(M.lg));
        rows.slice().sort((a,b)=>b.v-a.v).reverse().forEach(d=>{const p=CENLL(canon(d.name)),r=Math.round(5+Math.sqrt(d.v/mx)*17),oth=Math.max(0,d.v-d.inj-d.acu-d.ncd);
          L.marker(p,{icon:L.divIcon({className:'',html:pieSvg(r,[[d.inj,C.crit],[d.acu,C.amber],[d.ncd,C.blue],[oth,'#B8BDC7']]),iconSize:[2*r,2*r],iconAnchor:[r,r]})}).bindTooltip(`<b>${d.name}</b> · ${fmt(d.v)} visits<br>Injury ${d.inj} · Acute ${d.acu} · NCD ${d.ncd} · Other ${oth}`,TIPO).addTo(M.lg)});
        rows.slice().sort((a,b)=>b.v-a.v).slice(0,8).forEach(d=>{const r=Math.round(5+Math.sqrt(d.v/mx)*17);addLabel(M,CENLL(canon(d.name)),`<span style="position:relative;top:${r+9}px">${d.name} ${d.v}</span>`,'',0)})}};
    geoMap(M,A,cfg);cr._set();goTo.value=A.sel||'';lv.style.display=(!A.sel||hasWards(A.sel))?'':'none';
    if(m==='r'&&!A.sel)leg.innerHTML=[[C.crit,'Injury'],[C.amber,'Acute illness'],[C.blue,'NCD'],['#B8BDC7','Other / not classified']].map(r=>swatch(r[0],r[1])).join('')+'<span>Pie size = visits · lines = flow to ★ Kapsabet CRH</span>';
    else leg.innerHTML=swatch(C.primary,`Nandi (${fmt(CD.Nandi?CD.Nandi.v:0)} visits)`)+swatch(C.blue,`Other counties, darker = more ${m==='p'?'patients':'visits'}`)+swatch(C.grey,'No visits')+'<span>★ Kapsabet CRH</span>';
    const rl=A.sel&&layout(A.sel)&&layout(A.sel).real;
    note.textContent=!A.sel?(m==='r'?'Reason groups come from diagnosis_1 (Q2.6). "Other" is everything not classified as injury, acute illness or NCD. Zoom and pan freely; click a county to jump to it.':'Zoom and pan freely. Click a county to jump to its sub-counties. Nandi is outside the colour scale so the other counties stay readable.')
      :(layout(A.sel)?(A.lvl==='ward'&&hasWards(A.sel)?'Shading = visits per ward. Hover for the sub-county and patient count. Labels appear as you zoom in.':'Shading = visits per sub-county.')+(rl?'':' Shapes are a schematic division of the county until boundary polygons are loaded.'):`${A.sel} has no sub-county detail in the data yet.`)}
  draw()}

/* ============================== C · distance rings ============================== */
function mountRings(root){
  const R=P.rings,cats=R.cats,bands=R.bands,edges=R.edges||[2,5,10,20,50,100],real=cats.filter(c=>c!=='All visits');let cur=cats.indexOf('Injury & trauma')>=0?'Injury & trauma':cats[1]||cats[0];
  const bandOf=km=>{const i=edges.findIndex(e=>km<e);return i<0?edges.length:i};
  const dom=l=>{let b=null,bv=0;real.forEach(c=>{const v=l.v[c]||0;if(v>bv){bv=v;b=c}});return b};
  const card=H('div','card','',root);const head=H('div','card-head','',card);H('div','card-title','Do some diseases travel further? Distance from Kapsabet CRH, by disease group',head);
  const pl=pills(card,cats,cur,c=>{cur=c;draw()});
  const g2=H('div','g2','',card),left=H('div','','',g2),right=H('div','','',g2);
  const M=newMap(left,560),note=H('div','card-note','',left);
  const rt=H('div','card-title','',right);const rl=legend(right),bsvg=el('svg',{style:'width:100%'});right.appendChild(bsvg);
  H('div','card-title','All disease groups at once: % of each group\'s visits, and median distance',card).style.margin='16px 0 6px';const tbl=H('table','',null,card);
  function draw(){
    const d=R.byCat[cur]||{pct:bands.map(()=>0),median:null},a=(R.byCat['All visits']||d).pct,all=cur==='All visits';
    const locs=R.locs.filter(l=>(l.v[cur]||0)>=1),vmax=Math.max(1,...locs.map(l=>l.v[cur]||0)),colFor=l=>all?(CATCOL[dom(l)]||C.neutral):CATCOL[cur];
    geoMap(M,{sel:'Nandi',lvl:'sub'},{noLabels:true,sub:()=>({color:'#CFE3D9',op:.3}),overlay:M=>{
      [2,5,10,20].forEach(k=>{L.circle(HLL,{radius:k*1000,fill:false,color:C.primary,weight:1.3,dashArray:'5 4',interactive:false}).addTo(M.lg);addLabel(M,[HLL[0]+k/111.19,HLL[1]],`<span style="color:${C.primary}">${k} km</span>`,'',0)});
      locs.slice().sort((x,y)=>(y.v[cur]||0)-(x.v[cur]||0)).forEach(l=>{const v=l.v[cur]||0,dm=dom(l);
        L.circleMarker([l.lat,l.lon],{radius:3+Math.sqrt(v/vmax)*14,color:'#fff',weight:1.2,fillColor:colFor(l),fillOpacity:.78}).bindTooltip(`<b>${l.n}</b><br>${fmt(v)} ${cur.toLowerCase()} visits<br>${l.km.toFixed(1)} km from the hospital${all&&dm?`<br>Largest group: ${dm}`:''}`,TIPO).addTo(M.lg)})}});
    note.textContent=all?`Each bubble is a locality, sized by visits and coloured by its largest disease group. Median ${d.median==null?'n/a':d.median+' km'}. Rings are true distances from the hospital. Zoom and pan freely.`:`Each bubble is a locality; size = ${cur.toLowerCase()} visits. More weight in the outer rings means that group travels further. Median ${d.median==null?'n/a':d.median+' km'}. Rings are true distances from the hospital.`;
    bsvg.innerHTML='';bsvg.setAttribute('viewBox',`0 0 340 ${bands.length*32+6}`);
    if(all){rt.textContent='Share of visits by distance band, coloured by disease mix';rl.innerHTML=real.map(c=>swatch(CATCOL[c],c)).join('');
      const bandCat=bands.map(()=>({}));R.locs.forEach(l=>{const i=bandOf(l.km);real.forEach(c=>{bandCat[i][c]=(bandCat[i][c]||0)+(l.v[c]||0)})});
      bands.forEach((b,i)=>{const y=4+i*32,W=Math.min(220,a[i]/100*220),tot=real.reduce((s,c)=>s+(bandCat[i][c]||0),0);el('rect',{x:78,y,width:220,height:20,fill:C.s2,rx:3},bsvg);let x0=78;
        if(tot>0)real.forEach(c=>{const w=W*(bandCat[i][c]||0)/tot;if(w<=0)return;const r=el('rect',{x:x0,y,width:w,height:20,fill:CATCOL[c]},bsvg);const t=el('title',{},r);t.textContent=`${b} · ${c}: ${Math.round(100*(bandCat[i][c]||0)/tot)}% of category visits in this band`;x0+=w});
        tx(bsvg,b,{x:72,y:y+14,'text-anchor':'end','font-size':10,fill:C.text});tx(bsvg,a[i]+'%',{x:78+W+5,y:y+14,'font-size':10,'font-weight':600,fill:C.text})})}
    else{rt.textContent='Share of visits by distance band';rl.innerHTML=swatch(CATCOL[cur],cur)+swatch(C.neutral,'All visits');
      bands.forEach((b,i)=>{const y=4+i*32;el('rect',{x:78,y,width:220,height:14,fill:C.s2,rx:3},bsvg);el('rect',{x:78,y,width:Math.min(220,d.pct[i]/100*220),height:14,fill:CATCOL[cur],rx:3,'fill-opacity':.9},bsvg);el('rect',{x:78,y:y+16,width:Math.min(220,a[i]/100*220),height:8,fill:C.neutral,rx:2,'fill-opacity':.7},bsvg);
        tx(bsvg,b,{x:72,y:y+12,'text-anchor':'end','font-size':10,fill:C.text});tx(bsvg,d.pct[i]+'%',{x:78+d.pct[i]/100*220+5,y:y+12,'font-size':10,'font-weight':600,fill:CATCOL[cur]});tx(bsvg,a[i]+'%',{x:78+a[i]/100*220+5,y:y+23,'font-size':8.5,fill:C.muted})})}
    pl.querySelectorAll('button').forEach(b=>{const on=b.dataset.v===cur;b.classList.toggle('on',on);b.style.background=on?CATCOL[b.dataset.v]:'';b.style.borderColor=CATCOL[b.dataset.v];b.style.color=on?'#fff':''})}
  tbl.innerHTML='<thead><tr><th>Disease group</th>'+bands.map(b=>`<th style="text-align:center">${b}</th>`).join('')+'<th style="text-align:right">Median km</th></tr></thead>';
  const tb=H('tbody','','',tbl);const mx=Math.max(1,...Object.values(R.byCat).flatMap(x=>x.pct));
  Object.keys(R.byCat).forEach(k=>{const o=R.byCat[k],tr=H('tr','','',tb),cc=CATCOL[k]||C.neutral;tr.innerHTML=`<td style="font-weight:600"><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${cc};margin-right:7px"></i>${k}</td>`+o.pct.map(v=>{const t=Math.min(1,v/mx);return`<td style="text-align:center;font-weight:600;background:${rgba(cc,.06+.7*t)};color:${t>.55?'#fff':C.text}">${v}%</td>`}).join('')+`<td style="text-align:right;font-weight:700;color:${o.median>=2?C.crit:C.text}">${o.median==null?'n/a':o.median}</td>`});
  draw()}

/* ============================== E helpers ============================== */
function dumbbell(parent,rows,o){const svg=el('svg',{style:'width:100%'});parent.appendChild(svg);const L=o.L||110,W=o.W||340,mx=o.max,X=v=>L+v/mx*W,rh=o.rh||26;svg.setAttribute('viewBox',`0 0 ${L+W+o.right} ${rows.length*rh+34}`);
  o.ticks.forEach(v=>{el('line',{x1:X(v),y1:2,x2:X(v),y2:rows.length*rh+8,stroke:C.border},svg);tx(svg,fmt(v),{x:X(v),y:rows.length*rh+24,'text-anchor':'middle','font-size':9,fill:C.muted})});
  rows.forEach((r,i)=>{const y=14+i*rh,ch=r[1]?(r[2]-r[1])/r[1]*100:0,hi=o.hl&&o.hl(r);if(hi)el('rect',{x:2,y:y-rh/2+1,width:L+W+o.right-4,height:rh-2,rx:4,fill:'#E8F5F2'},svg);
    el('line',{x1:X(r[2]),y1:y,x2:X(r[1]),y2:y,stroke:r[2]>r[1]?C.primary:C.amber,'stroke-width':2.5},svg);el('circle',{cx:X(r[1]),cy:y,r:5.5,fill:C.blue},svg);el('circle',{cx:X(r[2]),cy:y,r:5.5,fill:C.amber},svg);
    tx(svg,r[0],{x:L-8,y:y+4,'text-anchor':'end','font-size':10.5,'font-weight':hi?700:400,fill:C.text});tx(svg,(ch>0?'+':'−')+Math.abs(ch).toFixed(0)+'%',{x:X(Math.max(r[1],r[2]))+10,y:y+4,'font-size':10,'font-weight':600,fill:ch>0?C.primary:C.amber});if(o.tag)o.tag(svg,r,y,hi)})}

/* ============================== D · disease burden by place ============================== */
function mountDisease(root){
  const cats=P.cats,D={cat:cats[0],lvl:'sub',sel:null,view:'one',region:false},card=H('div','card','',root);
  const head=H('div','card-head','',card);H('div','card-title','Where each disease group\'s patients live',head);
  const ctl=H('div','ctl','',card);seg(ctl,[['one','One disease group'],['all','All seven side by side']],D.view,v=>{D.view=v;if(v==='all'&&!D.sel)D.region=true;draw()});
  const lv=seg(ctl,[['sub','Sub-county'],['ward','Ward']],D.lvl,v=>{D.lvl=v;draw()});
  const pl=pills(card,cats,D.cat,c=>{D.cat=c;draw()});
  const cr=crumb(card,D,draw,()=>` › ${D.lvl==='ward'&&hasWards(D.sel)?'Wards':'Sub-counties'}`,()=>D.view==='one'?` · <b>${D.cat}</b>`:' · all seven groups');
  const one=H('div','g2','',card),oleft=H('div','','',one),oright=H('div','','',one);let OM=null;
  H('div','card-title','Wards with a higher share than expected',oright);const lqn=H('div','card-note','',oright);const lq=el('svg',{viewBox:'0 0 520 190',style:'width:100%;max-width:560px'});oright.appendChild(lq);
  const minis=H('div','mini-grid','',card),leg=legend(card),note=H('div','card-note','',card);const MM={};
  const CV=P.catv;const getc=(cat,n)=>((CV[cat]||{}).county||{})[nk(n)],gets=(cat,n,s)=>((CV[cat]||{}).sub||{})[nk(n)+'|'+s],getw=(cat,n,s,w)=>((CV[cat]||{}).ward||{})[nk(n)+'|'+s+'|'+w];
  function cfgFor(cat,small){const col=CATCOL[cat]||C.blue;const ok=n=>{const d=getc(cat,n);return d&&d.visits>=30};const cm=Math.max(1,...KN.filter(ok).map(n=>getc(cat,n).pct));
    return{noLabels:small,county:n=>{const d=getc(cat,n);if(!d||d.visits<30)return{color:C.grey,op:1,tip:`<b>${n}</b><br>${d?fmt(d.visits):0} visits: too few to show a share`};const f=sc(d.pct/cm,col);return{...f,dark:f.op>.55,tip:`<b>${n}</b><br>${cat}: ${d.pct}% of visits<br>${fmt(d.visits)} visits`}},
      sub:(n,name,i)=>{const rows=(SD[n]||[]).map(r=>gets(cat,n,r.n)).filter(Boolean),mx=Math.max(1,...rows.map(r=>r.pct)),d=gets(cat,n,name);if(!d||d.visits<30)return{color:C.grey,op:1,tip:`<b>${name}</b><br>too few visits`};const f=sc(d.pct/mx,col);return{...f,dark:f.op>.55,l2:d.pct+'%',tip:`<b>${name}</b> · ${n}<br>${cat}: ${d.pct}% of visits<br>${fmt(d.visits)} visits`}},
      ward:(n,s,w)=>{const d=getw(cat,n,s,w);if(!d||d.visits<30)return{color:C.grey,op:1,tip:`<b>${w}</b> · ${s}<br>too few visits`};const all=Object.keys((CV[cat]||{}).ward||{}).filter(x=>x.indexOf(nk(n)+'|')===0).map(x=>CV[cat].ward[x].pct),mx=Math.max(1,...all);const op=+(0.15+0.8*Math.sqrt(d.pct/mx)).toFixed(2);return{color:col,op,dark:op>.55,l2:d.pct+'%',tip:`<b>${w}</b> · ${s}<br>${cat}: ${d.pct}% of visits<br>${fmt(d.visits)} visits`}},
      click:n=>{D.sel=n;draw()}}}
  function draw(){
    const cat=D.cat;one.style.display=D.view==='one'?'grid':'none';minis.style.display=D.view==='all'?'grid':'none';pl.style.display=D.view==='one'?'flex':'none';lv.style.display=(!D.sel||hasWards(D.sel))?'':'none';
    cr._set();
    pl.querySelectorAll('button').forEach(b=>{const on=b.dataset.v===cat;b.classList.toggle('on',on);b.style.background=on?CATCOL[b.dataset.v]:'';b.style.borderColor=CATCOL[b.dataset.v];b.style.color=on?'#fff':''});
    if(D.view==='one'){if(!OM)OM=newMap(oleft,520);geoMap(OM,D,cfgFor(cat,false));showSoon(OM);lqn.textContent=`Nandi wards · ${cat} · 1.0× = catchment average`;lq.innerHTML='';const rows=((P.lq||{})[cat]||[]).slice(0,5);
      if(!rows.length){tx(lq,'Not enough ward volume yet',{x:10,y:30,'font-size':11,fill:C.muted})}
      else{const mx=Math.max(2.5,...rows.map(r=>r[1]));el('line',{x1:150+240/mx,y1:2,x2:150+240/mx,y2:170,stroke:C.muted,'stroke-dasharray':'3 3'},lq);tx(lq,'1.0× = catchment average',{x:154+240/mx,y:184,'font-size':9,fill:C.muted});
        rows.forEach((r,i)=>{const y=4+i*33,w=r[1]/mx*240;el('rect',{x:150,y,width:w,height:22,fill:r[1]>=1?CATCOL[cat]:C.neutral,'fill-opacity':.85,rx:2},lq);tx(lq,r[0],{x:144,y:y+15,'text-anchor':'end','font-size':10.5,fill:C.text});tx(lq,r[1].toFixed(1)+'×',{x:154+w,y:y+15,'font-size':10.5,'font-weight':600,fill:C.text})})}
      note.textContent=!D.sel?'Share of each county\'s visits that are '+cat.toLowerCase()+'. Colours scale within this disease group. Zoom and pan freely, or click a county to jump to its sub-counties.':'Darker = higher share of this disease group. Grey = under 30 visits.'}
    else{if(!minis.children.length)cats.forEach(c=>{const m=H('div','mini','',minis);const t=H('div','t',c,m);t.style.color=CATCOL[c];MM[c]=newMap(m,230,true)});
      cats.forEach(c=>{geoMap(MM[c],D,cfgFor(c,true));showSoon(MM[c])});note.textContent='Each map has its own colour scale: compare the pattern within a map, not shades across maps. Click a county on any map to zoom all seven.'}
    leg.innerHTML=swatch(C.grey,'Under 30 visits, too few for a share')+swatch(CATCOL[cat],'lower share',.3)+swatch(CATCOL[cat],'higher share')}
  draw()}

/* ============================== NCD · ward maps (Tab 4) ============================== */
/* mode 'burden': NCD patients per ward, one pill per condition. mode 'recur': share of a ward's NCD patients seen in both 2024 and 2025.
   P.wards = [{w,s,lat,lon,rings,...}] with rings as lon/lat; point-only wards (no boundary) are drawn as circles. */
/* ============================== E helpers ============================== */
function dumbbell(parent,rows,o){const svg=el('svg',{style:'width:100%'});parent.appendChild(svg);const L=o.L||110,W=o.W||340,mx=o.max,X=v=>L+v/mx*W,rh=o.rh||26;svg.setAttribute('viewBox',`0 0 ${L+W+o.right} ${rows.length*rh+34}`);
  o.ticks.forEach(v=>{el('line',{x1:X(v),y1:2,x2:X(v),y2:rows.length*rh+8,stroke:C.border},svg);tx(svg,fmt(v),{x:X(v),y:rows.length*rh+24,'text-anchor':'middle','font-size':9,fill:C.muted})});
  rows.forEach((r,i)=>{const y=14+i*rh,ch=r[1]?(r[2]-r[1])/r[1]*100:0,hi=o.hl&&o.hl(r);if(hi)el('rect',{x:2,y:y-rh/2+1,width:L+W+o.right-4,height:rh-2,rx:4,fill:'#E8F5F2'},svg);
    el('line',{x1:X(r[2]),y1:y,x2:X(r[1]),y2:y,stroke:r[2]>r[1]?C.primary:C.amber,'stroke-width':2.5},svg);el('circle',{cx:X(r[1]),cy:y,r:5.5,fill:C.blue},svg);el('circle',{cx:X(r[2]),cy:y,r:5.5,fill:C.amber},svg);
    tx(svg,r[0],{x:L-8,y:y+4,'text-anchor':'end','font-size':10.5,'font-weight':hi?700:400,fill:C.text});tx(svg,(ch>0?'+':'−')+Math.abs(ch).toFixed(0)+'%',{x:X(Math.max(r[1],r[2]))+10,y:y+4,'font-size':10,'font-weight':600,fill:ch>0?C.primary:C.amber});if(o.tag)o.tag(svg,r,y,hi)})}

/* ============================== D · disease burden by place ============================== */
function mountDisease(root){
  const cats=P.cats,D={cat:cats[0],lvl:'sub',sel:null,view:'one',region:false},card=H('div','card','',root);
  const head=H('div','card-head','',card);H('div','card-title','Where each disease group\'s patients live',head);
  const ctl=H('div','ctl','',card);seg(ctl,[['one','One disease group'],['all','All seven side by side']],D.view,v=>{D.view=v;if(v==='all'&&!D.sel)D.region=true;draw()});
  const lv=seg(ctl,[['sub','Sub-county'],['ward','Ward']],D.lvl,v=>{D.lvl=v;draw()});
  const pl=pills(card,cats,D.cat,c=>{D.cat=c;draw()});
  const cr=crumb(card,D,draw,()=>` › ${D.lvl==='ward'&&hasWards(D.sel)?'Wards':'Sub-counties'}`,()=>D.view==='one'?` · <b>${D.cat}</b>`:' · all seven groups');
  const one=H('div','g2','',card),oleft=H('div','','',one),oright=H('div','','',one);let OM=null;
  H('div','card-title','Wards with a higher share than expected',oright);const lqn=H('div','card-note','',oright);const lq=el('svg',{viewBox:'0 0 520 190',style:'width:100%;max-width:560px'});oright.appendChild(lq);
  const minis=H('div','mini-grid','',card),leg=legend(card),note=H('div','card-note','',card);const MM={};
  const CV=P.catv;const getc=(cat,n)=>((CV[cat]||{}).county||{})[nk(n)],gets=(cat,n,s)=>((CV[cat]||{}).sub||{})[nk(n)+'|'+s],getw=(cat,n,s,w)=>((CV[cat]||{}).ward||{})[nk(n)+'|'+s+'|'+w];
  function cfgFor(cat,small){const col=CATCOL[cat]||C.blue;const ok=n=>{const d=getc(cat,n);return d&&d.visits>=30};const cm=Math.max(1,...KN.filter(ok).map(n=>getc(cat,n).pct));
    return{noLabels:small,county:n=>{const d=getc(cat,n);if(!d||d.visits<30)return{color:C.grey,op:1,tip:`<b>${n}</b><br>${d?fmt(d.visits):0} visits: too few to show a share`};const f=sc(d.pct/cm,col);return{...f,dark:f.op>.55,tip:`<b>${n}</b><br>${cat}: ${d.pct}% of visits<br>${fmt(d.visits)} visits`}},
      sub:(n,name,i)=>{const rows=(SD[n]||[]).map(r=>gets(cat,n,r.n)).filter(Boolean),mx=Math.max(1,...rows.map(r=>r.pct)),d=gets(cat,n,name);if(!d||d.visits<30)return{color:C.grey,op:1,tip:`<b>${name}</b><br>too few visits`};const f=sc(d.pct/mx,col);return{...f,dark:f.op>.55,l2:d.pct+'%',tip:`<b>${name}</b> · ${n}<br>${cat}: ${d.pct}% of visits<br>${fmt(d.visits)} visits`}},
      ward:(n,s,w)=>{const d=getw(cat,n,s,w);if(!d||d.visits<30)return{color:C.grey,op:1,tip:`<b>${w}</b> · ${s}<br>too few visits`};const all=Object.keys((CV[cat]||{}).ward||{}).filter(x=>x.indexOf(nk(n)+'|')===0).map(x=>CV[cat].ward[x].pct),mx=Math.max(1,...all);const op=+(0.15+0.8*Math.sqrt(d.pct/mx)).toFixed(2);return{color:col,op,dark:op>.55,l2:d.pct+'%',tip:`<b>${w}</b> · ${s}<br>${cat}: ${d.pct}% of visits<br>${fmt(d.visits)} visits`}},
      click:n=>{D.sel=n;draw()}}}
  function draw(){
    const cat=D.cat;one.style.display=D.view==='one'?'grid':'none';minis.style.display=D.view==='all'?'grid':'none';pl.style.display=D.view==='one'?'flex':'none';lv.style.display=(!D.sel||hasWards(D.sel))?'':'none';
    cr._set();
    pl.querySelectorAll('button').forEach(b=>{const on=b.dataset.v===cat;b.classList.toggle('on',on);b.style.background=on?CATCOL[b.dataset.v]:'';b.style.borderColor=CATCOL[b.dataset.v];b.style.color=on?'#fff':''});
    if(D.view==='one'){if(!OM)OM=newMap(oleft,520);geoMap(OM,D,cfgFor(cat,false));showSoon(OM);lqn.textContent=`Nandi wards · ${cat} · 1.0× = catchment average`;lq.innerHTML='';const rows=((P.lq||{})[cat]||[]).slice(0,5);
      if(!rows.length){tx(lq,'Not enough ward volume yet',{x:10,y:30,'font-size':11,fill:C.muted})}
      else{const mx=Math.max(2.5,...rows.map(r=>r[1]));el('line',{x1:150+240/mx,y1:2,x2:150+240/mx,y2:170,stroke:C.muted,'stroke-dasharray':'3 3'},lq);tx(lq,'1.0× = catchment average',{x:154+240/mx,y:184,'font-size':9,fill:C.muted});
        rows.forEach((r,i)=>{const y=4+i*33,w=r[1]/mx*240;el('rect',{x:150,y,width:w,height:22,fill:r[1]>=1?CATCOL[cat]:C.neutral,'fill-opacity':.85,rx:2},lq);tx(lq,r[0],{x:144,y:y+15,'text-anchor':'end','font-size':10.5,fill:C.text});tx(lq,r[1].toFixed(1)+'×',{x:154+w,y:y+15,'font-size':10.5,'font-weight':600,fill:C.text})})}
      note.textContent=!D.sel?'Share of each county\'s visits that are '+cat.toLowerCase()+'. Colours scale within this disease group. Zoom and pan freely, or click a county to jump to its sub-counties.':'Darker = higher share of this disease group. Grey = under 30 visits.'}
    else{if(!minis.children.length)cats.forEach(c=>{const m=H('div','mini','',minis);const t=H('div','t',c,m);t.style.color=CATCOL[c];MM[c]=newMap(m,230,true)});
      cats.forEach(c=>{geoMap(MM[c],D,cfgFor(c,true));showSoon(MM[c])});note.textContent='Each map has its own colour scale: compare the pattern within a map, not shades across maps. Click a county on any map to zoom all seven.'}
    leg.innerHTML=swatch(C.grey,'Under 30 visits, too few for a share')+swatch(CATCOL[cat],'lower share',.3)+swatch(CATCOL[cat],'higher share')}
  draw()}

/* ============================== NCD · ward maps (Tab 4) ============================== */
/* mode 'burden': NCD patients per ward, one pill per condition. mode 'recur': share of a ward's NCD patients seen in both 2024 and 2025.
   P.wards = [{w,s,lat,lon,rings,...}] with rings as lon/lat; point-only wards (no boundary) are drawn as circles. */
function mountNcd(root){
  const recur=P.mode==='recur',ALL=P.all||'All NCDs',NOUN=P.noun||'NCD',box=H('div','','',root),AMB=C.amber,GREY='#9CA3AF',thr=P.thr||20;
  let cur=(P.conds||[])[0];
  const pl=recur?null:pills(box,P.labels,P.labels[0],lab=>{cur=P.conds[P.labels.indexOf(lab)];draw()});
  const M=newMap(box,P.h||560,false),leg=H('div','lgd-row','',box);
  const pts=[];(P.wards||[]).forEach(w=>{if(w.rings&&w.rings.length)w.rings.forEach(r=>r.forEach(q=>pts.push([q[1],q[0]])));else if(w.lat!=null)pts.push([w.lat,w.lon])});
  M.bounds=pts.length?L.latLngBounds(pts):L.latLngBounds([[0.0,34.8],[0.45,35.5]]);
  if(P.hospital)L.marker([P.hospital.lat,P.hospital.lon],{icon:L.divIcon({className:'',html:'<div style="width:14px;height:14px;background:#0F6E56;border:2px solid #fff;box-shadow:0 0 0 1px #0F6E56"></div>',iconSize:[14,14]}),interactive:false,zIndexOffset:900}).bindTooltip('Kapsabet CRH',{permanent:true,direction:'right',className:'lt'}).addTo(M.map);
  function draw(){
    M.lg.clearLayers();
    const val=w=>recur?(w.pat>=thr?w.p:undefined):(w.v||{})[cur];
    const mx=Math.max(1,...(P.wards||[]).map(w=>{const v=val(w);return v===undefined?0:(recur?v:Math.sqrt(v))}));
    (P.wards||[]).forEach(w=>{
      const v=val(w),has=v!==undefined,col=has?(recur?AMB:C.blue):GREY,op=has?.18+.72*(recur?v:Math.sqrt(v))/mx:.10;
      const t=has?(recur?`<b>${w.w}</b> (${w.s})<br>${fmt(w.both)} of ${fmt(w.pat)} patients seen in both years (${v.toFixed(1)}%)`:`<b>${w.w}</b> (${w.s})<br>${fmt(v)} ${cur===ALL?NOUN:cur.toLowerCase()} patients`)
                    :`<b>${w.w}</b> (${w.s})<br>${recur?`only ${fmt(w.pat)} ${NOUN} patients, below the ${thr} needed to show a rate`:'no patients recorded'}`;
      if(w.rings&&w.rings.length)w.rings.forEach(r=>L.polygon(r.map(q=>[q[1],q[0]]),{color:'#fff',weight:1.2,fillColor:col,fillOpacity:op}).bindTooltip(t,TIPO).addTo(M.lg));
      else if(w.lat!=null)L.circleMarker([w.lat,w.lon],{radius:has?9:5,color:'#fff',weight:1.5,fillColor:col,fillOpacity:has?.85:.4}).bindTooltip(t,TIPO).addTo(M.lg)});
    const fm=x=>x>=10?fmt(x):(Math.round(x*10)/10).toString();
    const top=recur?mx:mx*mx,base=recur?AMB:C.blue;
    const ticks=recur?[0,top/2,top].map(x=>fm(x)+'%'):['1',fm(top/4),fm(top)];
    const grad=`linear-gradient(90deg,${rgba(base,.18)},${rgba(base,.9)})`;
    const sw=[`<span class="lgd-sw"><i style="background:${GREY};opacity:.45"></i>${recur?`Fewer than ${thr} ${NOUN} patients, too few for a rate`:'No patients recorded'}</span>`];
    if(!recur){sw.push(`<span class="lgd-sw"><i style="background:#fff;border:1px solid #9CA3AF"></i>Not matched to a boundary: ${fmt((P.unplaced||{})[cur]||0)} patients</span>`);
      if(P.outside)sw.push(`<span class="lgd-sw"><i style="background:${C.crit}"></i>${P.outside}</span>`)}
    leg.innerHTML=`<div class="lgd"><div class="lgd-t">${recur?`Share of the ward\'s ${NOUN} patients seen in both years`:'Patients per ward'+(cur===ALL?'':' · '+cur.toLowerCase())}</div>`
      +`<div class="lgd-bar" style="background:${grad}"></div><div class="lgd-tk">${ticks.map(x=>`<span>${x}</span>`).join('')}</div>`
      +(recur?'':`<div class="lgd-s">square-root scale, so Kapsabet does not wash out the rest</div>`)+`</div>`
      +`<div class="lgd-sws">${sw.join('')}</div>`}
  draw();fit(M)}

/* ============================== E · catchment shift ============================== */
function mountShift(root){
  const E={area:'all',yr:'2024',region:true},card=H('div','card','',root);
  const head=H('div','card-head','',card);const title=H('div','card-title','',head);
  const ctl=H('div','ctl','',card);
  const counties=Object.keys(CD).sort((a,b)=>a==='Nandi'?-1:b==='Nandi'?1:CD[b].v-CD[a].v);
  const s1=sel(ctl,'Area',[['all','All origin counties (Kenya map)']].concat(counties.map(n=>[n,n])),E.area,v=>{E.area=v;draw()});
  const s2=sel(ctl,'Map shows',[['2024','2024 visits'],['2025','2025 visits'],['chg','Change 2024 → 2025']],E.yr,v=>{E.yr=v;draw()});
  const g2=H('div','g2','',card),left=H('div','','',g2),right=H('div','','',g2);const M=newMap(left,520),leg=legend(left);
  const rl=legend(right),dbox=H('div','','',right),dnote=H('div','card-note','',right);
  const growth=H('div','card','',root);growth.style.marginTop='14px';
  function niceTicks(m){const step=Math.pow(10,Math.floor(Math.log10(Math.max(1,m)/3)));const s=[1,2,5,10].map(x=>x*step).find(x=>m/x<=5)||step*10;const t=[];for(let v=0;v<=m;v+=s)t.push(v);return t}
  function draw(){
    const all=E.area==='all',area=E.area,yr=E.yr,k=yr==='2024'?'v24':'v25',rows=all?[]:(SD[area]||[]);
    title.textContent=all?'Visits by origin county, 2024 vs 2025':`Visits by sub-county in ${area}, 2024 vs 2025`;
    const mx=Math.sqrt(Math.max(1,...rows.map(r=>Math.max(r.v24,r.v25))));
    const omx=Math.max(1,...Object.keys(CD).filter(x=>x!=='Nandi').map(x=>Math.max(CD[x].v24||0,CD[x].v25||0)));
    const baseCol=area==='Nandi'?C.primary:C.blue;
    if(!all)geoMap(M,{sel:area,lvl:'sub'},{county:n=>n===area?{color:baseCol,op:.35,tip:`<b>${n}</b><br>no sub-county detail in the data yet`}:{color:C.grey,op:.4,tip:n},click:n=>M.map.fitBounds(KB[n],{padding:[14,14]}),sub:(n,name)=>{const r=subRow(n,name);if(!r)return{color:C.grey,op:1,tip:name};
      if(yr==='chg'){const ch=r.v24?(r.v25-r.v24)/r.v24*100:0,neg=ch<0;return{color:neg?C.amber:C.primary,op:+(0.25+0.75*Math.min(1,Math.abs(ch)/80)).toFixed(2),dark:Math.abs(ch)>25,l2:(neg?'−':'+')+Math.abs(ch).toFixed(0)+'%',tip:`<b>${name}</b><br>${fmt(r.v24)} → ${fmt(r.v25)} visits`}}
      const f=sc(Math.sqrt(r[k])/mx,baseCol);return{...f,dark:f.op>.5,l2:fmt(r[k]),tip:`<b>${name}</b><br>${fmt(r[k])} visits in ${yr}`}}});
    else geoMap(M,{sel:null,region:E.region},{county:n=>{const d=CD[n];if(n==='Nandi')return{color:C.primary,op:1,tip:`<b>Nandi</b><br>${fmt(d.v24)} → ${fmt(d.v25)} visits<br>on its own scale`,l1:null};if(!d||Math.max(d.v24,d.v25)<10)return{color:C.grey,op:1,tip:`<b>${n}</b><br>too few visits for a yearly comparison`};
      const ch=d.v24?(d.v25-d.v24)/d.v24*100:0,neg=ch<0;if(yr==='chg')return{color:neg?C.amber:C.primary,op:+(0.25+0.75*Math.min(1,Math.abs(ch)/55)).toFixed(2),tip:`<b>${n}</b><br>${fmt(d.v24)} → ${fmt(d.v25)} visits (${neg?'−':'+'}${Math.abs(ch).toFixed(0)}%)`};
      return{...sc(d[k]/omx,C.blue),tip:`<b>${n}</b><br>${fmt(d[k])} visits in ${yr}`}},click:n=>M.map.fitBounds(KB[n],{padding:[14,14]})});
    showSoon(M);
    leg.innerHTML=(yr==='chg'?swatch(C.amber,'Fewer visits in 2025')+swatch(C.primary,'More visits in 2025'):swatch(all?C.blue:baseCol,`Visits in ${yr}, same colour scale in both years`))+(all?swatch(C.primary,'Nandi (own scale)')+swatch(C.grey,'Too few visits'):'');
    if(all){const a=H('a','back',E.region?'Show whole Kenya':'Zoom to western Kenya',leg);a.onclick=()=>{E.region=!E.region;draw()}}
    rl.innerHTML=swatch(C.blue,'2024')+swatch(C.amber,'2025');dbox.innerHTML='';dnote.textContent='';
    if(!all){
      if(!rows.length){dbox.innerHTML=`<div class="explain">${area} has no sub-county detail in the data yet, so there is nothing to compare below county level. County total: ${fmt((CD[area]||{}).v24||0)} visits in 2024 → ${fmt((CD[area]||{}).v25||0)} in 2025.</div>`}
      else{const r=rows.slice().sort((x,y)=>Math.max(y.v24,y.v25)-Math.max(x.v24,x.v25)).slice(0,12).map(x=>[x.n,x.v24,x.v25]);const m=Math.max(10,...r.flatMap(x=>[x[1],x[2]]));dumbbell(dbox,r,{max:m*1.08,ticks:niceTicks(m),right:60});
        dnote.textContent=area==='Nandi'?'Every non-Emgwen sub-county fell 75–85% while Emgwen rose: an export gap, not a clinical signal.':`${area}: small volumes, so one or two patients move these dots.`}}
    else{const cs=counties.filter(x=>x!=='Nandi'&&Math.max(CD[x].v24,CD[x].v25)>=10).sort((a,b)=>CD[b].v24-CD[a].v24).slice(0,12).map(x=>[x,CD[x].v24,CD[x].v25]);const m=Math.max(10,...cs.flatMap(r=>[r[1],r[2]]));dumbbell(dbox,cs,{max:m*1.08,ticks:niceTicks(m),right:60});
      const n=CD.Nandi;dnote.textContent=`Nandi sits on a different scale (${fmt(n.v24)} → ${fmt(n.v25)}, ${((n.v25-n.v24)/n.v24*100).toFixed(0)}%). Pick a county in the dropdown to see its sub-counties. Clicking the map only zooms; it does not change the dropdown or the chart.`}
    s1.value=E.area;s2.value=E.yr}
  draw();
  /* growth watch */
  const G=P.growth||[];const gh=H('div','card-head','',growth);H('div','card-title','Top localities, 2024 vs 2025 (excluding Kapsabet): growth watch',gh);const gl=H('div','legend','',gh);gl.style.margin='0';gl.innerHTML=swatch(C.blue,'2024')+swatch(C.amber,'2025')+swatch(C.primary,'Fell less than peers');
  const gr=G.map(r=>[r.n,r.v24,r.v25,r]);const gm=Math.max(1,...gr.flatMap(r=>[r[1],r[2]]));
  const box=H('div','','',growth);box.style.maxWidth='900px';dumbbell(box,gr,{max:gm*1.08,ticks:niceTicks(gm),right:190,W:380,L:100,hl:r=>r[3].watch,tag:(svg,r,y,hi)=>{const rel=r[3].pts;tx(svg,(rel>0?'+':rel<0?'−':'±')+Math.abs(rel)+' pts vs peers',{x:100+380+55,y:y+4,'font-size':10,'font-weight':hi?700:400,fill:hi?C.primary:C.muted})}});
  const w=G.filter(r=>r.watch).map(r=>r.n);H('div','explain',`<b>Growth watch.</b> The 2025 feeds are incomplete, so the useful signal is who fell <i>least</i>, not who rose. The typical locality here changed ${P.growthMedian}%. Highlighted rows beat that by 2 points or more and gained share of their sub-county${w.length?': <b>'+w.join(', ')+'</b>':''}. Once the feeds are fixed this card re-ranks on real growth in visits and new patients.`,growth)}

/* ============================== F · female attendance ============================== */
function mountFemale(root){
  const F={sel:null,region:false},base=P.femaleBaseline||57,card=H('div','card','',root);
  const head=H('div','card-head','',card);H('div','card-title','Female attendance by county and sub-county',head);
  const leg=legend(card);leg.innerHTML=swatch(C.primary,`Above ${base}% baseline`)+swatch(C.amber,'Below baseline')+swatch(C.grey,'Under 30 visits');
  const cr=crumb(card,F,draw,()=>' › Sub-counties'),M=newMap(card,500),note=H('div','card-note','',card);
  const col=v=>({color:v>=base?C.primary:C.amber,op:+(0.18+0.82*Math.min(1,Math.abs(v-base)/14)).toFixed(2)});
  function draw(){geoMap(M,F,{county:n=>{const d=CD[n];if(!d||d.v<30||d.f==null)return{color:C.grey,op:1,tip:`<b>${n}</b><br>${d?fmt(d.v):0} visits: too few to show a split`};const f=col(d.f);return{...f,dark:Math.abs(d.f-base)>7,tip:`<b>${n}</b><br>${d.f.toFixed(1)}% female · ${fmt(d.v)} visits`}},
      sub:(n,name)=>{const r=subRow(n,name);if(!r||r.v<30||r.f==null)return{color:C.grey,op:1,tip:`<b>${name}</b><br>too few visits`};const f=col(r.f);return{...f,dark:Math.abs(r.f-base)>7,l2:r.f.toFixed(1)+'%',tip:`<b>${name}</b> · ${n}<br>${r.f.toFixed(1)}% female · ${fmt(r.v)} visits`}},click:n=>{F.sel=n;draw()}});
    cr._set();note.textContent=!F.sel?'Counties under 30 visits are grey. Zoom and pan freely, or click a county to see its sub-counties.':(layout(F.sel)?'Sub-county female share against the hospital baseline.':`${F.sel} has no sub-county detail in the data yet.`)}
  draw()}

({catchment:mountCatchment,rings:mountRings,disease:mountDisease,shift:mountShift,female:mountFemale,ncd:mountNcd})[window.KIND](document.getElementById('root'));
})();
