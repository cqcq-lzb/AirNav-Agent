"""三维 viewer 的 HTML 模板。

产物是**单个自包含 HTML**：
  - Three.js（UMD）内联，不依赖网络；
  - 几何数据经量化 + base64 内联；
  - 打开即用，可直接发给别人看，也可以离线录屏。

视觉语义
--------
  - 气道表面：菲涅尔着色，掠射处发亮 —— 管腔形状能读出来，而不是糊成一团雾；
  - 中心线  ：按邻接边画线段（**不是**按节点顺序连折线，那会连成乱麻）；
  - 路径    ：画成「管腔 + 器械」同轴双管，器械管按余量着色，
              一眼就能看出哪一段贴壁；
  - 最窄处  ：红色环 + 脉冲；
  - 结节    ：按体积定大小，选中项高亮；
  - 修复段  ：琥珀色，凡是依赖分割修复的路径必须能被看出来。

工程约定
--------
本文件只允许用编辑工具（Write / Edit）写入。用本机的 python.exe 覆写会被
DLP 透明加密，此后 Read / Edit 只能读到密文。
"""
from __future__ import annotations

import json
from pathlib import Path

_VENDOR = Path(__file__).with_name("vendor") / "three.min.js"

_STYLE = """
:root{
  --bg:#eef1f6; --bg2:#dde5ef; --panel:rgba(255,255,255,.9); --panel-brd:rgba(15,40,80,.12);
  --ink:#16202e; --ink-dim:#5a6a80; --ink-faint:#8896a9; --accent:#2f6fd0;
  --ok:#1f9d55; --warn:#c98a12; --bad:#d2413a;
  --shadow:0 1px 2px rgba(16,32,64,.06), 0 10px 28px rgba(16,32,64,.10);
}
body.dark{
  --bg:#0b0f15; --bg2:#141b25; --panel:rgba(20,27,37,.92); --panel-brd:rgba(140,180,255,.15);
  --ink:#e6edf6; --ink-dim:#9fb0c6; --ink-faint:#6c7f97; --accent:#5aa2ff;
  --ok:#3fbf74; --warn:#e8b84b; --bad:#ff6b62;
  --shadow:0 1px 2px rgba(0,0,0,.45), 0 12px 34px rgba(0,0,0,.5);
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;overflow:hidden}
body{
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",system-ui,sans-serif;
  color:var(--ink);
  background:radial-gradient(125% 95% at 50% -10%, var(--bg2) 0%, var(--bg) 60%);
  transition:background .25s ease,color .25s ease;
}
canvas{display:block}
#stage{position:absolute;inset:0}

.panel{
  position:absolute;top:14px;width:296px;max-height:calc(100% - 28px);
  background:var(--panel);border:1px solid var(--panel-brd);border-radius:15px;
  box-shadow:var(--shadow);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  overflow:auto;overscroll-behavior:contain;
}
#left{left:14px}
#right{right:14px;width:252px}
.panel::-webkit-scrollbar{width:8px}
.panel::-webkit-scrollbar-thumb{background:var(--panel-brd);border-radius:4px}

.head{padding:13px 15px 11px;border-bottom:1px solid var(--panel-brd)}
.head h1{margin:0;font-size:15px;font-weight:660;letter-spacing:-.01em}
.head .sub{margin-top:3px;font-size:11.5px;color:var(--ink-faint);font-variant-numeric:tabular-nums}
.badge{
  display:inline-block;margin-left:6px;padding:1px 7px;border-radius:20px;font-size:10.5px;
  font-weight:600;vertical-align:1px;background:rgba(47,111,208,.14);color:var(--accent);
}
body.dark .badge{background:rgba(90,162,255,.18)}
section{padding:12px 15px;border-bottom:1px solid var(--panel-brd)}
section:last-child{border-bottom:0}
h2{
  margin:0 0 9px;font-size:10.5px;font-weight:700;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-faint);
}

.kv{display:flex;justify-content:space-between;gap:10px;padding:3.5px 0;font-size:12.5px}
.kv .k{color:var(--ink-dim);white-space:nowrap}
.kv .v{font-weight:600;font-variant-numeric:tabular-nums;text-align:right}
.v.good{color:var(--ok)} .v.warn{color:var(--warn)} .v.bad{color:var(--bad)}
.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11.5px}

.toggle{
  display:flex;align-items:center;gap:9px;padding:5px 0;cursor:pointer;user-select:none;
  font-size:12.5px;color:var(--ink-dim);
}
.toggle:hover{color:var(--ink)}
.toggle input{appearance:none;margin:0;width:32px;height:18px;border-radius:10px;flex:0 0 auto;
  background:rgba(120,140,165,.38);position:relative;
  transition:background .18s ease;cursor:pointer}
.toggle input::after{
  content:"";position:absolute;top:2px;left:2px;width:14px;height:14px;border-radius:50%;
  background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.3);transition:transform .18s cubic-bezier(.4,1.3,.5,1);
}
.toggle input:checked{background:var(--accent)}
.toggle input:checked::after{transform:translateX(14px)}
.swatch{width:10px;height:10px;border-radius:3px;flex:0 0 auto}
.toggle .meta{margin-left:auto;font-size:11px;color:var(--ink-faint);font-variant-numeric:tabular-nums}

.grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px}
button{
  font:inherit;font-size:12px;padding:7px 9px;border-radius:9px;cursor:pointer;
  border:1px solid var(--panel-brd);background:transparent;color:var(--ink-dim);
  transition:all .15s ease;white-space:nowrap;
}
button:hover{color:var(--accent);border-color:rgba(47,111,208,.45);background:rgba(47,111,208,.07)}
body.dark button:hover{background:rgba(90,162,255,.1)}
button.on{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
button.primary:hover{filter:brightness(1.09);color:#fff;background:var(--accent)}

input[type=range]{width:100%;accent-color:var(--accent);margin:4px 0}

.ramp{height:9px;border-radius:5px;margin:7px 0 4px;
  background:linear-gradient(90deg,#c9403a,#d98a34,#d8b13a,#7fb547,#2f8f4e)}
.ramp-lab{display:flex;justify-content:space-between;font-size:10.5px;color:var(--ink-faint)}

.hint{font-size:11.5px;color:var(--ink-faint);line-height:1.6}
ul.warn{margin:5px 0 0;padding-left:16px;font-size:12px;color:var(--ink-dim)}
ul.warn li{margin:5px 0}
ul.warn li::marker{color:var(--warn)}

#tip{
  position:absolute;pointer-events:none;opacity:0;transition:opacity .12s ease;
  background:var(--panel);border:1px solid var(--panel-brd);border-radius:9px;
  box-shadow:var(--shadow);padding:7px 10px;font-size:12px;backdrop-filter:blur(12px);
  max-width:240px;line-height:1.45;
}
#tip .t{font-weight:660;margin-bottom:2px}
#tip .d{color:var(--ink-dim);font-size:11.5px;font-variant-numeric:tabular-nums}

#bottom{
  position:absolute;left:50%;transform:translateX(-50%);bottom:16px;
  display:flex;align-items:center;gap:12px;padding:9px 15px;
  background:var(--panel);border:1px solid var(--panel-brd);border-radius:13px;
  box-shadow:var(--shadow);backdrop-filter:blur(16px);min-width:460px;
}
#bottom .lbl{font-size:11.5px;color:var(--ink-faint);font-variant-numeric:tabular-nums;white-space:nowrap}
#scrub{flex:1}
#stats{
  position:absolute;left:50%;transform:translateX(-50%);bottom:74px;
  font-size:11.5px;color:var(--ink-faint);background:var(--panel);
  padding:5px 13px;border-radius:9px;border:1px solid var(--panel-brd);
  font-variant-numeric:tabular-nums;white-space:nowrap;
}
#stats b{color:var(--ink);font-weight:660}
#toast{
  position:absolute;left:50%;top:18px;transform:translateX(-50%);
  padding:7px 15px;border-radius:10px;background:var(--panel);border:1px solid var(--panel-brd);
  box-shadow:var(--shadow);font-size:12.5px;font-weight:600;opacity:0;
  transition:opacity .25s ease;pointer-events:none;
}
.hide{display:none!important}
@media (max-width:1220px){#left,#right{width:252px} #bottom{min-width:360px}}
"""

_APP_JS = r"""
const P = __AIRNAV_PAYLOAD__;

// ---------------------------------------------------------------- 解码
function b64ToBytes(b64){
  const bin = atob(b64), out = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) out[i] = bin.charCodeAt(i);
  return out;
}
function decodeMesh(g){
  const vq = new Uint16Array(b64ToBytes(g.vertices).buffer);
  const nq = new Int8Array(b64ToBytes(g.normals).buffer);
  const fc = new Uint32Array(b64ToBytes(g.faces).buffer);
  const n = g.vertexCount, o = g.vertexOrigin, s = g.vertexScale;

  const pos = new Float32Array(n*3), nor = new Float32Array(n*3);
  for(let i=0;i<n;i++){
    pos[i*3  ] = o[0] + vq[i*3  ]*s;
    pos[i*3+1] = o[1] + vq[i*3+1]*s;
    pos[i*3+2] = o[2] + vq[i*3+2]*s;
    nor[i*3  ] = nq[i*3  ]/127;
    nor[i*3+1] = nq[i*3+1]/127;
    nor[i*3+2] = nq[i*3+2]/127;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
  geo.setAttribute('normal',   new THREE.BufferAttribute(nor,3));
  geo.setIndex(new THREE.BufferAttribute(fc,1));
  return geo;
}

// ---------------------------------------------------------------- 余量配色
// 余量 c（mm）从「约 -1」到「>= safe」，映射到 红 -> 黄 -> 绿
function clearanceColor(c, safe, tight){
  if(c >= safe) return new THREE.Color(0x2f8f4e);
  if(c >= tight){
    const t = (c - tight)/(safe - tight);       // 0 贴壁 -> 1 宽松
    return new THREE.Color().setHSL(0.02 + 0.30*t, 0.74, 0.42);
  }
  const t = Math.max(0, Math.min(1, (c + 1.2)/1.7));
  return new THREE.Color().setHSL(0.0, 0.72, 0.30 + 0.18*t);
}

// ---------------------------------------------------------------- 平行传输管道
// 沿折线生成可变半径管道。用平行传输求帧，避免 Frenet 帧在直线段翻转。
function buildTube(points, radiusAt, radialSeg, colors){
  const n = points.length, RING = radialSeg + 1;
  const P3 = points.map(p => new THREE.Vector3(p[0],p[1],p[2]));
  const tan = [];
  for(let i=0;i<n;i++){
    const a = P3[Math.max(0,i-1)], b = P3[Math.min(n-1,i+1)];
    const t = new THREE.Vector3().subVectors(b,a);
    if(t.lengthSq() < 1e-12) t.set(0,0,1);
    tan.push(t.normalize());
  }
  let up = new THREE.Vector3(0,1,0);
  if(Math.abs(tan[0].dot(up)) > 0.95) up.set(1,0,0);
  let nrm = new THREE.Vector3().crossVectors(up, tan[0]).normalize();

  const verts = [], norms = [], cols = [], idx = [];
  for(let i=0;i<n;i++){
    if(i > 0){
      const q = new THREE.Quaternion().setFromUnitVectors(tan[i-1], tan[i]);
      nrm.applyQuaternion(q).normalize();
    }
    const bin = new THREE.Vector3().crossVectors(tan[i], nrm).normalize();
    const r = radiusAt(i);
    const c = colors ? colors(i) : null;
    for(let j=0;j<=radialSeg;j++){
      const a = (j/radialSeg)*Math.PI*2;
      const dir = new THREE.Vector3()
        .addScaledVector(nrm, Math.cos(a))
        .addScaledVector(bin, Math.sin(a));
      verts.push(P3[i].x + dir.x*r, P3[i].y + dir.y*r, P3[i].z + dir.z*r);
      norms.push(dir.x, dir.y, dir.z);
      if(c) cols.push(c.r, c.g, c.b);
    }
  }
  for(let i=0;i<n-1;i++){
    for(let j=0;j<radialSeg;j++){
      const a = i*RING + j, b = a + RING;
      idx.push(a, b, a+1, a+1, b, b+1);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts,3));
  geo.setAttribute('normal',   new THREE.Float32BufferAttribute(norms,3));
  if(cols.length) geo.setAttribute('color', new THREE.Float32BufferAttribute(cols,3));
  geo.setIndex(idx);
  return geo;
}

// ---------------------------------------------------------------- 场景
const stage = document.getElementById('stage');
const scene  = new THREE.Scene();
const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
stage.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(42, innerWidth/innerHeight, 0.4, 8000);

const bmin = new THREE.Vector3(...P.bounds.min), bmax = new THREE.Vector3(...P.bounds.max);
const center = new THREE.Vector3().addVectors(bmin,bmax).multiplyScalar(0.5);
const radius = Math.max(bmax.distanceTo(bmin)*0.5, 40);

scene.add(new THREE.HemisphereLight(0xffffff, 0x8fa3bb, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 1.15); key.position.set(1,1.4,0.9); scene.add(key);
const rimL = new THREE.DirectionalLight(0xbcd2f0, 0.7); rimL.position.set(-1.1,-0.5,-1); scene.add(rimL);

// 参考网格与初始取景都以「气道自身」的包围盒为准。
// 用全场包围盒会被远处的小结节把画面撑散，气管反而被切掉。
const ab = P.airwayBounds || {min:P.bounds.min, max:P.bounds.max};
const abMin = new THREE.Vector3(...ab.min), abMax = new THREE.Vector3(...ab.max);
const abCenter = new THREE.Vector3().addVectors(abMin, abMax).multiplyScalar(0.5);
const abRadius = Math.max(abMax.distanceTo(abMin)*0.5, 30);

const gridSize = Math.max(Math.ceil(abRadius*2.8/50)*50, 100);
const grid = new THREE.GridHelper(gridSize, Math.round(gridSize/25), 0xa8b8cc, 0xccd7e4);
grid.position.set(abCenter.x, abMin.y - 10, abCenter.z);
grid.material.transparent = true; grid.material.opacity = 0.42;
scene.add(grid);

// ---------------------------------------------------------------- 气道表面
// 菲涅尔着色：视线垂直穿壁时几乎透明，掠射时壁面发亮，管腔形状可读。
// 剖切在着色器里手动 discard，避免引入 clipping chunks 又要维护两套材质。
const airwayUniforms = {
  uColor:     { value: new THREE.Color(0x8fb0d4) },
  uBaseAlpha: { value: 0.06 },
  uRimAlpha:  { value: 0.95 },
  uRimPower:  { value: 2.3 },
  uLightDir:  { value: new THREE.Vector3(0.55, 0.78, 0.42).normalize() },
  uClipY:     { value: 1e9 },
};

// 气道表面用**两遍渲染**：先画背面（远侧管壁），再画正面（近侧管壁）。
// 单个 DoubleSide 网格在透明混合下没有确定的绘制顺序，前后壁会随机穿插，
// 画面就出现一圈圈波纹和雾感 —— 这是上一版最明显的毛病。
// 拆成两个 mesh 之后，靠 renderOrder 强制「由远及近」，混合结果就稳定了。
const AIRWAY_VERTEX_SHADER = [
  'varying vec3 vN; varying vec3 vV; varying vec3 vW;',
  'void main(){',
  '  vec4 wp = modelMatrix * vec4(position, 1.0);',
  '  vW = wp.xyz;',
  '  vN = normalize(mat3(modelMatrix) * normal);',
  '  vV = normalize(cameraPosition - wp.xyz);',
  '  gl_Position = projectionMatrix * viewMatrix * wp;',
  '}',
].join('\n');
const AIRWAY_FRAGMENT_SHADER = [
  'uniform vec3 uColor; uniform float uBaseAlpha; uniform float uRimAlpha;',
  'uniform float uRimPower; uniform vec3 uLightDir; uniform float uClipY;',
  'varying vec3 vN; varying vec3 vV; varying vec3 vW;',
  'void main(){',
  '  if(vW.y > uClipY) discard;',
  '  vec3 n = normalize(vN); vec3 v = normalize(vV);',
  '  float ndv = abs(dot(n, v));',
  '  float rim = pow(1.0 - ndv, uRimPower);',
  '  float lam = 0.42 + 0.58 * max(dot(n, normalize(uLightDir)), 0.0);',
  '  gl_FragColor = vec4(uColor * lam, clamp(uBaseAlpha + uRimAlpha * rim, 0.0, 1.0));',
  '  #include <__AIRNAV_ENCODING_CHUNK__>',
  '}',
].join('\n');

function makeAirwayMaterial(side, uniforms){
  return new THREE.ShaderMaterial({
    uniforms, side, transparent: true, depthWrite: false,
    vertexShader: AIRWAY_VERTEX_SHADER,
    fragmentShader: AIRWAY_FRAGMENT_SHADER,
  });
}
// 两遍共享 color / rim / clip 这些全局 uniform（同一份对象引用，改一处两遍都生效），
// 只有 uBaseAlpha 各用各的 —— 远侧壁要看得见，近侧壁要几乎透明。
const airwayUniformsBack  = Object.assign({}, airwayUniforms, { uBaseAlpha: { value: 0.13 } });
const airwayUniformsFront = Object.assign({}, airwayUniforms, { uBaseAlpha: { value: 0.02 } });

const airwayGeo = decodeMesh(P.airway);
const airwayMatBack  = makeAirwayMaterial(THREE.BackSide,  airwayUniformsBack);
const airwayMatFront = makeAirwayMaterial(THREE.FrontSide, airwayUniformsFront);
const airwayBack  = new THREE.Mesh(airwayGeo, airwayMatBack);
const airwayFront = new THREE.Mesh(airwayGeo, airwayMatFront);
airwayBack.frustumCulled = false;  airwayBack.renderOrder = 3;
airwayFront.frustumCulled = false; airwayFront.renderOrder = 4;
const airwayGroup = new THREE.Group();
airwayGroup.add(airwayBack, airwayFront);
scene.add(airwayGroup);

// 腔内 / 腔外、浅色 / 深色，一共四种组合。
// 腔外：背面 + 正面两遍，底色都压得很低，只让掠射的管壁发亮，
//       整体像一层玻璃铸造件，能看到内部的路径；
// 腔内：只留背面、底色抬高 —— 相机停在管腔里，等价于支气管镜看到的画面。
// 颜色必须跟着主题走：浅色主题下用浅蓝会把管壁冲成一片白，
// 深色主题下用暗色又会糊成黑块。
//
// rim/power 这两个值是调出来的：rim 太大时，凡是接近平行视线的面都会
// 被拉到完全不透明，气道树会冒出一大片边缘齐整的「白翅膀」；
// power 调高让衰减更陡，边缘就只剩细细一道亮边。
let darkMode = false;
const AIRWAY_LOOK = {
  outside: {
    light: { color:0x8fb0d4, backBase:0.15, frontBase:0.02, rim:0.66, power:3.05 },
    dark:  { color:0x86b4e8, backBase:0.13, frontBase:0.02, rim:0.70, power:3.05 },
  },
  inside: {
    light: { color:0x67849f, backBase:0.30, frontBase:0.00, rim:0.58, power:1.60 },
    dark:  { color:0x86b4e8, backBase:0.30, frontBase:0.00, rim:0.62, power:1.45 },
  },
};
function refreshAirwayMaterial(){
  const mode = insideMode ? 'inside' : 'outside';
  const look = AIRWAY_LOOK[mode][darkMode ? 'dark' : 'light'];
  airwayUniforms.uColor.value.set(look.color);
  airwayUniforms.uRimAlpha.value  = look.rim;
  airwayUniforms.uRimPower.value  = look.power;
  airwayUniformsBack.uBaseAlpha.value  = look.backBase;
  airwayUniformsFront.uBaseAlpha.value = look.frontBase;
  // 腔内时近侧壁正好贴在镜头上，必须彻底关掉，否则整个画面被它糊住
  airwayFront.visible = !insideMode;
  airwayMatBack.needsUpdate = true;
  airwayMatFront.needsUpdate = true;
}

// ---------------------------------------------------------------- 中心线
// positions 是图的节点数组，必须按 edges 画线段。
const clGeo = new THREE.BufferGeometry();
clGeo.setAttribute('position', new THREE.Float32BufferAttribute(P.centerline.positions.flat(), 3));
clGeo.setIndex(P.centerline.edges.flat());
const centerline = new THREE.LineSegments(clGeo, new THREE.LineBasicMaterial({
  color: 0x8496ab, transparent:true, opacity:0.40
}));
centerline.renderOrder = 1;
scene.add(centerline);

// 分叉点
const junctionPos = [];
(() => {
  const deg = new Uint16Array(P.centerline.nodeCount);
  P.centerline.edges.forEach(([a,b]) => { deg[a]++; deg[b]++; });
  for(let i=0;i<deg.length;i++) if(deg[i] >= 3) junctionPos.push(...P.centerline.positions[i]);
})();
const jGeo = new THREE.BufferGeometry();
jGeo.setAttribute('position', new THREE.Float32BufferAttribute(junctionPos, 3));
const junctions = new THREE.Points(jGeo, new THREE.PointsMaterial({
  color: 0x6b7f96, size: 2.6, sizeAttenuation: true, transparent:true, opacity:0.75
}));
junctions.renderOrder = 2;
scene.add(junctions);

// ---------------------------------------------------------------- 路径双管
const rPts = P.route.positions;
const clr  = P.route.clearanceMm;
const safe = P.clearanceScale.safeMm, tight = P.clearanceScale.tightMm;

const lumen = new THREE.Mesh(
  buildTube(rPts, i => Math.max(P.route.lumenRadiiMm[i], 0.8), 10, null),
  new THREE.MeshStandardMaterial({
    color: 0xcfdae8, roughness:1.0, metalness:0.0, transparent:true, opacity:0.28,
    side: THREE.DoubleSide, depthWrite:false
  })
);
lumen.renderOrder = 5;
scene.add(lumen);

const deviceGeo = buildTube(
  rPts,
  i => Math.max(P.route.deviceDrawRadiusMm, 0.5),
  12,
  i => clearanceColor(clr[i], safe, tight)
);
const device = new THREE.Mesh(deviceGeo, new THREE.MeshStandardMaterial({
  vertexColors:true, roughness:0.32, metalness:0.2
}));
device.renderOrder = 6;
scene.add(device);

// ---------------------------------------------------------------- 结节
const noduleGroup = new THREE.Group(); scene.add(noduleGroup);
const noduleHit = [];
P.nodules.forEach(nd => {
  const r = nd.drawRadiusMm, sel = nd.selected;

  // 选中项：实心橙色 + 自转圆环 —— 这是本次规划的靶点。
  // 未选中项：半透明内核 + 线框 —— 刻意画成「候选标记」而不是实心球。
  //   之前用实心球时，大结节会变成画面里一个没有立体感的大白盘，
  //   既抢视线又容易被误认成病灶本身。
  const geom = new THREE.SphereGeometry(r, sel ? 28 : 18, sel ? 20 : 12);
  const ball = new THREE.Mesh(geom, sel
    ? new THREE.MeshStandardMaterial({
        color:0xd9762a, roughness:0.40, metalness:0.08,
        emissive:0x7d3a06, emissiveIntensity:0.34
      })
    : new THREE.MeshStandardMaterial({
        color:0x93a7c0, roughness:0.6, metalness:0.0,
        transparent:true, opacity:0.30, depthWrite:false
      }));
  ball.position.set(...nd.position);
  ball.userData = nd;
  ball.renderOrder = 7;
  noduleGroup.add(ball);
  noduleHit.push(ball);

  if(!sel){
    const wire = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({
      color:0x7d8ea6, wireframe:true, transparent:true, opacity:0.55
    }));
    wire.position.copy(ball.position);
    wire.renderOrder = 7;
    noduleGroup.add(wire);
  } else {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(r*1.6, r*1.95, 48),
      new THREE.MeshBasicMaterial({color:0xd9762a, side:THREE.DoubleSide, transparent:true, opacity:0.9})
    );
    ring.position.copy(ball.position);
    ring.rotation.x = -Math.PI/2;
    ring.userData.spin = true;
    noduleGroup.add(ring);
  }
});

// ---------------------------------------------------------------- 分割修复段
let repairedGroup = null;
const repairedKeys = Object.keys(P.repaired || {});
if(repairedKeys.length){
  repairedGroup = new THREE.Group(); scene.add(repairedGroup);
  const tints = {recovered:0xd0892f, added:0xc65a8f, bridge:0x8f6ec4};
  repairedKeys.forEach(k => {
    const mesh = new THREE.Mesh(decodeMesh(P.repaired[k].geometry), new THREE.MeshStandardMaterial({
      color: tints[k] || 0xd0892f, roughness:0.7, metalness:0.0,
      transparent:true, opacity:0.8, side:THREE.DoubleSide
    }));
    mesh.frustumCulled = false;
    mesh.renderOrder = 3;
    repairedGroup.add(mesh);
  });
}

// ---------------------------------------------------------------- 标记
const bn = P.bottleneck;
const bPos = new THREE.Vector3(...bn.position);
const bGroup = new THREE.Group();
{
  const i = Math.min(Math.max(bn.index,1), rPts.length-1);
  const dir = new THREE.Vector3(...rPts[i]).sub(new THREE.Vector3(...rPts[i-1])).normalize();
  bGroup.position.copy(bPos);
  bGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1), dir);
  bGroup.add(new THREE.Mesh(
    new THREE.TorusGeometry(Math.max(bn.diameterMm/2*1.4, 1.8), 0.36, 12, 48),
    new THREE.MeshBasicMaterial({color:0xd2413a})
  ));
  bGroup.renderOrder = 8;
}
scene.add(bGroup);

const ePos = new THREE.Vector3(...P.entryPoint);
const entry = new THREE.Mesh(
  new THREE.ConeGeometry(2.4, 7.0, 22),
  new THREE.MeshStandardMaterial({color:0x2f8f4e, roughness:0.4, metalness:0.1,
    emissive:0x0d3d20, emissiveIntensity:0.3})
);
entry.position.copy(ePos);
entry.quaternion.setFromUnitVectors(
  new THREE.Vector3(0,1,0),
  new THREE.Vector3(...rPts[Math.min(8, rPts.length-1)]).sub(ePos).normalize()
);
entry.renderOrder = 8;
scene.add(entry);

// ---------------------------------------------------------------- 轨道控制
const target = new THREE.Vector3().copy(abCenter);
let theta = 0.62, phi = 1.02, dist = abRadius*2.45;
function applyOrbit(){
  phi  = Math.max(0.10, Math.min(Math.PI-0.10, phi));
  dist = Math.max(abRadius*0.08, Math.min(abRadius*10, dist));
  camera.position.set(
    target.x + dist*Math.sin(phi)*Math.sin(theta),
    target.y + dist*Math.cos(phi),
    target.z + dist*Math.sin(phi)*Math.cos(theta)
  );
  camera.lookAt(target);
}

let drag = null;
const dom = renderer.domElement;
dom.addEventListener('pointerdown', e => {
  drag = {x:e.clientX, y:e.clientY, b:e.button};
  try{ dom.setPointerCapture(e.pointerId); }catch(_){}
  autoRotate = false; syncAutoBtn();
});
dom.addEventListener('pointerup', e => { drag = null; try{dom.releasePointerCapture(e.pointerId);}catch(_){} });
dom.addEventListener('pointermove', e => {
  if(!drag) return;
  const dx = e.clientX-drag.x, dy = e.clientY-drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if(drag.b === 0){ theta -= dx*0.0062; phi -= dy*0.0062; }
  else {
    const fwd = camera.getWorldDirection(new THREE.Vector3());
    const right = new THREE.Vector3().crossVectors(camera.up, fwd).normalize();
    const upv = new THREE.Vector3().crossVectors(fwd, right).normalize();
    const k = dist*0.0016;
    target.addScaledVector(right, dx*k).addScaledVector(upv, -dy*k);
  }
  applyOrbit();
});
dom.addEventListener('wheel', e => {
  e.preventDefault(); dist *= (1 + Math.sign(e.deltaY)*0.09); applyOrbit();
}, {passive:false});
dom.addEventListener('contextmenu', e => e.preventDefault());

let pinch = null;
dom.addEventListener('touchstart', e => { if(e.touches.length===2) pinch = touchDist(e); }, {passive:true});
dom.addEventListener('touchmove', e => {
  if(e.touches.length===2 && pinch){ const d = touchDist(e); dist *= pinch/d; pinch = d; applyOrbit(); }
}, {passive:true});
function touchDist(e){
  const t = e.touches;
  return Math.hypot(t[0].clientX-t[1].clientX, t[0].clientY-t[1].clientY);
}

// ---------------------------------------------------------------- 悬停拾取
const ray = new THREE.Raycaster();
const tip = document.getElementById('tip');
const mouse = new THREE.Vector2();
const deviceHit = new THREE.Mesh(deviceGeo, new THREE.MeshBasicMaterial({visible:false}));
scene.add(deviceHit);

dom.addEventListener('pointermove', e => {
  mouse.x = (e.clientX/innerWidth)*2-1;
  mouse.y = -(e.clientY/innerHeight)*2+1;
  if(drag || flying){ hideTip(); return; }
  ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObjects([...noduleHit, deviceHit], false);
  if(!hits.length){ hideTip(); return; }
  const h = hits[0];
  if(h.object === deviceHit && h.face){
    const i = Math.floor(h.face.a/13);
    const j = Math.min(i, rPts.length-1);
    showTip(e, '路径 · 航点 ' + (j+1),
      '累计 ' + P.route.cumulativeMm[j] + ' mm · 管腔直径 ' +
      (P.route.lumenRadiiMm[j]*2).toFixed(2) + ' mm · 器械余量 ' + clr[j].toFixed(2) + ' mm');
  } else {
    const nd = h.object.userData;
    showTip(e, '结节候选 ' + nd.clientId,
      '体积 ' + nd.volumeMm3 + ' mm³ · 等效直径 ' + (nd.equivalentRadiusMm*2).toFixed(1) +
      ' mm · 服务端编号 ' + nd.serverId);
  }
});
function showTip(e, title, desc){
  tip.innerHTML = '<div class="t">' + title + '</div><div class="d">' + desc + '</div>';
  tip.style.opacity = 1;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(e.clientX+16, innerWidth-w-10) + 'px';
  tip.style.top  = Math.min(e.clientY+16, innerHeight-h-10) + 'px';
}
function hideTip(){ tip.style.opacity = 0; }

// ---------------------------------------------------------------- 相机运动
// instantCamera 为 true 时不做补间，直接落位。
// 无头浏览器截图/录屏时补间动画的结束时刻不可靠（虚拟时钟推进节奏和 rAF 不吻合），
// 会出现「截到相机飞行途中」的情况，所以截图路径必须走瞬时定位。
let instantCamera = false;
function flyTo(pos, look, ms, done){
  if(instantCamera){
    camera.position.copy(pos);
    target.copy(look);
    camera.lookAt(target);
    syncOrbit();
    flying = false;
    if(done) done();
    return;
  }
  ms = ms || 900;
  const p0 = camera.position.clone(), t0 = target.clone();
  const p1 = pos.clone(), t1 = look.clone();
  const start = performance.now();
  flying = true;
  (function step(now){
    const u = Math.min(1, (now-start)/ms);
    const k = u<0.5 ? 4*u*u*u : 1-Math.pow(-2*u+2,3)/2;
    camera.position.lerpVectors(p0,p1,k);
    target.lerpVectors(t0,t1,k);
    camera.lookAt(target);
    if(u < 1) requestAnimationFrame(step);
    else { flying = false; syncOrbit(); if(done) done(); }
  })(start);
}
function syncOrbit(){
  const d = camera.position.clone().sub(target);
  dist = Math.max(d.length(), 1e-3);
  phi = Math.acos(Math.max(-1,Math.min(1,d.y/dist)));
  theta = Math.atan2(d.x, d.z);
}
function frameOn(p, span, ms, done){
  const dir = camera.position.clone().sub(target);
  if(dir.lengthSq() < 1e-6) dir.set(0,0,1);
  dir.normalize();
  flyTo(p.clone().addScaledVector(dir, span), p, ms, done);
}
const routeMid = new THREE.Vector3(...rPts[Math.floor(rPts.length/2)]);
const overviewPos = abCenter.clone().add(new THREE.Vector3(
  abRadius*1.35, abRadius*0.78, abRadius*1.5
));
const views = {
  overview:   () => flyTo(overviewPos, abCenter, 1000),
  front:      () => flyTo(abCenter.clone().add(new THREE.Vector3(0,0,abRadius*2.7)), abCenter, 1000),
  route:      () => frameOn(routeMid, Math.max(abRadius*0.42, 55), 900),
  bottleneck: () => frameOn(bPos.clone(), Math.max(bn.diameterMm*9, 32), 900),
  nodule:     () => {
    const sel = P.nodules.find(n => n.selected) || P.nodules[0];
    if(sel) frameOn(new THREE.Vector3(...sel.position), Math.max(sel.drawRadiusMm*9, 38), 900);
  },
  inside:     () => { enterInside(); cursor = 0; updateFlyCursor(0); },
};
['Overview','Front','Route','Bottleneck','Nodule'].forEach(k => {
  const el = document.getElementById('v'+k);
  if(el) el.addEventListener('click', () => views[k.toLowerCase()]());
});

// ---------------------------------------------------------------- 腔内视角与飞行
let insideMode = false, flying = false, cursor = 0;
const statsEl = document.getElementById('stats'), bottomEl = document.getElementById('bottom');
const scrubEl = document.getElementById('scrub'), flyBtn = document.getElementById('fly');

// 腔内视角的图层覆盖。
// 相机就停在路径中心线上，如果把「器械管」和「管腔管」按原样画出来，
// 相机就正好在管子内部，整个画面会被管子的内壁糊死 ——
// 真实支气管镜里本来也看不到这两样东西，看到的只有气道壁。
// 所以进入腔内时强制隐藏它们，退出时再按复选框的状态恢复。
const INSIDE_OVERRIDES = [
  [() => device, false], [() => deviceHit, false],
  [() => lumen, false],
  [() => centerline, false], [() => junctions, false],
  [() => grid, false],
  [() => entry, false], [() => bGroup, true],   // 最窄处环保留：它正好套在视野前方
];

function applyInsideOverrides(){
  INSIDE_OVERRIDES.forEach(([get, want]) => { get().visible = want; });
}
function restoreLayerVisibility(){
  toggles.forEach(([id, fn]) => {
    const el = document.getElementById(id);
    if(el) fn(el.checked);
  });
}

function enterInside(){
  insideMode = true;
  document.getElementById('vInside').classList.add('on');
  bottomEl.classList.remove('hide'); statsEl.classList.remove('hide');
  scrubEl.value = 0;
  applyInsideOverrides();
  refreshAirwayMaterial();
}
function exitInside(){
  insideMode = false;
  document.getElementById('vInside').classList.remove('on');
  bottomEl.classList.add('hide'); statsEl.classList.add('hide');
  stopFly();
  restoreLayerVisibility();
  refreshAirwayMaterial();
}
document.getElementById('vInside').addEventListener('click', () => { enterInside(); updateFlyCursor(0); });
document.getElementById('vExit').addEventListener('click', exitInside);

function setFly(on){
  flying = on;
  flyBtn.classList.toggle('on', on);
  flyBtn.textContent = on ? '⏸ 暂停' : '▶ 沿路径飞行';
}
function stopFly(){ setFly(false); }
flyBtn.addEventListener('click', () => {
  if(!insideMode) enterInside();
  const next = !flying;
  if(!next){ setFly(false); return; }
  if(cursor >= 1) cursor = 0;
  setFly(true);
});
scrubEl.addEventListener('input', e => {
  setFly(false);
  if(!insideMode) enterInside();
  cursor = +e.target.value/1000;
  updateFlyCursor(cursor);
});

function posAt(u){
  const f = Math.min(Math.max(u,0),1)*(rPts.length-1);
  const i = Math.floor(f), t = f-i;
  const a = new THREE.Vector3(...rPts[Math.min(i, rPts.length-1)]);
  const b = new THREE.Vector3(...rPts[Math.min(i+1, rPts.length-1)]);
  return a.lerp(b, t);
}
function updateFlyCursor(u){
  const p = posAt(u), look = posAt(Math.min(u+0.05, 1));
  camera.position.copy(p);
  target.copy(look);
  camera.lookAt(target);
  const i = Math.round(u*(rPts.length-1));
  statsEl.innerHTML =
    '航点 <b>' + (i+1) + '</b>/' + rPts.length + ' &nbsp;·&nbsp; 累计 <b>' +
    P.route.cumulativeMm[i] + '</b> mm &nbsp;·&nbsp; 管腔直径 <b>' +
    (P.route.lumenRadiiMm[i]*2).toFixed(2) + '</b> mm &nbsp;·&nbsp; 器械余量 <b>' +
    clr[i].toFixed(2) + '</b> mm &nbsp;·&nbsp; 转角 <b>' +
    P.route.turnAnglesDeg[i].toFixed(1) + '</b>°';
}

// ---------------------------------------------------------------- 图层
const toggles = [
  ['tgAirway',     o => { airwayGroup.visible = o; }],
  ['tgCenterline', o => centerline.visible = o],
  ['tgJunction',   o => junctions.visible = o],
  ['tgLumen',      o => lumen.visible = o],
  ['tgDevice',     o => { device.visible = o; deviceHit.visible = o; }],
  ['tgNodules',    o => noduleGroup.visible = o],
  ['tgRepaired',   o => { if(repairedGroup) repairedGroup.visible = o; }],
  ['tgGrid',       o => grid.visible = o],
  ['tgMarkers',    o => { bGroup.visible = o; entry.visible = o; }],
];
toggles.forEach(([id, fn]) => {
  const el = document.getElementById(id);
  if(el) el.addEventListener('change', () => fn(el.checked));
});
if(!repairedKeys.length){
  const el = document.getElementById('tgRepaired');
  const row = el && el.closest('.toggle');
  if(row){ el.checked = false; row.style.opacity = .4; row.style.pointerEvents = 'none'; }
}

// ---------------------------------------------------------------- 剖切
const clipEl = document.getElementById('clip');
clipEl.addEventListener('input', e => {
  const v = +e.target.value/100;
  airwayUniforms.uClipY.value = v <= 0.001
    ? 1e9
    : bmax.y - (bmax.y - bmin.y)*v;
  document.getElementById('clipV').textContent = v <= 0.001 ? '关' : Math.round(v*100)+'%';
});

// ---------------------------------------------------------------- 自动旋转
let autoRotate = false;
const autoBtn = document.getElementById('auto');
function syncAutoBtn(){ autoBtn.classList.toggle('on', autoRotate); }
autoBtn.addEventListener('click', () => { autoRotate = !autoRotate; syncAutoBtn(); });

// ---------------------------------------------------------------- 自动演示
const toastEl = document.getElementById('toast');
let demoOn = false;
function toast(msg){
  toastEl.textContent = msg;
  toastEl.style.opacity = 1;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { toastEl.style.opacity = 0; }, 2200);
}
function runDemo(){
  demoOn = true;
  autoRotate = false; syncAutoBtn();
  exitInside();
  const steps = [
    ['整体气道与路径',       () => views.overview(),   2600],
    ['路径局部：器械余量着色', () => views.route(),      2400],
    ['最窄处',              () => views.bottleneck(), 2400],
    ['结节与靶点',           () => views.nodule(),     2400],
    ['腔内视角：沿路径飞行',  () => { enterInside(); cursor = 0; setFly(true); }, 0],
  ];
  let i = 0;
  (function next(){
    if(!demoOn) return;
    if(i >= steps.length){ demoOn = false; return; }
    const [label, fn, wait] = steps[i++];
    toast(label);
    fn();
    if(wait > 0) setTimeout(next, wait);
  })();
}
document.getElementById('demo').addEventListener('click', () => {
  if(demoOn){ demoOn = false; exitInside(); toast('已停止演示'); return; }
  runDemo();
});

// ---------------------------------------------------------------- 主题
document.getElementById('theme').addEventListener('click', () => {
  darkMode = document.body.classList.toggle('dark');
  document.getElementById('theme').textContent = darkMode ? '浅色' : '深色';
  refreshAirwayMaterial();
  centerline.material.color.set(darkMode ? 0x5b6f87 : 0x8496ab);
  junctions.material.color.set(darkMode ? 0x8098b4 : 0x6b7f96);
  lumen.material.color.set(darkMode ? 0x44536a : 0xcfdae8);
  grid.material.opacity = darkMode ? 0.2 : 0.42;
});

document.getElementById('reset').addEventListener('click', () => {
  demoOn = false;
  exitInside();
  autoRotate = false; syncAutoBtn();
  scrubEl.value = 0;
  flyTo(overviewPos, center, 800);
});

// ---------------------------------------------------------------- 主循环
let last = performance.now();
function animate(now){
  const dt = Math.min(0.05, (now-last)/1000); last = now;

  if(autoRotate && !flying){ theta += dt*0.16; applyOrbit(); }

  if(flying){
    cursor += dt*0.052;
    if(cursor >= 1){
      cursor = 1;
      updateFlyCursor(1);
      setFly(false);
      if(demoOn) setTimeout(() => { demoOn = false; exitInside(); toast('演示结束'); }, 1400);
    } else {
      scrubEl.value = cursor*1000;
      updateFlyCursor(cursor);
    }
  }

  noduleGroup.children.forEach(o => { if(o.userData.spin) o.rotation.z += dt*0.45; });
  bGroup.scale.setScalar(1 + Math.sin(now*0.0045)*0.07);

  if(debugOn) writeDebug();

  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

// ---------------------------------------------------------------- 初始化
// 图层开关的默认状态写在 HTML 的 checked 属性里，但场景对象的 visible
// 默认为 true。必须在这里按复选框的实际状态刷一遍，否则会出现
// 「复选框没勾，东西却画出来了」的不一致。
restoreLayerVisibility();
applyOrbit();
syncAutoBtn();
refreshAirwayMaterial();
requestAnimationFrame(animate);

addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---------------------------------------------------------------- URL 状态
// 支持 #view=route / #theme=dark / #fly=1 直接把 viewer 打开到某个状态。
// 用途：批量截图核对渲染、给演示视频定位到指定镜头、给别人发一个「直达链接」。
let debugOn = false;
(function applyHash(){
  const h = new URLSearchParams(location.hash.replace(/^#/, ''));
  if(h.get('instant') === '1') instantCamera = true;
  if(h.get('theme') === 'dark' && !document.body.classList.contains('dark')){
    document.getElementById('theme').click();
  }
  const view = h.get('view');
  if(view && views[view]){
    if(view === 'inside'){
      // 腔内视角不播动画，直接定位到指定进度，方便截图
      const t = Math.min(Math.max(parseFloat(h.get('t') || '0'), 0), 1);
      enterInside();
      cursor = t;
      scrubEl.value = t*1000;
      updateFlyCursor(t);
      if(h.get('fly') === '1') setFly(true);
    } else {
      views[view]();
    }
  }
  if(h.get('autorotate') === '1'){ autoRotate = true; syncAutoBtn(); }
  debugOn = h.get('debug') === '1';
})();

// 排障读数：在主循环里持续更新，配合无头浏览器的 --dump-dom 读取终态
function writeDebug(){
  const el = document.getElementById('debug');
  if(!el) return;
  const f = v => v.toFixed(2);
  el.textContent = 'DEBUG cam=' + camera.position.toArray().map(f).join(',') +
    ' target=' + target.toArray().map(f).join(',') +
    ' dist=' + f(camera.position.distanceTo(target)) +
    ' abRadius=' + f(abRadius) + ' radius=' + f(radius) +
    ' inside=' + insideMode + ' flying=' + flying + ' dark=' + darkMode +
    ' cursor=' + f(cursor);
}

// ---------------------------------------------------------------- 侧栏填充
(function fill(){
  const m = P.metrics || {};
  const num = (v,d) => (v===null||v===undefined||isNaN(v)) ? '—' : Number(v).toFixed(d);
  const req = P.meta.deviceDiameterMm + P.meta.deviceMarginMm*2;

  const rows = [
    ['路径总长',   num(m.route_length_mm,1)+' mm', ''],
    ['最窄处直径', num(m.minimum_diameter_mm,2)+' mm', m.minimum_diameter_mm >= req ? 'good' : 'bad'],
    ['最窄处余量', num(m.minimum_clearance_mm,2)+' mm',
        m.minimum_clearance_mm >= 1 ? 'good' : (m.minimum_clearance_mm >= 0 ? 'warn' : 'bad')],
    ['最大转角',   num(m.maximum_turn_angle_deg,1)+'°', m.maximum_turn_angle_deg > 90 ? 'warn' : ''],
    ['器械可通过', m.device_passable ? '是' : '否', m.device_passable ? 'good' : 'bad'],
    ['靶点距离',   num(m.target_distance_mm,2)+' mm', ''],
    ['航点数',     m.waypoint_count != null ? m.waypoint_count : '—', ''],
    ['器械外径',   num(P.meta.deviceDiameterMm,1)+' mm', ''],
  ];
  document.getElementById('metrics').innerHTML = rows.map(r =>
    '<div class="kv"><span class="k">'+r[0]+'</span><span class="v '+r[2]+'">'+r[1]+'</span></div>'
  ).join('');

  document.getElementById('topo').textContent = P.meta.topologySignature || '—';
  document.getElementById('hTitle').textContent = P.meta.caseId + ' · 候选 ' + P.meta.candidateId;
  document.getElementById('hSub').innerHTML =
    P.meta.profileTitle + ' <span class="badge">服务端编号 ' + (P.meta.serverCandidateId ?? '—') + '</span>';
  document.getElementById('hMeta').innerHTML =
    '器械 ' + P.meta.deviceDiameterMm + ' mm · 安全余量要求 ' + P.meta.deviceMarginMm + ' mm · ' + P.meta.entryMode + '<br>' +
    '体素 ' + P.meta.spacingZyx.map(v => v.toFixed(2)).join('×') + ' mm · CT ' + P.meta.imageSize.join('×') + '<br>' +
    '气道 ' + P.airwayStats.triangleCount + ' 三角面 · 中心线 ' + P.centerline.nodeCount +
      ' 节点 / ' + P.centerline.edgeCount + ' 边 · 分叉 ' + P.centerline.junctionCount + '<br>' +
    '<span style="color:var(--ink-faint)">' + P.meta.engine + ' · ' + P.meta.generatedAt + '</span>';

  document.getElementById('bneckInfo').innerHTML =
    '直径 <b>' + bn.diameterMm + '</b> mm · 余量 <b>' + bn.clearanceMm + '</b> mm<br>' +
    '位于路径 ' + bn.cumulativeMm + ' mm 处 · 航点 ' + (bn.index+1) + '<br>' +
    '坐标 ' + bn.position.map(v => v.toFixed(1)).join(', ') + ' mm';

  const rc = P.reachability || {};
  const gm = {adjacent:['紧邻气道','good'], reachable:['邻近可达','warn'],
              marginal:['明显偏离','bad'], unreachable:['远不可达','bad']};
  const g = gm[rc.grade] || ['—',''];
  document.getElementById('reach').innerHTML =
    '<div class="kv"><span class="k">可达性分级</span><span class="v '+g[1]+'">'+g[0]+'</span></div>' +
    '<div class="kv"><span class="k">靶点距中心线</span><span class="v">' +
      (rc.distance_mm != null ? rc.distance_mm : '—') + ' mm</span></div>' +
    '<div class="hint" style="margin-top:4px">' + (rc.label || '') + '</div>';

  const w = P.warnings || [];
  document.getElementById('warnBlock').classList.toggle('hide', !w.length);
  document.getElementById('warns').innerHTML = w.map(t => '<li>' + t + '</li>').join('');

  document.getElementById('legendCounts').innerHTML =
    '气道 ' + P.airwayStats.triangleCount + ' 面 · 中心线 ' + P.centerline.nodeCount +
    ' 点 · 路径 ' + P.route.waypointCount + ' 航点 · 结节 ' + P.nodules.length + ' 个';
})();
"""

_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AirNav 三维导航视图 · __AIRNAV_CASE__</title>
<style>__AIRNAV_STYLE__</style>
</head>
<body>
<div id="stage"></div>

<aside class="panel" id="left">
  <div class="head">
    <h1 id="hTitle">—</h1>
    <div class="sub" id="hSub"></div>
  </div>

  <section>
    <h2>规划指标</h2>
    <div id="metrics"></div>
  </section>

  <section>
    <h2>可达性</h2>
    <div id="reach"></div>
  </section>

  <section>
    <h2>最窄处</h2>
    <div id="bneckInfo" class="hint" style="font-variant-numeric:tabular-nums"></div>
  </section>

  <section>
    <h2>分支序列</h2>
    <div id="topo" class="mono" style="word-break:break-all;line-height:1.7"></div>
  </section>

  <section>
    <h2>病例信息</h2>
    <div id="hMeta" class="hint" style="font-variant-numeric:tabular-nums"></div>
  </section>

  <section id="warnBlock" class="hide">
    <h2>注意事项</h2>
    <ul class="warn" id="warns"></ul>
  </section>
</aside>

<aside class="panel" id="right">
  <div class="head"><h1>图层与视角</h1></div>

  <section>
    <h2>图层</h2>
    <label class="toggle"><input type="checkbox" id="tgAirway" checked>
      <span class="swatch" style="background:#8fb0d4"></span>气道表面</label>
    <label class="toggle"><input type="checkbox" id="tgDevice" checked>
      <span class="swatch" style="background:linear-gradient(90deg,#c9403a,#d8b13a,#2f8f4e)"></span>器械路径</label>
    <label class="toggle"><input type="checkbox" id="tgLumen" checked>
      <span class="swatch" style="background:#cfdae8"></span>路径管腔</label>
    <label class="toggle"><input type="checkbox" id="tgNodules" checked>
      <span class="swatch" style="background:#d9762a"></span>结节候选</label>
    <label class="toggle"><input type="checkbox" id="tgCenterline">
      <span class="swatch" style="background:#8496ab"></span>中心线</label>
    <label class="toggle"><input type="checkbox" id="tgJunction">
      <span class="swatch" style="background:#6b7f96"></span>分叉点</label>
    <label class="toggle"><input type="checkbox" id="tgRepaired" checked>
      <span class="swatch" style="background:#d0892f"></span>分割修复段</label>
    <label class="toggle"><input type="checkbox" id="tgMarkers" checked>
      <span class="swatch" style="background:#d2413a"></span>最窄处 / 入口</label>
    <label class="toggle"><input type="checkbox" id="tgGrid" checked>
      <span class="swatch" style="background:#ccd7e4"></span>参考网格</label>
  </section>

  <section>
    <h2>器械余量</h2>
    <div class="ramp"></div>
    <div class="ramp-lab"><span>贴壁 / 不可过</span><span>宽松</span></div>
    <div class="hint" style="margin-top:7px">
      器械管按 <b>管腔半径 − 器械半径 − 安全余量</b> 逐点着色。
    </div>
  </section>

  <section>
    <h2>剖切</h2>
    <input type="range" id="clip" min="0" max="100" value="0">
    <div class="ramp-lab"><span>横断面剖切</span><span id="clipV">关</span></div>
  </section>

  <section>
    <h2>视角</h2>
    <div class="grid2">
      <button id="vOverview">全局</button>
      <button id="vFront">正位</button>
      <button id="vRoute">路径</button>
      <button id="vBottleneck">最窄处</button>
      <button id="vNodule">结节</button>
      <button id="vInside">腔内</button>
    </div>
    <div class="grid2" style="margin-top:6px">
      <button id="demo" class="primary">▶ 自动演示</button>
      <button id="auto">自动旋转</button>
    </div>
    <div class="grid2" style="margin-top:6px">
      <button id="theme">深色</button>
      <button id="reset">复位</button>
    </div>
    <div class="hint" style="margin-top:8px">
      拖动旋转 · 右键平移 · 滚轮缩放
    </div>
  </section>

  <section>
    <h2>构成</h2>
    <div class="hint" id="legendCounts"></div>
  </section>
</aside>

<div id="tip"></div>
<div id="toast"></div>
<div id="stats" class="hide"></div>
<div id="debug" style="position:absolute;left:0;bottom:0;font-size:9px;color:#888;opacity:.01"></div>

<div id="bottom" class="hide">
  <button id="fly" class="primary">▶ 沿路径飞行</button>
  <input type="range" id="scrub" min="0" max="1000" value="0">
  <span class="lbl">腔内视角</span>
  <button id="vExit">退出</button>
</div>

<script>__AIRNAV_THREE__</script>
<script>__AIRNAV_APP__</script>
</body>
</html>
"""


def three_source() -> str:
    """返回内联用的 Three.js 源码。"""
    if not _VENDOR.is_file():
        raise FileNotFoundError(
            f"缺少内联依赖 {_VENDOR}。请重新下载 three.min.js 后再生成 viewer。"
        )
    return _VENDOR.read_text(encoding="utf-8")


def encoding_chunk_name(three_src: str) -> str:
    """探测当前 Three.js 版本使用的输出色彩空间 shader chunk 名。

    r152 起改名为 colorspace_fragment，之前叫 encodings_fragment。
    名字写错的后果是着色器编译失败、对应物体整个消失 —— 所以这里探测而不是写死。
    """
    return "colorspace_fragment" if "colorspace_fragment" in three_src else "encodings_fragment"


def render_html(payload: dict) -> str:
    """把负载渲染成单文件 HTML。"""
    three = three_source()
    # </script> 会提前结束脚本块，必须转义
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    app = (
        _APP_JS.replace("__AIRNAV_PAYLOAD__", data)
        .replace("__AIRNAV_ENCODING_CHUNK__", encoding_chunk_name(three))
    )
    return (
        _HTML.replace("__AIRNAV_STYLE__", _STYLE)
        .replace("__AIRNAV_THREE__", three)
        .replace("__AIRNAV_APP__", app)
        .replace("__AIRNAV_CASE__", str(payload.get("meta", {}).get("caseId", "case")))
    )


__all__ = ["encoding_chunk_name", "render_html", "three_source"]
