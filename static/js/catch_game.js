/* Catch it yourself: a real recorded throw from the robot's head camera, with the robot's arm and catcher plate drawn
   over it. Grabbing the plate starts a throw, dragging moves it; when the clip ends the arm goes back to its ready pose
   (user 2026-11-24). A pulsing ring on the plate, "Drag the plate to play", shows the first time the game comes into
   view and goes away at the first grab.
   Data: catch_game_data.js (figures/supp_videos/catch_game: prep.py, make_clips.py, export_js.py).
   - Camera: the session's calibrated head camera (intrinsics and pose), so the drawn arm stands where the real one did.
   - Ball: the throw's fitted flight (gravity + drag) from its release; the clip has the real ball painted out from the
     release on, and the page draws it at the same place.
   - Arm: the pointer's pixel picks joint targets from a precomputed reach table (plate face on that pixel's viewing
     ray, inside the deployed policy's joint box); the joints follow the targets like the training simulator's arm
     (critically damped, 8 Hz, motor speed caps).
   - Catch: the ball crosses the plate face within the plate's radius (the training simulator's contact rule), then
     stays on the plate. A simplified browser demo, not the paper's simulator. */
(function () {
  const D = window.CATCH_GAME, T3 = window.THREE;
  const root = document.getElementById('catch-game');
  if (!D || !T3 || !root) return;
  const stage = root.querySelector('.cg-stage'), video = root.querySelector('video'), canvas = root.querySelector('canvas');
  const msg = root.querySelector('.cg-msg'), score = root.querySelector('.cg-score');
  const [FX, FY, CX, CY] = D.K, W = D.W, H = D.H, DT = 0.002;

  /* ---- renderer and the head camera ---- */
  let renderer;
  try { renderer = new T3.WebGLRenderer({ canvas, alpha: true, antialias: true }); }
  catch (e) { msg.textContent = 'This browser cannot show the 3D view (WebGL is off).'; return; }
  renderer.setClearColor(0x000000, 0);
  const scene = new T3.Scene();
  scene.add(new T3.HemisphereLight(0xffffff, 0x666666, 0.9));
  const sun = new T3.DirectionalLight(0xffffff, 0.8); sun.position.set(-1, 0.3, 2); scene.add(sun);
  const camera = new T3.Camera();
  const near = 0.05, far = 30;
  camera.projectionMatrix.set(
    2 * FX / W, 0, 1 - 2 * CX / W, 0,
    0, 2 * FY / H, 2 * CY / H - 1, 0,
    0, 0, -(far + near) / (far - near), -2 * far * near / (far - near),
    0, 0, -1, 0);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  camera.matrixAutoUpdate = false;
  function setCamera(s) {   // OpenCV axes (x right, y down, z forward) in the robot base frame -> GL camera
    const { c, R } = D.cams[s];
    const x = new T3.Vector3(R[0][0], R[1][0], R[2][0]), y = new T3.Vector3(-R[0][1], -R[1][1], -R[2][1]),
          z = new T3.Vector3(-R[0][2], -R[1][2], -R[2][2]);
    camera.matrix.makeBasis(x, y, z).setPosition(c[0], c[1], c[2]);
    camera.updateMatrixWorld(true);
  }

  /* ---- the arm: kinematic tree + meshes ---- */
  const V3 = T3.Vector3, Q4 = T3.Quaternion;
  const mq = a => new Q4(a[1], a[2], a[3], a[0]);          // MuJoCo (w, x, y, z) -> three.js (x, y, z, w)
  const B = D.bodies.map(b => ({ ...b, P: new V3(...b.pos), Qb: mq(b.quat), A: new V3(...b.axis), J: new V3(...b.jpos) }));
  function fk(q) {
    const X = [];
    for (const b of B) {
      const par = b.parent < 0 ? { p: new V3(), r: new Q4() } : X[b.parent];
      const p = par.p.clone().add(b.P.clone().applyQuaternion(par.r)), r = par.r.clone().multiply(b.Qb);
      if (b.q >= 0) {
        const jq = new Q4().setFromAxisAngle(b.A, q[b.q]);
        p.add(b.J.clone().sub(b.J.clone().applyQuaternion(jq)).applyQuaternion(r));
        r.multiply(jq);
      }
      X.push({ p, r });
    }
    return X;
  }
  const meshes = D.geoms.map(g => {
    const geo = new T3.BufferGeometry();
    geo.setAttribute('position', new T3.Float32BufferAttribute(g.v.map(x => x * 1e-4), 3));
    geo.setIndex(g.f);
    geo.computeVertexNormals();
    const m = new T3.Mesh(geo, new T3.MeshLambertMaterial({ color: new T3.Color(...g.rgb) }));
    m.matrixAutoUpdate = false; scene.add(m);
    return { m, body: g.body, P: new V3(...g.pos), Qg: mq(g.quat) };
  });
  const tmpM = new T3.Matrix4(), one = new V3(1, 1, 1);
  function poseArm(X) {
    for (const g of meshes) {
      const b = X[g.body];
      g.m.matrix.compose(b.p.clone().add(g.P.clone().applyQuaternion(b.r)), b.r.clone().multiply(g.Qg), one);
      g.m.matrixWorldNeedsUpdate = true;
    }
  }

  /* ---- the ball: two-tone felt ball ---- */
  const bgeo = new T3.SphereGeometry(D.ball_r, 24, 16), cols = [];
  const pos = bgeo.getAttribute('position');
  for (let i = 0; i < pos.count; i++) {
    const c = pos.getY(i) > 0 ? [0.98, 0.35, 0.6] : [0.72, 0.9, 0.2];
    cols.push(...c);
  }
  bgeo.setAttribute('color', new T3.Float32BufferAttribute(cols, 3));
  const ball = new T3.Mesh(bgeo, new T3.MeshLambertMaterial({ vertexColors: true }));
  ball.visible = false; ball.renderOrder = 1; scene.add(ball);
  // the head camera sees the back of the plate, so a caught ball (on the face) is drawn over it to stay visible
  const showStuck = on => { ball.material.depthTest = !on; };

  /* ---- flight: RK4, gravity + quadratic drag, sampled every DT from the release ---- */
  const G = new V3(...D.g);
  function flight(T) {
    const acc = v => G.clone().sub(v.clone().multiplyScalar(D.k_drag * v.length()));
    function step(p, v, h) {
      const a1 = acc(v), v2 = v.clone().addScaledVector(a1, h / 2), a2 = acc(v2), v3 = v.clone().addScaledVector(a2, h / 2),
            a3 = acc(v3), v4 = v.clone().addScaledVector(a3, h), a4 = acc(v4);
      p.addScaledVector(v.clone().add(v2.clone().multiplyScalar(2)).add(v3.clone().multiplyScalar(2)).add(v4), h / 6);
      v.addScaledVector(a1.clone().add(a2.clone().multiplyScalar(2)).add(a3.clone().multiplyScalar(2)).add(a4), h / 6);
    }
    const p = new V3(...T.p0), v = new V3(...T.v0);
    for (let n = Math.round(-T.t_rel / DT); n > 0; n--) step(p, v, -DT);    // back to the release
    const out = [];
    for (let i = 0; i <= Math.round(1.6 / DT); i++) { out.push(p.clone()); step(p, v, DT); }
    return out;
  }

  /* ---- reach table: pointer pixel -> joint targets (bilinear) ---- */
  function targetsAt(s, u, v) {
    const g = D.grid, Q = g.q[s];
    const a = Math.min(Math.max((u - g.u0) / g.du, 0), g.nu - 1.001), b = Math.min(Math.max((v - g.v0) / g.dv, 0), g.nv - 1.001);
    const a0 = Math.floor(a), b0 = Math.floor(b), fa = a - a0, fb = b - b0, out = [];
    for (let j = 0; j < 6; j++) {
      const at = (i, k) => Q[(i * g.nv + k) * 6 + j];
      out.push((1 - fa) * ((1 - fb) * at(a0, b0) + fb * at(a0, b0 + 1)) + fa * ((1 - fb) * at(a0 + 1, b0) + fb * at(a0 + 1, b0 + 1)));
    }
    return out;
  }

  /* ---- game state ---- */
  const W0 = 2 * Math.PI * D.pd_hz;
  let q = D.q_ready.slice(), qd = [0, 0, 0, 0, 0, 0], qt = D.q_ready.slice();
  let cur = null, path = null, tSim = 0, live = false, stuck = null, prevRel = null, result = null, dragging = false;
  let grabId = null, grabOff = [0, 0], lastX = null;
  let caught = 0, played = 0;
  const order = D.throws.map((_, i) => i).sort(() => Math.random() - 0.5);
  let next = 0;

  function plateFrame(X) { const b = X[D.plate_body]; return { p: b.p, r: b.r, ri: b.r.clone().invert() }; }

  function stepArm() {
    for (let j = 0; j < 6; j++) {
      const a = W0 * W0 * (qt[j] - q[j]) - 2 * W0 * qd[j];
      qd[j] = Math.max(-D.vmax[j], Math.min(D.vmax[j], qd[j] + a * DT));
      q[j] += qd[j] * DT;
    }
  }

  function ballAt(t) {      // t: time since the fit's first flight point
    const x = (t - cur.t_rel) / DT;
    if (x < 0 || x >= path.length - 1) return null;
    const i = Math.floor(x), f = x - i;
    return path[i].clone().lerp(path[i + 1], f);
  }

  function physics(t1) {
    let X = fk(q);
    if (tSim < t1 - 0.1) tSim = t1 - 0.1;            // after a stall, do not replay a long backlog
    while (tSim + DT <= t1) {
      tSim += DT;
      stepArm();
      X = fk(q);
      const pf = plateFrame(X);
      if (stuck) continue;
      const b = ballAt(tSim);
      if (!b) { prevRel = null; continue; }
      const rel = b.clone().sub(pf.p).applyQuaternion(pf.ri);
      if (prevRel && prevRel.z > D.ball_r && rel.z <= D.ball_r && result === null) {
        const f = (prevRel.z - D.ball_r) / Math.max(prevRel.z - rel.z, 1e-6);
        const x = prevRel.x + f * (rel.x - prevRel.x), y = prevRel.y + f * (rel.y - prevRel.y);
        if (Math.hypot(x, y) <= D.plate_r) { stuck = new V3(x, y, D.ball_r); showStuck(true); finish(true); }
      }
      prevRel = rel;
    }
    return X;
  }

  function finish(ok) {
    if (result !== null) return;
    result = ok; played++; if (ok) caught++;
    score.textContent = `Caught ${caught} of ${played}`;
    msg.textContent = ok ? 'Caught!' : 'Missed.';
    msg.className = 'cg-msg show ' + (ok ? 'ok' : 'miss');
  }

  function load(i) {
    cur = D.throws[i]; path = flight(cur);
    setCamera(String(cur.session));
    video.src = cur.clip; video.load();
  }

  function start() {
    if (live) return;
    stuck = null; prevRel = null; result = null; live = true; tSim = cur.t_start; showStuck(false);
    msg.className = 'cg-msg'; msg.textContent = '';
    video.currentTime = 0;
    video.play().catch(() => {});
  }

  function release() {
    if (grabId !== null) { try { stage.releasePointerCapture(grabId); } catch (_) {} }
    dragging = false; grabId = null; stage.style.cursor = '';
  }

  video.muted = true; video.playsInline = true; video.preload = 'auto';
  video.addEventListener('ended', () => {
    if (!live) return;
    live = false;
    if (result === null) finish(false);
    msg.textContent += ' Drag the plate for the next throw.';
    release(); qt = D.q_ready.slice(); stuck = null; showStuck(false);    // the arm goes back to its ready pose
    next = (next + 1) % order.length; load(order[next]);
  });

  /* ---- the plate on screen, in video pixels: centre and radius ---- */
  function platePx() {
    const pf = plateFrame(lastX || fk(q)), pr = v => { const s = v.clone().project(camera); return [(s.x + 1) / 2 * W, (1 - s.y) / 2 * H]; };
    const c = pr(pf.p);
    let rad = 0;
    for (const ax of [new V3(1, 0, 0), new V3(0, 1, 0)]) {
      const e = pr(ax.applyQuaternion(pf.r).multiplyScalar(D.plate_r).add(pf.p));
      rad = Math.max(rad, Math.hypot(e[0] - c[0], e[1] - c[1]));
    }
    return { c, rad };
  }
  function toPx(e) { const r = stage.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * W, (e.clientY - r.top) / r.height * H]; }
  function onPlate(e) {      // generous: the arm hides part of the plate, and fingers are wide
    const { c, rad } = platePx(), p = toPx(e);
    return Math.hypot(p[0] - c[0], p[1] - c[1]) <= rad * (e.pointerType === 'touch' ? 1.7 : 1.35) + 6;
  }

  /* ---- first visit: a pulsing ring on the plate, "Drag the plate to play" ---- */
  const hint = document.createElement('div');
  hint.className = 'cg-hint';
  hint.innerHTML = '<span class="cg-ring"></span><span class="cg-hint-label">Drag the plate to play</span>';
  stage.appendChild(hint);
  let hintState = 0;       // 0 not seen yet, 1 showing, 2 gone (the plate was grabbed)
  const showHint = () => { if (hintState === 0) { hintState = 1; hint.classList.add('show'); } };
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(es => { if (es.some(x => x.isIntersecting)) { showHint(); io.disconnect(); } }, { threshold: 0.6 });
    io.observe(stage);
  } else showHint();
  function placeHint() {
    const { c, rad } = platePx();
    hint.style.left = (c[0] / W * 100) + '%'; hint.style.top = (c[1] / H * 100) + '%';
    hint.style.setProperty('--d', (2.3 * rad / W * stage.getBoundingClientRect().width) + 'px');
  }

  /* ---- pointer: grab the plate to start a throw, drag to move it ---- */
  function aim(e) {
    const p = toPx(e);
    qt = targetsAt(String(cur.session), p[0] + grabOff[0], p[1] + grabOff[1]);
  }
  stage.addEventListener('pointerdown', e => {
    if (!onPlate(e)) return;
    e.preventDefault(); dragging = true; grabId = e.pointerId; stage.style.cursor = 'grabbing';
    try { stage.setPointerCapture(e.pointerId); } catch (_) {}
    const { c } = platePx(), p = toPx(e);
    grabOff = [c[0] - p[0], c[1] - p[1]];   // the plate keeps its place under the pointer; it moves only when dragged
    if (hintState === 1) { hintState = 2; hint.classList.remove('show'); }
    start();
  });
  stage.addEventListener('pointermove', e => {
    if (dragging) aim(e);
    else if (e.pointerType === 'mouse') stage.style.cursor = onPlate(e) ? 'grab' : '';
  });
  stage.addEventListener('pointerup', release); stage.addEventListener('pointercancel', release);

  /* ---- frame loop: physics on the video's clock, then draw ---- */
  function resize() {
    const r = stage.getBoundingClientRect();
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setSize(Math.max(1, Math.round(r.width)), Math.max(1, Math.round(r.height)), false);
  }
  window.addEventListener('resize', resize);
  function frame() {
    let X;
    if (live) {
      X = physics(cur.t_start + video.currentTime);
      if (result === null && tSim > cur.t_rel + 1.6 - 0.05) finish(false);
    } else {
      X = physics(tSim + 1 / 60);                       // between throws the arm still settles on its target
    }
    poseArm(X); lastX = X;
    if (hintState === 1) placeHint();
    const pf = plateFrame(X);
    if (stuck) { ball.position.copy(stuck.clone().applyQuaternion(pf.r).add(pf.p)); ball.visible = true; }
    else { const b = live ? ballAt(tSim) : null; ball.visible = !!b; if (b) ball.position.copy(b); }
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  load(order[0]); resize();
  requestAnimationFrame(frame);
  // test hooks (headless checks only)
  window.__catchGame = { fk, targetsAt, platePx, state: () => ({ q, qt, tSim, live, result, caught, played, cur: cur && cur.ep, dragging, hintState }),
                         project: (p) => { const v = new V3(...p).project(camera); return [(v.x + 1) / 2 * W, (1 - v.y) / 2 * H]; } };
})();
