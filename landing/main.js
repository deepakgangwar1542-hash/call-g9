(() => {
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  // Desk URL: ?desk=... overrides the <meta name="desk-url"> default.
  const deskUrl = new URLSearchParams(location.search).get("desk")
    || document.querySelector('meta[name="desk-url"]')?.content;
  if (deskUrl) document.querySelectorAll("[data-desk-link]").forEach((a) => { a.href = deskUrl; });

  // Staggered reveal on enter.
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
  }, { threshold: 0.15 });
  document.querySelectorAll(".reveal").forEach((el, i) => {
    el.style.setProperty("--i", el.closest(".hero-copy, .ev-grid") ? i % 5 : 0);
    io.observe(el);
  });

  if (reduced) { setTourStep(0); return; }

  // Pointer state shared by hero tilt and card tilt.
  const pointer = { x: 0, y: 0 }, eased = { x: 0, y: 0 };
  addEventListener("pointermove", (e) => {
    pointer.x = (e.clientX / innerWidth) * 2 - 1;
    pointer.y = (e.clientY / innerHeight) * 2 - 1;
  }, { passive: true });

  const heroWindow = document.getElementById("hero-window");
  const depthLayers = [...document.querySelectorAll("[data-depth]")];
  const tour = document.getElementById("tour");
  const tourWindow = document.getElementById("tour-window");
  const tourBar = document.getElementById("tour-bar");
  const steps = [...document.querySelectorAll(".steps li")];

  // Camera pose per tour step — the window swings like the video's dolly shots.
  const poses = [
    { rx: 14, ry: -18, rz: 1, s: 0.96 },
    { rx: 32, ry: 8, rz: -2, s: 1.0 },
    { rx: 8, ry: 20, rz: 1, s: 0.97 },
    { rx: 18, ry: -6, rz: 0, s: 1.02 },
    { rx: 10, ry: 14, rz: -1, s: 0.98 },
  ];
  const urls = ["dispatch", "pulse", "control-room", "capacity", "shift-review"];
  let currentStep = -1;

  function setTourStep(i) {
    if (i === currentStep) return;
    currentStep = i;
    steps.forEach((li, n) => li.classList.toggle("on", n === i));
    document.querySelectorAll(".screen").forEach((s) => s.classList.toggle("is-active", +s.dataset.screen === i));
    document.querySelectorAll("#tour-nav > span").forEach((n, k) => n.classList.toggle("on", k === i));
    document.getElementById("tour-url").textContent = `callsev.triage / ${urls[i]}`;
  }

  let tourProgress = 0, tourStepProgress = 0;
  function readScroll() {
    const y = scrollY;
    depthLayers.forEach((el) => { el.dataset.y = String(y * parseFloat(el.dataset.depth)); });

    const r = tour.getBoundingClientRect();
    const span = r.height - innerHeight;
    tourProgress = clamp(-r.top / span, 0, 1);
    const raw = tourProgress * poses.length;
    setTourStep(clamp(Math.floor(raw), 0, poses.length - 1));
    tourStepProgress = raw - Math.floor(raw);
    tourBar.style.width = `${tourProgress * 100}%`;
  }
  addEventListener("scroll", readScroll, { passive: true });
  addEventListener("resize", readScroll);
  readScroll();

  const tiltCards = [...document.querySelectorAll("[data-tilt]")];
  tiltCards.forEach((card) => {
    card.addEventListener("pointermove", (e) => {
      const b = card.getBoundingClientRect();
      const px = (e.clientX - b.left) / b.width - 0.5;
      const py = (e.clientY - b.top) / b.height - 0.5;
      card.style.transform = `rotateX(${-py * 10}deg) rotateY(${px * 12}deg) translateZ(10px)`;
    });
    card.addEventListener("pointerleave", () => { card.style.transform = ""; });
  });

  function frame() {
    eased.x = lerp(eased.x, pointer.x, 0.06);
    eased.y = lerp(eased.y, pointer.y, 0.06);
    const y = scrollY;

    // Hero window: flattens as you scroll away, tilts toward the pointer.
    const heroT = clamp(y / innerHeight, 0, 1);
    const rx = lerp(20, 4, heroT) - eased.y * 6;
    const ry = -10 + eased.x * 10;
    heroWindow.style.setProperty("--rx", `${rx}deg`);
    heroWindow.style.setProperty("--ry", `${ry}deg`);
    heroWindow.parentElement.style.translate = `0 ${-y * 0.12}px`;

    // Parallax layers: scroll depth + pointer drift.
    depthLayers.forEach((el) => {
      const d = parseFloat(el.dataset.depth);
      const sy = parseFloat(el.dataset.y || "0");
      el.style.translate = `${eased.x * d * 30}px ${sy + eased.y * d * 20}px`;
    });

    // Tour window: interpolate toward the next pose for a continuous camera move.
    const a = poses[currentStep], b = poses[Math.min(currentStep + 1, poses.length - 1)];
    const t = clamp((tourStepProgress - 0.55) / 0.45, 0, 1) * 0.35;
    const p = { rx: lerp(a.rx, b.rx, t), ry: lerp(a.ry, b.ry, t), rz: lerp(a.rz, b.rz, t), s: lerp(a.s, b.s, t) };
    tourWindow.style.transform =
      `rotateX(${p.rx - eased.y * 4}deg) rotateY(${p.ry + eased.x * 6}deg) rotateZ(${p.rz}deg) scale(${p.s})`;

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
