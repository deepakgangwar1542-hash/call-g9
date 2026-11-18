(() => {
  const root = document.documentElement,
    nav = document.querySelector(".nav"),
    menu = document.querySelector(".menu"),
    links = document.querySelector("nav"),
    toggle = document.querySelector(".theme");
  const saved = localStorage.getItem("triage-theme");
  if (saved) root.dataset.theme = saved;
  toggle.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
    localStorage.setItem("triage-theme", root.dataset.theme);
  });
  menu.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    menu.setAttribute("aria-expanded", open);
    menu.setAttribute(
      "aria-label",
      open ? "Close navigation" : "Open navigation",
    );
  });
  links.addEventListener("click", (e) => {
    if (e.target.tagName === "A") links.classList.remove("open");
  });
  addEventListener(
    "scroll",
    () => nav.classList.toggle("scrolled", scrollY > 24),
    { passive: true },
  );
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduce && "IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (es) =>
        es.forEach((e, i) => {
          if (e.isIntersecting) {
            e.target.classList.add("show");
            io.unobserve(e.target);
          }
        }),
      { threshold: 0.12 },
    );
    document.querySelectorAll(".reveal").forEach((e) => io.observe(e));
    const chart = document.querySelector(".chart"),
      timeline = document.querySelector(".timeline");
    new IntersectionObserver(
      (es) =>
        es.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add(e.target === chart ? "animate" : "draw");
          }
        }),
      { threshold: 0.3 },
    ).observe(chart);
    new IntersectionObserver(
      (es) =>
        es.forEach((e) => {
          if (e.isIntersecting) e.target.classList.add("draw");
        }),
      { threshold: 0.3 },
    ).observe(timeline);
  } else
    document
      .querySelectorAll(".reveal")
      .forEach((e) => e.classList.add("show"));
  const form = document.querySelector(".demo"),
    output = document.querySelector(".demo-result");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    form.querySelectorAll(".field-error").forEach((x) => x.remove());
    let first;
    [...form.querySelectorAll("select")].forEach((s) => {
      if (!s.value) {
        const m = document.createElement("small");
        m.className = "field-error";
        m.textContent = `Select ${s.closest("label").firstChild.textContent.trim().toLowerCase()}.`;
        s.after(m);
        first ??= s;
      }
    });
    if (first) {
      first.focus();
      return;
    }
    const speed = +form.speed.value,
      dark = form.light.value.startsWith("Darkness");
    const score = Math.min(
      63,
      Math.round(
        10 +
          speed * 0.35 +
          (dark ? 13 : 0) +
          (form.weather.value.includes("Rain") ? 5 : 0),
      ),
    );
    const high = score >= 38;
    output.innerHTML = `<section class="verdict ${high ? "high" : "low"}"><div class="ring"><strong>${score}</strong><small>%</small></div><div><span class="verdict-tag ${high ? "high" : ""}">${high ? "HIGH RISK" : "LOW RISK"}</span><p>${high ? "Enhanced response may be appropriate." : "Standard dispatch protocol remains appropriate."}<br><small>Simulation result; simplified lookup only.</small></p></div></section>`;
  });
})();
