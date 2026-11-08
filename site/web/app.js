/* Resilience Rewired — single-page frontend.
   All financial figures come from the API (engine.py); this file only formats and draws them. */
(() => {
  "use strict";

  // ---------------------------------------------------------------- helpers
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const money = (x, signed = false) => {
    if (x == null || !isFinite(x)) return "—";
    const r = Math.round(Math.abs(x));
    const s = "$" + r.toLocaleString("en-US");
    if (x < 0 && r !== 0) return "−" + s;
    return (signed && x > 0 && r !== 0 ? "+" : "") + s;
  };
  const moneyK = (x) => {
    const a = Math.abs(x), sign = x < 0 ? "−" : "";
    if (a >= 1e6) return sign + "$" + (a / 1e6).toFixed(a >= 1e7 ? 0 : 1).replace(/\.0$/, "") + "M";
    if (a >= 1e3) return sign + "$" + (a / 1e3).toFixed(a >= 1e4 ? 0 : 1).replace(/\.0$/, "") + "k";
    return sign + "$" + Math.round(a);
  };
  const monthLabel = (m, short = false) => {
    const [y, mo] = m.split("-").map(Number);
    const d = new Date(Date.UTC(y, mo - 1, 1));
    return d.toLocaleString("en-US", { month: short ? "short" : "long", year: short ? "2-digit" : "numeric", timeZone: "UTC" })
      .replace(" ", short ? " ’" : " ");
  };
  const toast = (msg) => {
    const t = $("#toast"); t.textContent = msg; t.hidden = false;
    clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 2600);
  };
  // API origin: "" = same server. A team backend elsewhere sets window.RR_API_BASE (see index.html).
  const API_BASE = (window.RR_API_BASE || "").replace(/\/$/, "");
  const api = async (path, opts) => {
    const r = await fetch(API_BASE + path, opts);
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  };

  // number roll: interpolates the shown value, then lands on the exact engine text
  const ease = (t) => 1 - Math.pow(1 - t, 4);
  function roll(el, to, format, finalText) {
    if (to == null || !isFinite(to) || reduced) { el.textContent = finalText ?? format(to); el._v = to; return; }
    const from = el._v ?? to, start = performance.now(), dur = 420;
    cancelAnimationFrame(el._raf); clearTimeout(el._land);
    el._land = setTimeout(() => { cancelAnimationFrame(el._raf); el.textContent = finalText ?? format(to); el._v = to; }, dur + 60);
    if (from === to) { el.textContent = finalText ?? format(to); el._v = to; return; }
    const step = (now) => {
      const t = Math.min((now - start) / dur, 1), v = from + (to - from) * ease(t);
      el.textContent = t < 1 ? format(v) : (finalText ?? format(to));
      el._v = t < 1 ? v : to;
      if (t < 1) el._raf = requestAnimationFrame(step);
    };
    el._raf = requestAnimationFrame(step);
  }
  function swapText(el, text) {           // a status word that has no number to roll (No shortfall ↔ 120+)
    if (el.textContent === text) return;
    el.textContent = text; el._v = null;
    if (reduced) return;
    el.animate([{ opacity: 0, transform: "translateY(6px)", filter: "blur(3px)" }, { opacity: 1, transform: "none", filter: "none" }],
      { duration: 260, easing: "cubic-bezier(0.22, 1, 0.36, 1)" });
  }

  // ---------------------------------------------------------------- state
  const PRESETS = {
    none: { keep: 100, bills: 0, oneOff: 0, q: "auto", label: "" },
    ei55: { keep: 55, bills: 0, oneOff: 0, q: "last_months", label: "job loss (EI 55%)" },
    furnace: { keep: 100, bills: 0, oneOff: "S", q: "can_pay", label: "" },     // one month of this account's typical spending
    combined: { keep: 55, bills: 15, oneOff: 0, q: "last_months", label: "job loss (EI 55%)" },
  };
  const NO_EXTRA = { extraOut: 0, windfall: 0, startIn: 0, lasts: 0, afterPct: 100 };
  const S = {
    overview: null, ids: [], byId: new Map(), accounts: new Map(),
    acct: null, data: null, sim: null,
    preset: "ei55", shock: { keep: 55, bills: 0, oneOff: 0, ...NO_EXTRA }, question: "last_months", label: "job loss (EI 55%)",
    plans: new Map(), peerMode: "income", msgKey: null,
  };
  const planFor = (id) => {
    if (!S.plans.has(id)) S.plans.set(id, { target: 6, checked: false, cut: 0, extra: 0, kmon: 3, save: 0 });
    return S.plans.get(id);
  };

  // ---------------------------------------------------------------- type-ahead (landing + switcher)
  function typeahead(input, list, box) {
    let items = [], active = -1;
    const close = () => { list.hidden = true; box.setAttribute("aria-expanded", "false"); active = -1; input.removeAttribute("aria-activedescendant"); };
    const render = () => {
      const q = input.value.replace(/\D/g, "");
      if (!q) { close(); return; }
      items = S.ids.filter((id) => String(id).startsWith(q)).slice(0, 8);
      if (!items.length) {
        list.innerHTML = `<li class="sl-empty" role="option" aria-disabled="true">No account starts with ${esc(q)}</li>`;
      } else {
        list.innerHTML = items.map((id, i) => {
          const a = S.byId.get(id);
          const risk = a.status === "buffer_gone" || a.status === "shortfall";
          return `<li role="option" id="${list.id}-${i}" data-id="${id}" aria-selected="${i === active}">
            <span class="sl-id">#${id}</span><span class="sl-b">${money(a.B)}</span><span class="sl-r${risk ? " risk" : ""}">${esc(a.runway)}</span></li>`;
        }).join("");
      }
      list.hidden = false; box.setAttribute("aria-expanded", "true");
    };
    const move = (d) => {
      if (!items.length) return;
      active = (active + d + items.length) % items.length;
      $$("li", list).forEach((li, i) => li.setAttribute("aria-selected", String(i === active)));
      input.setAttribute("aria-activedescendant", `${list.id}-${active}`);
    };
    const commit = () => {
      const q = input.value.replace(/\D/g, "");
      if (!q) return;
      const id = active >= 0 ? items[active] : (S.byId.has(Number(q)) ? Number(q) : items[0]);
      if (id == null) { toast(`We couldn't find account ${q}.`); return; }
      close(); input.blur(); openAccount(id);
    };
    input.addEventListener("input", () => { active = -1; render(); });
    input.addEventListener("focus", render);
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (e.key === "Enter") { e.preventDefault(); commit(); }
      else if (e.key === "Escape") { close(); input.blur(); }
    });
    list.addEventListener("mousedown", (e) => {
      const li = e.target.closest("li[data-id]"); if (!li) return;
      e.preventDefault(); close(); input.value = ""; input.blur(); openAccount(Number(li.dataset.id));
    });
    input.addEventListener("blur", () => setTimeout(close, 120));
  }

  // ---------------------------------------------------------------- stage changes
  async function openAccount(id) {
    if (!S.byId.has(id)) { toast(`We couldn't find account ${id}.`); return; }
    let data = S.accounts.get(id);
    try {
      if (!data) { data = await api(`/api/account/${id}`); S.accounts.set(id, data); }
    } catch (err) { toast(err.message); return; }
    const first = S.acct == null || $("#main").hidden;
    S.acct = id; S.data = data;
    if (S.preset === "furnace") S.shock.oneOff = data.profile.S;
    if (location.hash !== `#/account/${id}`) history.pushState(null, "", `#/account/${id}`);
    $("#switch-input").value = "";
    $("#switch-input").placeholder = `#${id}`;
    if (first) await enterCenter();
    renderAccount();
    simulate();
  }

  async function enterCenter() {
    const landing = $("#landing"), main = $("#main");
    if (!landing.hidden && !reduced) {
      landing.classList.add("leaving");
      await new Promise((r) => setTimeout(r, 260));
    }
    landing.hidden = true; landing.classList.remove("leaving");
    main.hidden = false;
    $$(".tile", main).forEach((t, i) => t.style.setProperty("--i", i));
    main.classList.add("entering");
    window.scrollTo(0, 0);
    setTimeout(() => main.classList.remove("entering"), 1100);
    document.title = `Account ${S.acct} · Resilience Rewired`;
  }

  function showLanding() {
    $("#main").hidden = true; $("#landing").hidden = false;
    document.title = "Resilience Rewired: cash resilience for any bank account";
    $("#spot-input").value = "";
    $("#spot-input").focus({ preventScroll: true });
  }

  // ---------------------------------------------------------------- account render (static parts)
  function renderAccount() {
    const d = S.data, p = d.profile;
    const pills = [`${monthLabel(d.first_month)} – ${monthLabel(d.last_month)}`, `${d.n_months} months recorded`];
    let html = pills.map((t) => `<span class="pill">${esc(t)}</span>`).join("");
    if (d.last_month < "2025-12") html += `<span class="pill warn">Records end ${esc(monthLabel(d.last_month))}</span>`;
    if (!d.data_ok) html += `<span class="pill warn">Data check needed</span>`;
    $("#meta-pills").innerHTML = html;

    const figs = [
      ["Cash cushion", money(p.B), "6-month average balance — historical, not today's balance", p.B <= 0],
      ["Money in", money(p.I), "Typical month (median). Not necessarily salary."],
      ["Money out", money(p.S), "Typical month (median), incl. cash withdrawals."],
      ["Household bills", money(p.H), "Typical month (median)."],
      ["Regular payments", money(p.F), "Bills + insurance + loan payments, median of monthly sums."],
      ["Loan payment", p.P > 0 ? money(p.P) : "None", p.P > 0 ? "Already included in money out, not added again." : "No loan running at the end of 2025."],
    ];
    $("#hero-figs").innerHTML = figs.map(([k, v, title, risk]) =>
      `<div title="${esc(title)}"><dt>${esc(k)}</dt><dd class="${risk ? "risk" : ""}">${esc(v)}</dd></div>`).join("");

    const oneMax = Math.max(Math.ceil((p.S * 6) / 1000) * 1000, 10000);
    $("#r-oneoff").max = oneMax;
    syncControls();
    drawMonths();
    renderPeers();
  }

  // ---------------------------------------------------------------- controls
  function syncControls() {
    const sh = S.shock, plan = planFor(S.acct), p = S.data.profile;
    const setRange = (el, v) => { el.value = v; el.style.setProperty("--pct", ((v - el.min) / (el.max - el.min)) * 100 + "%"); };
    $("#r-keep").max = Math.max(100, Math.ceil(sh.keep));       // a raise or a big bill rise stretches the slider
    $("#r-bills").max = Math.max(100, Math.ceil(sh.bills));
    setRange($("#r-keep"), sh.keep);
    setRange($("#r-bills"), sh.bills);
    setRange($("#r-start"), sh.startIn); setRange($("#r-lasts"), sh.lasts); setRange($("#r-after"), sh.afterPct);
    $("#o-start").textContent = sh.startIn ? `${sh.startIn} mo` : "Now";
    $("#o-lasts").textContent = sh.lasts ? `${sh.lasts} mo` : "Ongoing";
    $("#o-after").textContent = `${sh.afterPct}%`;
    $("#c-after").classList.toggle("off", !sh.lasts); $("#r-after").disabled = !sh.lasts;
    if (document.activeElement !== $("#i-xout")) $("#i-xout").value = sh.extraOut || "";
    if (document.activeElement !== $("#i-wind")) $("#i-wind").value = sh.windfall || "";
    setRange($("#r-oneoff"), Math.min(sh.oneOff, Number($("#r-oneoff").max)));
    $("#o-keep").textContent = `${sh.keep}%`;
    $("#n-keep").textContent = sh.keep === 100 ? "No change to money in."
      : sh.keep > 100 ? `${+(sh.keep - 100).toFixed(1)}% more than usual · ${money((p.I * (sh.keep - 100)) / 100, true)} a month`
      : `${+(100 - sh.keep).toFixed(1)}% less than usual · ${money(-(p.I * (100 - sh.keep)) / 100)} a month`;
    $("#o-bills").textContent = sh.bills ? `+${sh.bills}%` : "No change";
    $("#n-bills").textContent = !sh.bills ? "Household bills only — insurance and loan payments don't change."
      : p.H > 0 ? `+${money((p.H * sh.bills) / 100)} a month on ${money(p.H)} of recorded bills` : "No household bills are recorded for this account.";
    $("#o-oneoff").textContent = sh.oneOff ? money(sh.oneOff) : "None";
    $$("#presets button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.preset === S.preset)));
    $("#ei-note").hidden = !(S.preset === "ei55" || S.preset === "combined");

    $$("#targets button").forEach((b) => b.setAttribute("aria-checked", String(Number(b.dataset.target) === plan.target)));
    $("#sw-checked").setAttribute("aria-checked", String(plan.checked));
    const cut = $("#i-cut"); cut.disabled = !plan.checked; cut.value = plan.cut || "";
    cut.placeholder = plan.checked ? "0" : "Not checked yet";
    $("#f-cut").classList.toggle("off", !plan.checked);
    $("#i-extra").value = plan.extra || ""; $("#i-extra").placeholder = "0";
  }

  function bindControls() {
    const onShock = () => { S.preset = "custom"; syncControls(); simulate(); };
    $("#r-keep").addEventListener("input", (e) => { S.shock.keep = Number(e.target.value); S.label = ""; onShock(); });
    const num = (sel, key, min = 0) => $(sel).addEventListener("input", (e) => { S.shock[key] = Math.max(Number(e.target.value) || 0, min); onShock(); });
    num("#i-xout", "extraOut", -1e9); num("#i-wind", "windfall");
    $("#r-start").addEventListener("input", (e) => { S.shock.startIn = Number(e.target.value); onShock(); });
    $("#r-lasts").addEventListener("input", (e) => { S.shock.lasts = Number(e.target.value); onShock(); });
    $("#r-after").addEventListener("input", (e) => { S.shock.afterPct = Number(e.target.value); onShock(); });
    $("#twist-go").addEventListener("click", readTwist);
    $("#twist-text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); readTwist(); } });
    $("#msg-go").addEventListener("click", rewriteMessage);
    $$(".theme-btn").forEach((b) => b.addEventListener("click", toggleTheme));
    $("#r-bills").addEventListener("input", (e) => { S.shock.bills = Number(e.target.value); onShock(); });
    $("#r-oneoff").addEventListener("input", (e) => { S.shock.oneOff = Number(e.target.value); onShock(); });
    $("#presets").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-preset]"); if (!b) return;
      const cfg = PRESETS[b.dataset.preset];
      S.preset = b.dataset.preset; S.question = cfg.q; S.label = cfg.label;
      S.shock = { keep: cfg.keep, bills: cfg.bills, oneOff: cfg.oneOff === "S" ? S.data.profile.S : cfg.oneOff, ...NO_EXTRA };
      syncControls(); simulate();
    });
    $("#targets").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-target]"); if (!b) return;
      planFor(S.acct).target = Number(b.dataset.target); syncControls(); simulate();
    });
    $("#sw-checked").addEventListener("click", () => {
      const pl = planFor(S.acct); pl.checked = !pl.checked; syncControls(); simulate();
      if (pl.checked) $("#i-cut").focus();
    });
    const numInput = (sel, key) => $(sel).addEventListener("input", (e) => {
      const v = Math.max(Number(e.target.value) || 0, 0); planFor(S.acct)[key] = v; simulate();
    });
    numInput("#i-cut", "cut"); numInput("#i-extra", "extra");
  }

  // ---------------------------------------------------------------- simulate (debounced to one request per frame)
  let pending = 0, inflight = null;
  function simulate() {
    clearTimeout(pending);
    pending = setTimeout(async () => {
      if (inflight) inflight.abort();
      inflight = new AbortController();
      const pl = planFor(S.acct), sh = S.shock;
      try {
        const sim = await api("/api/simulate", {
          method: "POST", headers: { "Content-Type": "application/json" }, signal: inflight.signal,
          body: JSON.stringify(simBody()),
        });
        if (sim.account_id !== S.acct) return;
        S.sim = sim; renderSim();
      } catch (err) { if (err.name !== "AbortError") toast(err.message); }
    }, 0);
  }

  function simBody() {
    const pl = planFor(S.acct), sh = S.shock;
    return { account_id: S.acct, keep: sh.keep, bills_pct: sh.bills, one_off: sh.oneOff, extra_out: sh.extraOut,
      windfall: sh.windfall, start_in: sh.startIn, lasts: sh.lasts, after_pct: sh.afterPct, question: S.question,
      label: S.label, target: pl.target, checked: pl.checked, cut: pl.cut, extra: pl.extra, kmon: pl.kmon, save: pl.save };
  }

  // ---------------------------------------------------------------- twist card (Gemini, rule-based backup)
  async function readTwist() {
    const text = $("#twist-text").value.trim(), btn = $("#twist-go");
    if (!text) { $("#twist-text").focus(); toast("Paste a twist card first."); return; }
    if (btn.classList.contains("busy")) return;
    btn.classList.add("busy"); btn.firstElementChild.textContent = "Reading…";
    try {
      const t = await api("/api/twist", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_id: S.acct, text }) });
      const before = { ...S.shock, target: planFor(S.acct).target };
      S.shock = { ...t.shock }; S.preset = "custom"; S.question = t.question || "auto";
      S.label = t.label || "";
      if (t.target) planFor(S.acct).target = t.target;
      const note = $("#twist-note");
      note.innerHTML = esc(t.note) + (t.warn.length ? `<span class="warn">Not modelled: ${esc(t.warn.join("; "))}</span>` : "");
      note.hidden = false;
      const more = S.shock.extraOut || S.shock.windfall || S.shock.startIn || S.shock.lasts;
      if (more) $("#more-shock").open = true;
      syncControls(); simulate();
      // glow the controls the card changed, one after another
      const map = { keep: "#r-keep", bills: "#r-bills", oneOff: "#r-oneoff", extraOut: "#i-xout", windfall: "#i-wind",
        startIn: "#r-start", lasts: "#r-lasts", afterPct: "#r-after", target: "#targets" };
      let k = 0;
      Object.entries(map).forEach(([key, sel]) => {
        const now = key === "target" ? planFor(S.acct).target : S.shock[key];
        if (now === before[key]) return;
        const el = $(sel).closest(".control, .money-field") || $(sel);
        el.classList.remove("flash"); void el.offsetWidth; el.style.setProperty("--k", k++); el.classList.add("flash");
      });
    } catch (err) { toast(err.message); }
    finally { btn.classList.remove("busy"); btn.firstElementChild.textContent = "Read twist card"; }
  }

  // ---------------------------------------------------------------- theme (circular reveal from the button)
  function currentTheme() {
    const t = document.documentElement.dataset.theme;
    return t || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }
  function toggleTheme(e) {
    const next = currentTheme() === "dark" ? "light" : "dark";
    const apply = () => {
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("rr-theme", next); } catch (err) { /* private mode */ }
      document.querySelector('meta[name="theme-color"]').content = next === "dark" ? "#0D0D0C" : "#EFECE6";
    };
    if (!document.startViewTransition || reduced) { apply(); if (S.sim) renderTrajectory(false); return; }
    const r = e.currentTarget.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
    const end = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
    const vt = document.startViewTransition(apply);
    vt.ready.then(() => document.documentElement.animate(
      { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${end}px at ${x}px ${y}px)`] },
      { duration: 620, easing: "cubic-bezier(0.22, 1, 0.36, 1)", pseudoElement: "::view-transition-new(root)" }));
  }

  const isRisk = (r) => r.status === "buffer_gone" || r.status === "shortfall";
  function renderRunway(el, r) {
    if (r.status === "shortfall") roll(el, r.raw, (v) => `${v.toFixed(1)} months`, r.headline);
    else if (r.status === "buffer_gone") roll(el, 0, (v) => `${v.toFixed(1)} months`, r.headline);
    else swapText(el, r.headline);
  }

  function renderSim() {
    const s = S.sim, a = s.after, b = s.base, pl = s.planned, t = s.target;
    // hero
    const num = $("#runway-num");
    renderRunway(num, a);
    num.classList.toggle("risk", !s.lasts);
    $("#runway-sub").textContent = a.sub;
    const v = $("#verdict");
    const ok = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>';
    const no = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 7v6M12 16.5v.5"/><circle cx="12" cy="12" r="9"/></svg>';
    const cp = s.can_pay, src = cp && cp.windfall ? `savings + the ${money(cp.windfall)} coming in` : "savings";
    if (s.question === "can_pay" && cp) {
      if (cp.left > 0) { v.className = "verdict yes"; v.innerHTML = `${ok}Can pay the ${money(cp.cost)} from ${src} · ${money(cp.left)} left`; }
      else { v.className = "verdict no"; v.innerHTML = `${no}${src[0].toUpperCase() + src.slice(1)} (${money(cp.have)}) can't cover ${money(cp.cost)} · ${money(-cp.left)} short`; }
    } else if (s.question === "how_long") {
      v.className = "verdict " + (s.lasts ? "yes" : "no");
      v.innerHTML = `${s.lasts ? ok : no}Money lasts ${esc(a.headline.toLowerCase())}`;
    }
    else if (s.lasts) { v.className = "verdict yes"; v.innerHTML = `${ok}Covers the ${t}-month target`; }
    else if (a.status === "buffer_gone") { v.className = "verdict no"; v.innerHTML = `${no}No cushion to draw on`; }
    else { v.className = "verdict no"; v.innerHTML = `${no}Short of ${t} months by ${esc(money(s.short_by))}`; }
    const tm = $("#runway-timed");
    if (s.timed) {
      const ends = s.timed.lasts ? `lasts ${s.timed.lasts} months, then money in at ${+s.timed.after_pct.toFixed(1)}%` : "ongoing";
      tm.innerHTML = `Month by month (${s.timed.start_in ? `starts in ${s.timed.start_in} months` : "starts now"}, ${ends}): ` +
        (s.timed.out != null ? `money runs out in <b>${s.timed.out.toFixed(1)} months</b>.` : "money <b>does not run out</b>.") +
        ` <span class="fineprint">The headline uses the judges' formula, which assumes the shock never ends.</span>`;
      tm.hidden = false;
    } else tm.hidden = true;
    swapText($("#cmp-today"), b.headline);
    swapText($("#cmp-shock"), a.headline);
    swapText($("#cmp-plan"), s.confirmed > 0 ? pl.headline : "No changes yet");
    $("#cmp-plan").classList.toggle("empty", !(s.confirmed > 0));

    renderGap(); renderGuide(); renderTable(); renderBackup(); renderTrajectory(); renderPlans(); renderRisk(); renderMessage();
  }

  function renderGap() {
    const s = S.sim, need = s.need;
    const conf = $("#gb-conf"), open = $("#gb-open");
    roll($("#gb-c"), s.confirmed, money);
    const pl = planFor(S.acct);
    if (s.confirmed === 0 && !pl.checked && !pl.extra) $("#gb-c").textContent = "Not checked";
    if (need == null) {
      $("#gb-need").textContent = "—"; $("#gb-r").textContent = "—";
      conf.style.transform = "scaleX(0)"; open.style.transform = "scaleX(1)";
      $("#gb-note").textContent = "No cushion to start with — monthly changes can't cover the starting gap.";
      $("#gb-r").classList.add("open");
      return;
    }
    roll($("#gb-need"), need, money);
    roll($("#gb-r"), s.remaining, money);
    $("#gb-r").classList.toggle("open", s.remaining > 0);
    const c = need > 0 ? Math.min(s.confirmed / need, 1) : 0;
    conf.style.transform = `scaleX(${c})`;
    open.style.transform = `scaleX(${need > 0 && s.remaining > 0 ? 1 : 0})`;
    let note = need === 0 ? `Already lasts ${s.target} months without changes.`
      : s.remaining > 0 ? `${money(s.remaining)} a month is still open.` : `What you confirmed covers ${s.target} months.`;
    if (s.cut_clamped) note = `You entered ${money(s.entered_cut)} — more than all money out under this shock. We apply ${money(s.applied_cut)}. ` + note;
    $("#gb-note").textContent = note;
  }

  function renderGuide() {
    const g = S.sim.guide;
    swapText($("#guide-state"), S.sim.response.label);
    $("#guide-why").textContent = g.why.join(" ");
    $("#guide-steps").innerHTML = g.actions.map((x) => `<li>${esc(x)}</li>`).join("");
    const res = $("#guide-result");
    res.className = "guide-result" + (g.open_gap ? " open" : "");
    res.innerHTML = g.result.map((x) => `<p>${esc(x)}</p>`).join("");
  }

  function renderTable() {
    const s = S.sim, cols = [s.base, s.after, s.planned];
    const row = (label, f, riskFn) => `<tr><td>${label}</td>${cols.map((r) => `<td class="${riskFn && riskFn(r) ? "risk" : ""}">${f(r)}</td>`).join("")}</tr>`;
    $("#figtable").innerHTML = `<thead><tr><th>Per month</th><th>Today</th><th>With this shock</th><th>With your plan</th></tr></thead><tbody>
      ${row("Money in", (r) => money(r.income))}
      ${row("Money out", (r) => money(r.spending))}
      ${row("Change in cash", (r) => money(-r.gap, true), (r) => r.gap > 0)}
      ${row("Cushion after one-off", (r) => money(r.buffer_left), (r) => r.buffer_left <= 0)}
      ${row("How long it lasts", (r) => esc(r.display),
        (r) => r.status === "buffer_gone" || (r.status === "shortfall" && r.raw < s.target))}
    </tbody>`;
    // cells whose value changed fade up once, so the eye finds what the shock moved
    const cells = $$("#figtable tbody td:not(:first-child)"), texts = cells.map((td) => td.textContent);
    if (!reduced && renderTable._prev && renderTable._acct === S.acct)
      cells.forEach((td, i) => {
        if (texts[i] !== renderTable._prev[i])
          td.animate([{ opacity: 0.2, transform: "translateY(4px)" }, { opacity: 1, transform: "none" }],
            { duration: 320, easing: "cubic-bezier(0.22, 1, 0.36, 1)" });
      });
    renderTable._prev = texts; renderTable._acct = S.acct;
  }

  function renderBackup() {
    const s = S.sim, pl = planFor(S.acct);
    $("#backup").innerHTML = s.backup.map((x) => `<li>${esc(x)}</li>`).join("");
    const f = s.fund, p = S.data.profile;
    const prog = f.target > 0 ? Math.min(Math.max(p.B / f.target, 0), 1) : 1;
    const status = f.reached ? "The recorded cushion already meets this goal."
      : f.months == null ? "Set a monthly amount to see when you'd reach it." : `${f.months} months at ${money(s.save)} a month.`;
    const fund = $("#fund");
    if (!fund.dataset.built) {
      fund.innerHTML = `
        <div class="fund-row"><span>Emergency fund · months of typical spending</span></div>
        <div class="fund-inputs">
          <div class="seg" role="radiogroup" aria-label="Emergency fund goal" id="kmon">
            ${[3, 6, 12].map((k) => `<button type="button" role="radio" data-k="${k}">${k} mo</button>`).join("")}
          </div>
          <label class="money-field"><span>Set aside each month</span>
            <span class="money-input"><span aria-hidden="true">$</span><input type="number" min="0" step="50" id="i-save" inputmode="numeric" placeholder="0"></span></label>
        </div>
        <div class="fund-row"><span>Goal</span><b id="fu-goal"></b></div>
        <div class="fund-track"><span id="fu-bar"></span></div>
        <div class="fund-row"><span>Still to build</span><b id="fu-left"></b></div>
        <p class="fineprint" id="fu-status"></p>
        <p class="fineprint">Uses the recorded cushion and typical spending before the shock; not deducted from the projection above.</p>`;
      fund.dataset.built = "1";
      $("#kmon").addEventListener("click", (e) => {
        const b = e.target.closest("button[data-k]"); if (!b) return; planFor(S.acct).kmon = Number(b.dataset.k); simulate();
      });
      $("#i-save").addEventListener("input", (e) => { planFor(S.acct).save = Math.max(Number(e.target.value) || 0, 0); simulate(); });
    }
    $$("#kmon button").forEach((b) => b.setAttribute("aria-checked", String(Number(b.dataset.k) === pl.kmon)));
    if (document.activeElement !== $("#i-save")) $("#i-save").value = pl.save || "";
    roll($("#fu-goal"), f.target, money);
    roll($("#fu-left"), f.to_build, money);
    $("#fu-bar").style.transform = `scaleX(${prog})`;
    $("#fu-status").textContent = status;
  }

  // ---------------------------------------------------------------- charts (hand-drawn SVG)
  function niceTicks(min, max, n = 4) {
    const span = max - min || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step, out = [];
    for (let v = lo; v <= hi + step / 2; v += step) out.push(+v.toFixed(6));
    return out;
  }

  // Cash trajectory. Lines glide from what was on screen to the new engine projection
  // (interruptible: every tween starts from the values currently drawn).
  const T = { cur: null, raf: 0, land: 0 };
  const resample = (arr, len) => {
    const k = arr.length, slope = k > 1 ? arr[k - 1] - arr[k - 2] : 0;
    return Array.from({ length: len }, (_, i) => (i < k ? arr[i] : arr[k - 1] + slope * (i - k + 1)));
  };
  const easeOut = (t) => 1 - Math.pow(1 - t, 4);

  // smooth curve through the points (Catmull-Rom → cubic Bézier); straight data stays straight
  const smoothPath = (pts) => {
    if (pts.length < 3) return pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
      const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
      const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
      d += `C${c1[0].toFixed(1)},${c1[1].toFixed(1)} ${c2[0].toFixed(1)},${c2[1].toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
    }
    return d;
  };

  function renderTrajectory(animate = true) {
    const box = $("#traj"), sim = S.sim, s = sim.projection, n = s.months.length - 1;
    const W = box.clientWidth || 800, H = box.clientHeight || 320, m = { l: 60, r: 18, t: 36, b: 44 };
    const showPlan = sim.confirmed > 0;
    const to = { today: s.today, shock: s.shock, plan: showPlan ? s.plan : s.shock };
    const vals = s.today.concat(s.shock, showPlan ? s.plan : [], s.timed || [], [0]);
    // focus on the cushion and where it reaches $0: floor at −35% of the cushion, top stretches to fit every line
    const B = S.data.profile.B, maxV = Math.max(...vals);
    let y0, y1, ticks;
    if (B > 0) {
      y0 = -0.3 * B; y1 = Math.max(1.12 * B, maxV * 1.05);
      ticks = niceTicks(y0, y1).filter((t) => t >= y0 - 1e-6 && t <= y1 + 1e-6);
    } else {
      ticks = niceTicks(Math.min(...vals), maxV); y0 = ticks[0]; y1 = ticks[ticks.length - 1];
    }
    const X = (i) => m.l + (i / n) * (W - m.l - m.r), Y = (v) => m.t + (1 - (v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
    const pts = (v) => v.map((y, i) => [X(i), Y(y)]);
    const xt = []; const every = n <= 12 ? 1 : n <= 24 ? 2 : n <= 36 ? 3 : 6;
    for (let i = 0; i <= n; i += every) xt.push(i);
    const bottom = H - m.b, zeroY = Y(0), a = sim.after, pl = sim.planned;
    const area = (v) => `${smoothPath(pts(v))}L${X(n).toFixed(1)},${zeroY.toFixed(1)}L${X(0).toFixed(1)},${zeroY.toFixed(1)}Z`;

    // runway exhaustion pin (engine display value, never re-rounded here)
    const pinAt = (r) => r.status === "shortfall" && r.raw != null && r.raw <= n;
    const dots = [];
    if (pinAt(a)) dots.push(`<circle class="pulse is-shock" cx="${X(a.raw)}" cy="${zeroY}" r="5.5"/><circle class="dot is-shock" cx="${X(a.raw)}" cy="${zeroY}" r="5.5"/>`);
    if (showPlan && pinAt(pl)) dots.push(`<circle class="dot is-plan" cx="${X(pl.raw)}" cy="${zeroY}" r="5"/>`);

    box.classList.remove("settled");
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Projected cash by month: today, with this shock${showPlan ? " and with your plan" : ""}. ${esc(a.headline)} with this shock.">
      <defs>
        <clipPath id="plot-clip"><rect x="${m.l}" y="${m.t - 4}" width="${W - m.l - m.r}" height="${bottom - m.t + 8}"/></clipPath>
        <clipPath id="above-zero"><rect x="${m.l}" y="${m.t - 4}" width="${W - m.l - m.r}" height="${Math.max(zeroY - m.t + 4, 0)}"/></clipPath>
        <linearGradient id="g-shock" gradientUnits="userSpaceOnUse" x1="0" y1="${m.t}" x2="0" y2="${zeroY}">
          <stop offset="0" stop-color="#F57327" stop-opacity=".16"/><stop offset="1" stop-color="#F57327" stop-opacity=".04"/></linearGradient>
        <linearGradient id="g-plan" gradientUnits="userSpaceOnUse" x1="0" y1="${m.t}" x2="0" y2="${zeroY}">
          <stop offset="0" stop-color="#10B981" stop-opacity=".14"/><stop offset="1" stop-color="#10B981" stop-opacity=".03"/></linearGradient>
      </defs>
      <g class="grid">${ticks.map((t) => `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/>`).join("")}</g>
      <g class="axis">${ticks.map((t) => `<text x="${m.l - 10}" y="${Y(t) + 4}" text-anchor="end">${moneyK(t)}</text>`).join("")}
        ${xt.map((i) => `<text x="${X(i)}" y="${bottom + 18}" text-anchor="middle">${i}</text>`).join("")}
        <text class="axis-title" x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">Months from now</text></g>
      <g clip-path="url(#above-zero)">
        <path class="a-plan" id="ar-plan" fill="url(#g-plan)" style="opacity:${showPlan ? 1 : 0}"/><path class="a-shock" id="ar-shock" fill="url(#g-shock)"/>
      </g>
      <line class="zero" x1="${m.l}" x2="${W - m.r}" y1="${zeroY}" y2="${zeroY}"/>
      <g clip-path="url(#plot-clip)">
        <path class="l-today" id="ln-today"/><path class="l-plan" id="ln-plan" style="opacity:${showPlan ? 1 : 0}"/><path class="l-shock" id="ln-shock"/>
      </g>
      ${s.timed ? `<g clip-path="url(#plot-clip)"><path class="l-timed" id="ln-timed" d="${smoothPath(pts(s.timed))}"/></g>` : ""}
      <g class="ann">${dots.join("")}</g>
      <line class="hover-line" id="hv" y1="${m.t}" y2="${bottom}" x1="-10" x2="-10" visibility="hidden"/>
    </svg>${pinAt(a) ? `<div class="pin" id="pin">Runs out · <b>${esc(a.display)} mo</b></div>` : ""}<div class="chart-tip" id="tip" style="opacity:0"></div>`;

    const pin = $("#pin", box);
    if (pin) {   // above the dot, kept inside the chart
      const rect = box.getBoundingClientRect(), sx = rect.width / W, sy = rect.height / H, half = pin.offsetWidth / 2;
      pin.style.left = Math.min(Math.max(X(a.raw) * sx, half + 4), rect.width - half - 4) + "px";
      pin.style.top = zeroY * sy - 14 + "px";
    }

    // tween from what was drawn (first time: from a flat line at today's cushion)
    const keys = ["today", "shock", "plan"];
    const from = {};
    keys.forEach((k) => { from[k] = T.cur ? resample(T.cur[k], n + 1) : to[k].map(() => to[k][0]); });
    const els = Object.fromEntries(keys.map((k) => [k, $(`#ln-${k}`, box)]));
    const areas = { shock: $("#ar-shock", box), plan: $("#ar-plan", box) };
    const draw = (e) => {
      const cur = {};
      keys.forEach((k) => {
        cur[k] = to[k].map((v, i) => from[k][i] + (v - from[k][i]) * e);
        els[k].setAttribute("d", smoothPath(pts(cur[k])));
        if (areas[k]) areas[k].setAttribute("d", area(cur[k]));
      });
      T.cur = cur;
    };
    cancelAnimationFrame(T.raf); clearTimeout(T.land);
    const still = !animate || reduced || keys.every((k) => to[k].every((v, i) => Math.abs(v - from[k][i]) < 0.5));
    if (still) { draw(1); box.classList.add("settled"); }
    else {
      const start = performance.now(), dur = 560;
      const step = (now) => { const t = Math.min((now - start) / dur, 1); draw(easeOut(t)); if (t < 1) T.raf = requestAnimationFrame(step); };
      draw(0); T.raf = requestAnimationFrame(step);
      T.land = setTimeout(() => { cancelAnimationFrame(T.raf); draw(1); }, dur + 80);   // lands even if frames are throttled
      setTimeout(() => box.classList.add("settled"), 30);
    }

    const lt = $("#ln-timed", box);
    $("#lg-timed").hidden = !s.timed;
    if (lt && animate && !reduced) {         // the month-by-month line draws itself in
      const L = lt.getTotalLength();
      lt.animate([{ strokeDasharray: `${L} ${L}`, strokeDashoffset: L }, { strokeDasharray: `${L} ${L}`, strokeDashoffset: 0 }],
        { duration: 900, easing: "cubic-bezier(0.22, 1, 0.36, 1)", delay: 200 });
    }
    const svg = $("svg", box), tip = $("#tip", box), hv = $("#hv", box);
    svg.addEventListener("pointermove", (e) => {
      const rect = svg.getBoundingClientRect(), x = ((e.clientX - rect.left) / rect.width) * W;
      const i = Math.max(0, Math.min(n, Math.round(((x - m.l) / (W - m.l - m.r)) * n)));
      hv.setAttribute("x1", X(i)); hv.setAttribute("x2", X(i)); hv.setAttribute("visibility", "visible");
      tip.innerHTML = `<b>Month ${i}</b><div class="row"><span>Today</span><span>${money(s.today[i])}</span></div>
        <div class="row"><span>With this shock</span><span>${money(s.shock[i])}</span></div>
        ${showPlan ? `<div class="row"><span>With your plan</span><span>${money(s.plan[i])}</span></div>` : ""}
        ${s.timed ? `<div class="row"><span>Month by month</span><span>${money(s.timed[i])}</span></div>` : ""}`;
      const px = (X(i) / W) * rect.width;
      tip.style.left = Math.min(Math.max(px + 14, 0), rect.width - 180) + "px"; tip.style.top = "8px"; tip.style.opacity = 1;
    });
    svg.addEventListener("pointerleave", () => { tip.style.opacity = 0; hv.setAttribute("visibility", "hidden"); });
    $(".lg-plan").parentElement.style.opacity = showPlan ? 1 : .4;
  }

  // ---------------------------------------------------------------- monthly spending (last 12 recorded months)
  function drawMonths() {
    const d = S.data, box = $("#ch-months"); if (!d || !box) return;
    const rows = d.spending, typ = d.profile.S;
    const W = box.clientWidth || 900, H = box.clientHeight || 240, m = { l: 52, r: 8, t: 10, b: 26 };
    const max = Math.max(...rows.map((r) => r.money_out), typ, 1), ticks = niceTicks(0, max, 3), top = ticks[ticks.length - 1];
    const bw = (W - m.l - m.r) / Math.max(rows.length, 1), Y = (v) => m.t + (1 - v / top) * (H - m.t - m.b);
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Money out per month, last ${rows.length} recorded months, with the typical month as a dashed line">
      <g class="grid">${ticks.map((t) => `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/>`).join("")}</g>
      <g class="axis">${ticks.map((t) => `<text x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end">${moneyK(t)}</text>`).join("")}</g>
      ${rows.map((r, i) => `<rect class="bar" x="${m.l + i * bw + bw * 0.18}" y="${Y(r.money_out)}"
         width="${bw * 0.64}" height="${Math.max(Y(0) - Y(r.money_out), 0)}" rx="4"><title>${monthLabel(r.month)}: ${money(r.money_out)}</title></rect>
         <text class="axis" x="${m.l + i * bw + bw / 2}" y="${H - 8}" text-anchor="middle">${monthLabel(r.month, true)}</text>`).join("")}
      <line class="typical" x1="${m.l}" x2="${W - m.r}" y1="${Y(typ)}" y2="${Y(typ)}"/>
    </svg>`;
    $("#sp-typical").textContent = money(typ);
    $("#sp-note").textContent = rows.length
      ? `Actual money out, ${monthLabel(rows[0].month)} – ${monthLabel(rows[rows.length - 1].month)}. ` +
        `The typical month is the median over all ${d.n_months} recorded months, the figure the judging method uses, so single months can sit above or below it.`
      : "No months recorded.";
  }

  // ---------------------------------------------------------------- peer benchmark
  function renderPeers() {
    const peers = S.data.peers, g = peers[S.peerMode];
    $$("#peer-mode button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === S.peerMode)));
    const pct = (x) => (x == null || !isFinite(x) ? "—" : Math.round(x * 100) + "%");
    const fmtv = { S: money, S_over_I: pct, F_over_I: pct, P_over_I: pct, B_over_S: (x) => (x == null ? "—" : x.toFixed(1) + " mo") };
    const box = $("#peers-body");
    if (!g.ok) { box.innerHTML = `<p class="peer-note">${esc(g.reason)}</p>`; return; }
    const desc = [];
    if (g.age_lo != null) desc.push(`owner age ${g.age_lo}–${g.age_hi} at Dec 31, 2025`);
    if (g.lo != null) desc.push(`money in ${money(g.lo)}–${money(g.hi)} a month (±${Math.round(g.band * 100)}%${g.widened ? ", widened to find 30" : ""})`);
    else desc.push("any money in");
    box.innerHTML = `<p class="peer-note">${g.n.toLocaleString("en-US")} accounts · ${esc(desc.join(" · "))}. A comparison, not a spending target. Family, housing and other accounts aren't visible, and your account isn't in the group.</p>
      <div class="peer-grid"><table class="peer-table"><thead><tr><th>Measure</th><th>You</th><th>Typical</th><th>Middle 50%</th><th>Position</th></tr></thead><tbody>
        ${g.rows.map((r) => `<tr><td>${esc(r.label)}</td><td class="you">${fmtv[r.key](r.you)}</td><td>${fmtv[r.key](r.median)}</td>
          <td>${fmtv[r.key](r.p25)} – ${fmtv[r.key](r.p75)}</td><td>${esc(r.position)}</td></tr>`).join("")}</tbody></table>
        <div class="peer-mix"><h3>Where the money goes · share of money out</h3>
          <ul class="legend" aria-hidden="true" style="margin-top:12px"><li><i class="lg lg-bar"></i>You</li><li><i class="lg lg-bar" style="background:var(--peer-bar, #CFCAC0)"></i>Typical similar account</li></ul>
          <div class="mix">${g.mix.filter((x) => (x.you || 0) + (x.peer || 0) > 0.005).map((x) => `<div class="mix-row"><span>${esc(x.label)}</span>
            <span class="mix-bars"><span class="you" style="transform:scaleX(${x.you || 0})"></span><span class="peer" style="transform:scaleX(${x.peer || 0})"></span></span>
            <b>${pct(x.you)}</b></div>`).join("")}</div></div></div>`;
  }

  // ---------------------------------------------------------------- ranked plans (core.py), FLIP re-order
  const boldNums = (t) => esc(t).replace(/(\$[\d,]+(?:\/month)?|\d+(?:\.\d+)?%|about \d+ months)/g, "<b>$1</b>");
  const evidence = (t) => esc(t).replace(/(\d+\.\d%) \((\d+\/\d+), 95% CI ([^)]+)\)/g,
    '<span class="stat">$1</span> <span class="ci">($2 · CI $3)</span>');
  function renderPlans() {
    const box = $("#plans"), plans = S.sim.plans;
    $("#ranked-count").textContent = `${plans.length} plan${plans.length === 1 ? "" : "s"}`;
    const first = new Map($$(".plan-card", box).map((el) => [el.dataset.key, el.getBoundingClientRect()]));
    const fresh = renderPlans._acct !== S.acct;
    box.innerHTML = plans.length ? plans.map((pl, i) => {
      const key = pl.title.replace(/[\d$,.%]+/g, "#");
      const model = pl.new == null ? "" : `<div class="plan-model"><div class="bars">
          <span class="b-now" style="transform:scaleX(${Math.min(pl.start / 0.4, 1)})"></span>
          <span class="b-after" style="transform:scaleX(${Math.min(pl.new / 0.4, 1)})"></span></div>
          <span>Predicted risk ${(pl.start * 100).toFixed(1)}% → ${(pl.new * 100).toFixed(1)}% for customers already in the "after" profile.
          ${pl.test ? `<span class="plan-test${pl.sig ? " sig" : ""}">${esc(pl.test)}</span>` : ""}</span></div>`;
      return `<article class="plan-card k-${pl.kind}" data-key="${esc(key)}" style="--i:${i}">
        <div class="plan-top"><span class="plan-pill">${esc(pl.pill)}</span></div>
        <h3 class="plan-title">${esc(pl.title)}</h3>
        <p class="plan-do">${boldNums(pl.do)}</p>
        <div class="why"><span>For this customer</span><div>${esc(pl.you)}</div></div>
        <div class="why"><span>In the data</span><div>${evidence(pl.data)}</div></div>
        <div class="why"><span>What Tangerine can do</span><div>${esc(pl.bank)}</div></div>${model}</article>`;
    }).join("") : `<p class="plan-do">This customer can absorb this shock without changes.</p>`;
    renderPlans._acct = S.acct;
    if (reduced) return;
    $$(".plan-card", box).forEach((el) => {
      const was = first.get(el.dataset.key);
      if (!was || fresh) { el.classList.add("enter"); return; }
      const now = el.getBoundingClientRect(), dy = was.top - now.top;
      if (Math.abs(dy) > 2) el.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }],
        { duration: 520, easing: "cubic-bezier(0.22, 1, 0.36, 1)" });
    });
  }

  function renderRisk() {
    const r = S.sim.risk.base, col = r >= 0.10 ? "var(--risk)" : r >= 0.04 ? "#E2A50F" : "var(--ok)";
    const g = $("#gauge"); g.style.setProperty("--g-col", col);
    $("#g-fill").style.strokeDashoffset = 264 * (1 - Math.min(r / 0.4, 1));
    roll($("#g-num"), r * 100, (v) => `${v.toFixed(1)}%`);
    $("#g-note").textContent = `Logistic model on savings, transfers and past overdrafts · AUC ${S.sim.risk.auc.toFixed(2)} on unseen customers. Scale tops out at 40%.`;
  }

  function renderMessage() {
    const key = JSON.stringify(simBody());
    if (S.msgKey === key) return;              // keep a Gemini rewrite until the scenario changes
    S.msgKey = key; $("#msg-body").textContent = S.sim.message; $("#msg-src").textContent = "Template";
  }
  async function rewriteMessage() {
    const btn = $("#msg-go"), body = $("#msg-body");
    if (btn.classList.contains("busy")) return;
    btn.classList.add("busy"); body.classList.add("loading");
    try {
      const m = await api("/api/message", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(simBody()) });
      body.classList.remove("loading");
      $("#msg-src").textContent = m.source === "Gemini" ? "Gemini" : "Template";
      if (m.source !== "Gemini") toast(`${m.error || "Gemini unavailable"}: kept the template message.`);
      typeOut(body, m.message);
    } catch (err) { body.classList.remove("loading"); toast(err.message); }
    finally { btn.classList.remove("busy"); }
  }
  function typeOut(el, text) {               // types the message in, about 1.2 s whatever its length
    if (reduced) { el.textContent = text; return; }
    cancelAnimationFrame(el._raf); el.classList.add("caret");
    const start = performance.now(), dur = 1200;
    const step = (now) => {
      const t = Math.min((now - start) / dur, 1);
      el.textContent = text.slice(0, Math.ceil(text.length * t));
      if (t < 1) el._raf = requestAnimationFrame(step); else el.classList.remove("caret");
    };
    el._raf = requestAnimationFrame(step);
  }

  // ---------------------------------------------------------------- findings (bars grow when scrolled into view)
  async function loadFindings() {
    let f;
    try { f = await api("/api/findings"); } catch (err) { return; }
    $("#find-scope").textContent = f.scope;
    const fmt = (c, v) => (c.fmt === "auc" ? v.toFixed(2) : (v * 100).toFixed(1) + "%");
    $("#fgrid").innerHTML = f.cards.map((c, i) => {
      const top = c.scale || Math.max(...c.rows.map((r) => r.value)) || 1;
      const good = /AUC|0\.5/.test(c.big);
      return `<article class="fcard${good ? " good" : ""}" style="--i:${i}">
        <h3><i>${String(i + 1).padStart(2, "0")}</i>${esc(c.title)}</h3>
        <div class="fbig">${esc(c.big)}</div><p class="fsub">${esc(c.sub)}</p>
        <div class="frows">${c.rows.map((r, j) => `<div class="frow" title="${r.lo != null ? `95% CI ${(r.lo * 100).toFixed(1)}%–${(r.hi * 100).toFixed(1)}%` : ""}">
          <span>${esc(r.label)}${r.note ? `<small>${esc(r.note)}</small>` : ""}</span>
          <span class="ftrack"><span class="t-${r.tone}" style="--w:${Math.min(r.value / top, 1)};--j:${j}"></span>
            ${r.lo != null ? `<i class="t-${r.tone}" style="left:${Math.min(r.lo / top, 1) * 100}%;width:${Math.max(Math.min(r.hi / top, 1) - Math.min(r.lo / top, 1), 0) * 100}%"></i>` : ""}</span>
          <b>${fmt(c, r.value)}</b></div>`).join("")}</div>
        ${c.test ? `<p class="ftest"><span>Test</span>${esc(c.test)}</p>` : ""}
        ${c.notes.map((n) => `<p class="fnote">${esc(n)}</p>`).join("")}</article>`;
    }).join("");
    const cards = $$(".fcard");
    if (!("IntersectionObserver" in window) || reduced) { cards.forEach((c) => c.classList.add("seen")); return; }
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("seen"); io.unobserve(e.target); }
    }), { threshold: 0.25 });
    cards.forEach((c) => io.observe(c));
  }

  // ---------------------------------------------------------------- boot
  async function boot() {
    let o;
    try { o = await api("/api/overview"); } catch (err) { toast(`Couldn't load data: ${err.message}`); return; }
    S.overview = o; S.ids = o.accounts.map((a) => a.id); o.accounts.forEach((a) => S.byId.set(a.id, a));
    typeahead($("#spot-input"), $("#spot-list"), $(".spotlight"));
    typeahead($("#switch-input"), $("#switch-list"), $(".switcher"));
    bindControls();
    loadFindings();
    $("#peer-mode").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-mode]"); if (!b || !S.data) return; S.peerMode = b.dataset.mode; renderPeers();
    });
    $("#back").addEventListener("click", () => { history.pushState(null, "", "#/"); showLanding(); });
    document.addEventListener("keydown", (e) => {
      if (e.key !== "/" || /input|textarea|select/i.test(document.activeElement.tagName)) return;
      e.preventDefault(); ($("#main").hidden ? $("#spot-input") : $("#switch-input")).focus();
    });
    window.addEventListener("popstate", route);
    let rz; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { if (S.sim) renderTrajectory(false); drawMonths(); }, 120); });
    route();
  }
  function route() {
    const m = location.hash.match(/^#\/account\/(\d+)/);
    if (m) openAccount(Number(m[1])); else if (!$("#main").hidden) showLanding();
  }
  boot();
})();
