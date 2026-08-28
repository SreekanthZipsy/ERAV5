/* Attention Bill Timeline — interactive layer */

(function () {
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

  function eraBadge(era) {
    return `<span class="badge badge-era-${era}">${window.ERAS[era]?.title || era}</span>`;
  }

  function vizFor(m) {
    const id = m.id;
    if (id === "scaled-dot-product") {
      return `<div class="viz" aria-hidden="true"><svg viewBox="0 0 320 120" width="320" height="120">
        <text x="8" y="18" fill="#4a5568" font-size="11" font-family="IBM Plex Mono,monospace">QKᵀ → softmax → V</text>
        ${[0,1,2,3].map((i)=>[0,1,2,3].map((j)=>{
          const v = i>=j ? 0.2+((i*3+j)%5)/8 : 0.05;
          const c = Math.round(13+v*140);
          return `<rect x="${40+j*28}" y="${30+i*20}" width="24" height="16" rx="2" fill="rgb(${200+c/2},${120+c/3},${80})" opacity="${0.35+v}"/>`;
        }).join("")).join("")}
        <text x="180" y="70" fill="#0d5c63" font-size="12">n×n scores</text>
      </svg></div>`;
    }
    if (id === "mqa" || id === "gqa") {
      const groups = id === "mqa" ? 1 : 2;
      return `<div class="viz"><svg viewBox="0 0 320 100" width="320" height="100">
        <text x="8" y="16" fill="#4a5568" font-size="11" font-family="IBM Plex Mono,monospace">${id.toUpperCase()}: Q heads share KV</text>
        ${[0,1,2,3].map((i)=>`<rect x="${20+i*36}" y="36" width="28" height="18" rx="3" fill="#d7e8ea" stroke="#0d5c63"/>`).join("")}
        <text x="20" y="72" font-size="10" fill="#4a5568">Q₀ Q₁ Q₂ Q₃</text>
        ${Array.from({length:groups},(_,g)=>`<rect x="${200+g*40}" y="36" width="32" height="18" rx="3" fill="#c45c26"/>`).join("")}
        <text x="200" y="72" font-size="10" fill="#4a5568">KV×${groups}</text>
      </svg></div>`;
    }
    if (id === "sliding-window" || id === "attention-sinks") {
      return `<div class="viz"><svg viewBox="0 0 340 90" width="340" height="90">
        ${[0,1,2,3,4,5,6,7,8,9].map((i)=>{
          const sink = id==="attention-sinks" && i<2;
          const win = i>=6;
          const fill = sink ? "#c45c26" : win ? "#0d5c63" : "#d4cfc4";
          return `<rect x="${16+i*30}" y="30" width="24" height="24" rx="4" fill="${fill}"/>`;
        }).join("")}
        <text x="16" y="78" font-size="10" fill="#4a5568">${id==="attention-sinks"?"sinks + rolling window":"local window only"}</text>
      </svg></div>`;
    }
    if (id === "rope" || id === "sinusoidal" || id === "alibi" || id === "drope" || id === "ntk-aware" || id === "yarn" || id === "learned-absolute") {
      return `<div class="viz"><svg viewBox="0 0 300 90" width="300" height="90">
        <circle cx="70" cy="48" r="28" fill="none" stroke="#0d5c63" stroke-width="2"/>
        <line x1="70" y1="48" x2="92" y2="28" stroke="#c45c26" stroke-width="2"/>
        <circle cx="92" cy="28" r="4" fill="#c45c26"/>
        <text x="120" y="40" font-size="11" fill="#1a2332">${m.short}</text>
        <text x="120" y="58" font-size="10" fill="#4a5568">position as geometry / bias</text>
      </svg></div>`;
    }
    if (id === "linear-attention" || id === "delta-rule" || id === "gated-deltanet") {
      return `<div class="viz"><svg viewBox="0 0 300 90" width="300" height="90">
        <rect x="20" y="25" width="70" height="50" rx="6" fill="#1a2332"/>
        <text x="32" y="55" fill="#f3f1ec" font-size="12" font-family="IBM Plex Mono,monospace">S d×d</text>
        <path d="M100 50 H150" stroke="#c45c26" stroke-width="2"/>
        <polygon points="150,46 160,50 150,54" fill="#c45c26"/>
        <text x="168" y="45" font-size="11" fill="#1a2332">fixed state · grows with d, not n</text>
      </svg></div>`;
    }
    if (id === "mla" || id === "deepseek-nsa" || id === "deepseek-dsa" || id === "sparse-topk") {
      return `<div class="viz"><svg viewBox="0 0 320 90" width="320" height="90">
        <rect x="20" y="28" width="90" height="36" rx="4" fill="#e8a882"/>
        <text x="30" y="50" font-size="11">${id==="sparse-topk"?"dense n×n":"fat K/V"}</text>
        <text x="120" y="50" font-size="16" fill="#c45c26">→</text>
        <rect x="150" y="32" width="50" height="28" rx="4" fill="#0d5c63"/>
        <text x="155" y="50" font-size="10" fill="#fff">${id==="mla"?"latent":"sparse"}</text>
        <text x="220" y="50" font-size="11" fill="#1a2332">${
          id === "mla" ? "cache this" :
          id === "deepseek-nsa" ? "compress+select" :
          id === "deepseek-dsa" ? "top-k on MLA" :
          "keep top-k / pattern"
        }</text>
      </svg></div>`;
    }
    return "";
  }

  function cardHTML(m) {
    const arxivLink = m.arxiv
      ? `<a href="https://arxiv.org/abs/${m.arxiv}" target="_blank" rel="noopener">arXiv:${m.arxiv}</a>`
      : "see source note";
    const bonus = m.bonus ? `<span class="badge badge-bonus">bonus · not in class</span>` : "";
    return `
<details class="mech-card" data-era="${m.era}" data-id="${m.id}" data-bonus="${m.bonus ? "1" : "0"}" id="${m.id}">
  <summary class="mech-head">
    <div>
      <h3 class="mech-title">${m.name}</h3>
      <div class="mech-sub">
        ${eraBadge(m.era)}
        ${bonus}
        <span>${m.authors}</span>
      </div>
    </div>
    <div class="mech-date">${m.dateLabel}<br><span style="font-size:0.72rem;color:#4a5568">${m.date}</span></div>
  </summary>
  <div class="mech-body">
    <div class="problem-box">
      <div class="section-label">Problem at the time</div>
      <p>${m.problem}</p>
    </div>
    <div class="answer-box">
      <div class="section-label">What it paid for</div>
      <p>${m.answer}</p>
    </div>
    ${m.formula ? `<div class="formula">${m.formula}</div>` : ""}
    ${vizFor(m)}
    <div class="trade">
      <div class="trade-pro">
        <h4>Buys</h4>
        <ul>${m.buys.map((x) => `<li>${x}</li>`).join("")}</ul>
      </div>
      <div class="trade-con">
        <h4>Gives up</h4>
        <ul>${m.costs.map((x) => `<li>${x}</li>`).join("")}</ul>
      </div>
    </div>
    <div class="when-box">
      <div class="section-label">When you would actually pick it</div>
      <p>${m.when}</p>
      <p style="margin-top:0.5rem"><strong>Pick for:</strong> ${m.pickFor} · <strong>Skip for:</strong> ${m.skipFor}</p>
    </div>
    <p class="cite">${m.paper} · ${arxivLink}<br>${m.sourceNote}</p>
  </div>
</details>`;
  }

  function renderTimeline() {
    const root = $("#timeline");
    const mechs = window.MECHANISMS.slice().sort((a, b) =>
      a.date === b.date ? 0 : a.date < b.date ? -1 : 1
    );

    let html = "";
    let lastYear = null;
    for (const m of mechs) {
      const year = m.date.slice(0, 4);
      if (year !== lastYear) {
        html += `<div class="year-mark" data-year="${year}">${year}</div>`;
        lastYear = year;
      }
      html += cardHTML(m);
    }
    root.innerHTML = html;

    // Open the first (standard attention) by default
    const first = root.querySelector("details.mech-card");
    if (first) first.open = true;
  }

  function setupFilters() {
    $$("[data-filter]").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$("[data-filter]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const f = btn.dataset.filter;
        $$(".mech-card").forEach((card) => {
          let show = true;
          if (f === "class") show = card.dataset.bonus !== "1";
          else if (f === "bonus") show = card.dataset.bonus === "1";
          else if (f !== "all") show = card.dataset.era === f;
          card.classList.toggle("hidden", !show);
        });
        $$(".year-mark").forEach((ym) => {
          let next = ym.nextElementSibling;
          let any = false;
          while (next && !next.classList.contains("year-mark")) {
            if (next.classList.contains("mech-card") && !next.classList.contains("hidden")) any = true;
            next = next.nextElementSibling;
          }
          ym.style.display = any ? "" : "none";
        });
      });
    });
  }

  /* —— Vanilla attention playground —— */
  const DEMO_SENTENCES = {
    cat: ["The", "cat", "sat", "on", "the", "mat"],
    code: ["def", "add", "(", "a", ",", "b", ")"],
    short: ["Hello", "world"]
  };

  // Tiny deterministic "embeddings" for demo (not real model weights)
  function embed(token, dim = 8) {
    let h = 2166136261;
    for (let i = 0; i < token.length; i++) {
      h ^= token.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    const v = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      h = Math.imul(h ^ d, 1973) >>> 0;
      v[d] = ((h % 1000) / 500 - 1) * 1.2;
    }
    return v;
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function softmax(arr) {
    const m = Math.max(...arr);
    const ex = arr.map((x) => Math.exp(x - m));
    const z = ex.reduce((a, b) => a + b, 0);
    return ex.map((x) => x / z);
  }

  function causalAttention(tokens) {
    const dim = 8;
    const Q = tokens.map((t) => embed(t + "|q", dim));
    const K = tokens.map((t) => embed(t + "|k", dim));
    const n = tokens.length;
    const scores = Array.from({ length: n }, () => Array(n).fill(-Infinity));
    const weights = Array.from({ length: n }, () => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j <= i; j++) {
        const s = dot(Q[i], K[j]) / Math.sqrt(dim);
        scores[i][j] = s;
        row.push(s);
      }
      const sm = softmax(row);
      for (let j = 0; j <= i; j++) weights[i][j] = sm[j];
    }
    return { scores, weights, n };
  }

  function heatColor(t) {
    // t in [0,1]
    const r = Math.round(243 - t * 100);
    const g = Math.round(241 - t * 160);
    const b = Math.round(236 - t * 180);
    return `rgb(${r},${g},${b})`;
  }

  let demoState = { tokens: DEMO_SENTENCES.cat, focus: 0 };

  function renderDemo() {
    const { tokens, focus } = demoState;
    const { weights, n } = causalAttention(tokens);

    const tokEl = $("#tokens");
    tokEl.innerHTML = tokens
      .map(
        (t, i) =>
          `<button type="button" class="token ${i === focus ? "active" : ""}" data-i="${i}">${t}</button>`
      )
      .join("");

    $$(".token", tokEl).forEach((btn) => {
      btn.addEventListener("click", () => {
        demoState.focus = +btn.dataset.i;
        renderDemo();
      });
    });

    // Full weight matrix
    const grid = $("#attn-matrix");
    grid.style.gridTemplateColumns = `repeat(${n}, minmax(2.4rem, 1fr))`;
    let cells = "";
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const w = weights[i][j];
        const label = j > i ? "·" : w.toFixed(2);
        const bg = j > i ? "#eceae4" : heatColor(w);
        cells += `<div class="attn-cell" style="background:${bg}" title="q=${tokens[i]} → k=${tokens[j]}">${label}</div>`;
      }
    }
    grid.innerHTML = cells;

    // Focus row distribution
    const focusEl = $("#focus-row");
    focusEl.style.gridTemplateColumns = `repeat(${n}, minmax(2.4rem, 1fr))`;
    focusEl.innerHTML = weights[focus]
      .map((w, j) => {
        if (j > focus) return `<div class="attn-cell" style="background:#eceae4">·</div>`;
        return `<div class="attn-cell" style="background:${heatColor(w)}">${tokens[j]}<br>${w.toFixed(2)}</div>`;
      })
      .join("");

    // Bill meter: compare n=seq vs larger
    const seq = n;
    const big = +$("#bill-n").value;
    const computeSmall = seq * seq;
    const computeBig = big * big;
    const memSmall = seq;
    const memBig = big;
    const computePct = Math.min(100, (computeBig / (1000 * 1000)) * 100);
    const memPct = Math.min(100, (memBig / 1000) * 100);

    $("#bill-readout").innerHTML = `
      At <b>${seq}</b> tokens: attention scores ≈ <b>${computeSmall.toLocaleString()}</b> entries.
      At <b>${big.toLocaleString()}</b> tokens: ≈ <b>${computeBig.toLocaleString()}</b> entries
      (${(computeBig / computeSmall).toFixed(0)}× more compute-shaped work) while KV cache tokens grow only
      <b>${(memBig / memSmall).toFixed(0)}×</b> — the square bill vs the linear bill from class.
      <div class="bars">
        <div class="bar-row"><span>compute ~n²</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(4, computePct)}%"></div></div><span>${big}²</span></div>
        <div class="bar-row"><span>KV ~n</span><div class="bar-track"><div class="bar-fill mem" style="width:${Math.max(4, memPct)}%"></div></div><span>${big}</span></div>
      </div>`;
  }

  function setupDemo() {
    $("#sentence").addEventListener("change", (e) => {
      demoState.tokens = DEMO_SENTENCES[e.target.value];
      demoState.focus = Math.min(demoState.focus, demoState.tokens.length - 1);
      renderDemo();
    });
    $("#bill-n").addEventListener("input", renderDemo);
    $("#open-standard").addEventListener("click", () => {
      const el = document.getElementById("scaled-dot-product");
      if (el) {
        el.open = true;
        el.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
    renderDemo();
  }

  document.addEventListener("DOMContentLoaded", () => {
    renderTimeline();
    setupFilters();
    setupDemo();
    $("#count").textContent = String(window.MECHANISMS.length);
    $("#class-count").textContent = String(
      window.MECHANISMS.filter((m) => m.coveredInClass).length
    );
    const have = new Set(window.MECHANISMS.map((m) => m.id));
    const missing = (window.REQUIRED_IDS || []).filter((id) => !have.has(id));
    if (missing.length) {
      console.error("Missing required mechanisms:", missing);
    }
    const dates = window.MECHANISMS.map((m) => m.date);
    const sorted = [...dates].sort();
    if (dates.join() !== sorted.join()) {
      console.warn("Mechanisms are not in date order in the data file.");
    }
  });
})();
