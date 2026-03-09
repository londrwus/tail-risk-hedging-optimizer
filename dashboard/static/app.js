// state
let portfolio = { tickers: [], weights: [] };
let cfg = {};
let tabCache = {};
let faqData = {};
let activeTab = "portfolio";

const TABS = [
    { id: "portfolio", label: "Portfolio Overview" },
    { id: "hedge", label: "Hedge Analysis" },
    { id: "frontier", label: "Hedge Frontier" },
    { id: "stress", label: "Stress Testing" },
    { id: "predictions", label: "Price Predictions" },
    { id: "factors", label: "Factor & Drawdown" },
    { id: "riskparity", label: "Risk Parity" },
    { id: "calibration", label: "Portfolio Calibration" },
    { id: "entry", label: "Entry Timing" },
    { id: "ml", label: "ML Models" },
];

// ── init ─────────────────────────────────────────────────────────────────────

async function init() {
    cfg = await fetchJSON("/api/config");
    portfolio.tickers = cfg.tickers;
    portfolio.weights = cfg.weights;

    faqData = await fetchJSON("/api/faq");

    buildTabs();
    buildPortfolioPanel();
    updateSummary();
    switchTab("portfolio");
}

// ── fetch helper ─────────────────────────────────────────────────────────────

async function fetchJSON(url) {
    let r = await fetch(url);
    return r.json();
}

// ── tabs ─────────────────────────────────────────────────────────────────────

function buildTabs() {
    let nav = document.getElementById("tab-nav");
    TABS.forEach(t => {
        let btn = document.createElement("button");
        btn.className = "tab-btn";
        btn.textContent = t.label;
        btn.dataset.tab = t.id;
        btn.onclick = () => switchTab(t.id);
        nav.appendChild(btn);
    });
}

function switchTab(id) {
    activeTab = id;

    // toggle visibility
    document.querySelectorAll(".tab-pane").forEach(p => p.style.display = "none");
    let pane = document.getElementById("tab-" + id);
    if (pane) pane.style.display = "block";

    // highlight nav
    document.querySelectorAll(".tab-btn").forEach(b => {
        b.classList.toggle("active", b.dataset.tab === id);
    });

    loadTab(id);
}

async function loadTab(id) {
    // use cache unless portfolio changed
    if (tabCache[id]) {
        renderTab(id, tabCache[id]);
        return;
    }

    showLoading(true);
    try {
        let url = "/api/tab/" + id;
        // predictions has its own params
        if (id === "predictions") {
            let tk = document.getElementById("pred-ticker")?.value || "SPY";
            let tf = document.getElementById("pred-timeframe")?.value || "1y";
            let ns = document.getElementById("pred-nsims")?.value || "5000";
            url += `?ticker=${tk}&timeframe=${tf}&n_sims=${ns}`;
        }
        let data = await fetchJSON(url);
        if (id !== "predictions" && id !== "ml") tabCache[id] = data; // don't cache dynamic tabs
        renderTab(id, data);
    } catch (e) {
        console.error("Tab load failed:", e);
    }
    showLoading(false);
}

function showLoading(on) {
    document.getElementById("loading").style.display = on ? "flex" : "none";
}

// ── portfolio config ─────────────────────────────────────────────────────────

function buildPortfolioPanel() {
    let div = document.getElementById("portfolio-config");
    let html = '<span class="cfg-label">Portfolio</span>';

    for (let i = 0; i < 11; i++) {
        let t = portfolio.tickers[i] || "";
        let w = i < portfolio.weights.length ? (portfolio.weights[i] * 100).toFixed(0) : "";

        html += '<div class="slot">';
        html += `<select id="pt-${i}">`;
        html += '<option value="">—</option>';
        for (let [label, val] of Object.entries(cfg.universe)) {
            let sel = val === t ? " selected" : "";
            html += `<option value="${val}"${sel}>${label}</option>`;
        }
        html += '</select>';
        html += `<input id="pw-${i}" type="number" min="0" max="100" step="1" value="${w}" placeholder="%">`;
        html += '</div>';
    }

    html += '<div><button id="btn-apply" onclick="applyPortfolio()">Apply</button>';
    html += '<div id="port-status"></div></div>';
    div.innerHTML = html;
}

async function applyPortfolio() {
    let tickers = [], weights = [];
    for (let i = 0; i < 11; i++) {
        let t = document.getElementById("pt-" + i).value;
        let w = parseFloat(document.getElementById("pw-" + i).value);
        if (t && w > 0) {
            tickers.push(t);
            weights.push(w / 100);
        }
    }
    if (!tickers.length) return;

    showLoading(true);
    let res = await fetch("/api/portfolio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers, weights }),
    }).then(r => r.json());

    if (res.status === "ok") {
        portfolio.tickers = res.tickers;
        portfolio.weights = res.weights;
        tabCache = {}; // clear tab cache
        document.getElementById("port-status").textContent = "Applied";

        // update weight inputs with normalized values
        for (let i = 0; i < 11; i++) {
            let wInput = document.getElementById("pw-" + i);
            if (i < res.weights.length) {
                wInput.value = (res.weights[i] * 100).toFixed(1);
            }
        }
        updateSummary();
        loadTab(activeTab);
    }
    showLoading(false);
}

function updateSummary() {
    let el = document.getElementById("portfolio-summary");
    let parts = portfolio.tickers.map((t, i) =>
        `${t}: ${(portfolio.weights[i] * 100).toFixed(0)}%`);
    el.textContent = parts.join(" | ");
}

// ── FAQ ──────────────────────────────────────────────────────────────────────

function showFaq(key) {
    let faq = faqData[key];
    if (!faq) return;
    document.getElementById("faq-title").textContent = faq.title;
    document.getElementById("faq-body").textContent = faq.body;
    document.getElementById("faq-overlay").style.display = "flex";
}

function closeFaq() {
    document.getElementById("faq-overlay").style.display = "none";
}

// ── table builder ────────────────────────────────────────────────────────────

function makeTable(tbl) {
    if (!tbl || !tbl.rows || !tbl.rows.length) return "<p class='muted'>No data</p>";
    let html = "<table><thead><tr>";
    tbl.cols.forEach(c => html += `<th>${c}</th>`);
    html += "</tr></thead><tbody>";
    tbl.rows.forEach(row => {
        html += "<tr>";
        row.forEach(cell => html += `<td>${cell}</td>`);
        html += "</tr>";
    });
    html += "</tbody></table>";
    return html;
}

// ── chart helper ─────────────────────────────────────────────────────────────

function plotChart(divId, figData) {
    if (!figData || !figData.data) return;
    let el = document.getElementById(divId);
    if (!el) return;
    Plotly.newPlot(el, figData.data, figData.layout, { responsive: true });
}

// ── renderers per tab ────────────────────────────────────────────────────────

function renderTab(id, data) {
    let renderers = {
        portfolio: renderPortfolio,
        hedge: renderHedge,
        frontier: renderFrontier,
        stress: renderStress,
        predictions: renderPredictions,
        factors: renderFactors,
        riskparity: renderRiskParity,
        calibration: renderCalibration,
        entry: renderEntry,
        ml: renderML,
    };
    if (renderers[id]) renderers[id](data);
}

function renderPortfolio(d) {
    plotChart("chart-fan", d.fan_chart);
    plotChart("chart-dist", d.return_dist);
    plotChart("chart-rvol", d.rolling_vol);
    plotChart("chart-cum", d.cum_return);
    plotChart("chart-attrib", d.risk_attrib);

    // stats as 2-col table
    let html = "<table>";
    d.stats.forEach(row => {
        html += `<tr><td style="font-weight:600">${row[0]}</td><td>${row[1]}</td></tr>`;
    });
    html += "</table>";
    document.getElementById("tbl-stats").innerHTML = html;
}

function renderHedge(d) {
    plotChart("chart-payoff", d.payoff);
    plotChart("chart-hedged", d.hedged_dist);
    document.getElementById("tbl-greeks").innerHTML = makeTable(d.greeks);
    document.getElementById("tbl-premiums").innerHTML = makeTable(d.premiums);
}

function renderFrontier(d) {
    plotChart("chart-frontier", d.frontier);
    document.getElementById("tbl-opt").innerHTML = makeTable(d.opt_table);
}

function renderStress(d) {
    plotChart("chart-stress-hist", d.stress_hist);
    plotChart("chart-stress-hypo", d.stress_hypo);
    plotChart("chart-corr-normal", d.corr_normal);
    plotChart("chart-corr-stress", d.corr_stress);
    document.getElementById("tbl-stress-detail").innerHTML = makeTable(d.detail);
}

function renderPredictions(d) {
    if (d.error) {
        document.getElementById("pred-current-info").innerHTML =
            `<p style="color:var(--red)">${d.error}</p>`;
        return;
    }

    // populate dropdowns if empty
    let sel = document.getElementById("pred-ticker");
    if (sel.options.length <= 1) {
        sel.innerHTML = "";
        for (let [tk, label] of Object.entries(cfg.pred_tickers)) {
            let opt = document.createElement("option");
            opt.value = tk;
            opt.textContent = `${tk} — ${label}`;
            sel.appendChild(opt);
        }
    }
    let tfSel = document.getElementById("pred-timeframe");
    if (tfSel.options.length <= 1 || tfSel.dataset.built !== "1") {
        tfSel.innerHTML = "";
        for (let [k, label] of Object.entries(cfg.pred_timeframes)) {
            let opt = document.createElement("option");
            opt.value = k;
            opt.textContent = label;
            if (k === "1y") opt.selected = true;
            tfSel.appendChild(opt);
        }
        tfSel.dataset.built = "1";
    }

    let ci = d.current_info;
    document.getElementById("pred-current-info").innerHTML =
        `<h4>${ci.label} (${ci.ticker})</h4>` +
        `<p>Current: $${ci.price.toFixed(2)} | 52w High: $${ci.high_52w.toFixed(2)} | ` +
        `52w Low: $${ci.low_52w.toFixed(2)} | From High: ${(ci.pct_from_high * 100).toFixed(1)}%</p>`;

    plotChart("chart-pred-fan", d.fan_chart);
    plotChart("chart-pred-terminal", d.terminal_dist);
    plotChart("chart-pred-stress", d.stress);
    document.getElementById("tbl-pred-stats").innerHTML = makeTable(d.stats);
    document.getElementById("tbl-pred-levels").innerHTML = makeTable(d.price_levels);

    // risk flags
    let flagsEl = document.getElementById("pred-risk-flags");
    flagsEl.innerHTML = "";
    (d.risk_flags || []).forEach(f => {
        let li = document.createElement("li");
        li.textContent = f;
        let isRed = /high|extreme|tail|coin-flip|30%/i.test(f);
        li.style.color = isRed ? "var(--red)" : "var(--yellow)";
        flagsEl.appendChild(li);
    });

    // rates
    if (d.rates) document.getElementById("tbl-rates").innerHTML = makeTable(d.rates);
    if (d.rates_chart) plotChart("chart-rates", d.rates_chart);
}

function renderFactors(d) {
    plotChart("chart-momentum", d.momentum);
    plotChart("chart-tail", d.tail_dep);
    plotChart("chart-rcorr", d.rolling_corr);
    plotChart("chart-dd", d.drawdown);
    document.getElementById("tbl-dd").innerHTML = makeTable(d.dd_table);
}

function renderRiskParity(d) {
    plotChart("chart-rp-weights", d.weights);
    plotChart("chart-rp-cum", d.cum_return);
    document.getElementById("tbl-rp-comp").innerHTML = makeTable(d.comparison);
}

function renderCalibration(d) {
    plotChart("chart-calib-weights", d.weights);
    plotChart("chart-eff-frontier", d.eff_frontier);
    plotChart("chart-calib-mc", d.mc_box);
    document.getElementById("tbl-calib-strat").innerHTML = makeTable(d.strategy_table);
    document.getElementById("tbl-calib-mc").innerHTML = makeTable(d.mc_table);
    document.getElementById("tbl-calib-detail").innerHTML = makeTable(d.detail_table);
}

function renderEntry(d) {
    let t = d.timing;
    let scoreColor = t.score >= 70 ? "var(--accent)" :
                     t.score >= 50 ? "var(--yellow)" :
                     t.score >= 35 ? "var(--blue)" : "var(--red)";

    let html = `<div style="margin-bottom:12px">`;
    html += `<span class="muted" style="font-size:16px">Timing Score: </span>`;
    html += `<span class="score" style="color:${scoreColor}">${t.score}/100</span></div>`;
    html += `<p class="verdict">${t.verdict}</p>`;
    html += `<p class="recommended">Recommended: ${t.rec_label}</p>`;
    html += `<hr>`;
    html += `<p class="muted" style="font-size:12px;margin-bottom:4px">Key Factors:</p>`;
    t.factors.forEach(f => {
        html += `<p class="factor">&bull; ${f}</p>`;
    });
    html += `<hr>`;
    html += `<p class="context">Portfolio spot: $${t.spot.toFixed(2)} | ` +
            `Ann. vol: ${(t.sigma * 100).toFixed(1)}% | Drift: ${(t.mu * 100).toFixed(1)}%</p>`;

    document.getElementById("entry-verdict").innerHTML = html;
    plotChart("chart-dip-probs", d.dip_probs);
    plotChart("chart-dca", d.dca_comparison);
    plotChart("chart-dca-sched", d.schedules);
    document.getElementById("tbl-dca").innerHTML = makeTable(d.dca_table);
    document.getElementById("tbl-levels").innerHTML = makeTable(d.level_plan);
}

function renderML(d) {
    // summary bar
    let s = d.summary;
    let crashColor = !s.crash_prob ? "var(--muted)" :
                     s.crash_prob > 0.5 ? "var(--red)" :
                     s.crash_prob > 0.3 ? "var(--yellow)" : "var(--accent)";
    let retColor = s.return_pred == null ? "var(--muted)" :
                   s.return_pred > 0 ? "var(--accent)" : "var(--red)";

    let html = `<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:center">`;
    html += `<div><span class="muted">Regime:</span> <strong>${s.regime}</strong></div>`;
    if (s.return_pred != null) {
        html += `<div><span class="muted">21d Return Forecast:</span> <strong style="color:${retColor}">${(s.return_pred * 100).toFixed(2)}%</strong>`;
        if (s.return_dir_acc != null) html += ` <span class="muted">(${(s.return_dir_acc * 100).toFixed(0)}% dir. acc.)</span>`;
        html += `</div>`;
    }
    if (s.vol_pred != null) {
        html += `<div><span class="muted">Predicted Vol:</span> <strong>${(s.vol_pred * 100).toFixed(1)}%</strong></div>`;
    }
    if (s.crash_prob != null) {
        html += `<div><span class="muted">Crash Prob:</span> <strong style="color:${crashColor}">${(s.crash_prob * 100).toFixed(0)}%</strong></div>`;
    }
    html += `</div>`;
    document.getElementById("ml-summary-bar").innerHTML = html;

    plotChart("chart-ml-regime", d.regime_chart);
    document.getElementById("tbl-ml-regime").innerHTML = makeTable(d.regime_centers);
    plotChart("chart-ml-crash", d.crash_gauge);
    document.getElementById("tbl-ml-crash").innerHTML = makeTable(d.crash_stats);
    plotChart("chart-ml-ret-imp", d.return_importance);
    document.getElementById("tbl-ml-ret").innerHTML = makeTable(d.return_stats);
    plotChart("chart-ml-vol-imp", d.vol_importance);
    document.getElementById("tbl-ml-vol").innerHTML = makeTable(d.vol_stats);
    plotChart("chart-ml-cv", d.cv_chart);
}

// ── prediction controls ──────────────────────────────────────────────────────

document.addEventListener("change", e => {
    if (["pred-ticker", "pred-timeframe", "pred-nsims"].includes(e.target.id)) {
        if (activeTab === "predictions") {
            tabCache["predictions"] = null;
            loadTab("predictions");
        }
    }
});

// ── go ───────────────────────────────────────────────────────────────────────

init();
