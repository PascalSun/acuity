(() => {
  const $ = (id) => document.getElementById(id);
  const fmt = (n) =>
    n != null ? n.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "-";
  const fmtSec = (ms) => (ms != null ? fmt(ms / 1000) : "-");
  const trunc = (s, n) => (s && s.length > n ? s.slice(0, n) + "..." : s || "");
  const esc = (s) =>
    (s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  function syntaxHighlightJson(jsonText) {
    const safe = esc(jsonText);
    return safe.replace(
      /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
      (match) => {
        let cls = "json-number";
        if (/^"/.test(match)) {
          cls = /:$/.test(match) ? "json-key" : "json-string";
        } else if (/true|false/.test(match)) {
          cls = "json-boolean";
        } else if (/null/.test(match)) {
          cls = "json-null";
        }
        return `<span class="${cls}">${match}</span>`;
      },
    );
  }

  function renderCodeBlock(el, text) {
    if (!el) return;
    if (!text) {
      el.innerHTML = `<pre class="code-pre"></pre>`;
      return;
    }
    const raw = String(text);
    try {
      const obj = JSON.parse(raw);
      const pretty = JSON.stringify(obj, null, 2);
      el.innerHTML = `<pre class="code-pre">${syntaxHighlightJson(pretty)}</pre>`;
    } catch (_e) {
      el.innerHTML = `<pre class="code-pre">${esc(raw)}</pre>`;
    }
  }
  const fmtTs = (value, withSeconds) => {
    const raw = value == null ? "" : String(value);
    if (!raw) return "-";
    let s = raw;
    if (s.includes(" ") && !s.includes("T")) s = s.replace(" ", "T");
    if (!/[zZ]|[+-]\d\d:\d\d$/.test(s)) s += "Z";
    const d = new Date(s);
    if (Number.isNaN(d.getTime())) return raw;
    const opts = {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    };
    if (withSeconds) opts.second = "2-digit";
    return d.toLocaleString("sv-SE", opts);
  };

  function setTab(tab) {
    const show = (id, on) => {
      const el = $(id);
      if (el) el.style.display = on ? "block" : "none";
    };
    show("viewOverview", tab === "overview");
    show("viewEndpoints", tab === "endpoints");
    show("viewRuns", tab === "runs");
    show("viewLogs", tab === "logs");

    const setActive = (id, on) => {
      const el = $(id);
      if (el) el.classList.toggle("active", on);
    };
    setActive("tabOverview", tab === "overview");
    setActive("tabEndpoints", tab === "endpoints");
    setActive("tabRuns", tab === "runs");
    setActive("tabLogs", tab === "logs");

    if (tab === "logs") {
      const tbody = $("tDetailed")?.querySelector("tbody");
      if (tbody && tbody.children.length === 0) loadDetailed(true);
    }
  }

  function drawChart(data) {
    const cvs = $("trafficChart");
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    const w = (cvs.width = cvs.clientWidth * 2);
    const h = (cvs.height = cvs.clientHeight * 2);
    ctx.scale(2, 2);

    ctx.clearRect(0, 0, w, h);

    const pts = data || [];
    if (!pts.length) {
      ctx.fillStyle = "#64748b";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No data available", w / 4, h / 4);
      return;
    }

    const maxVal = Math.max(1, ...pts.map((p) => p.requests));
    const padding = 30;
    const availW = w / 2 - padding * 2;
    const availH = h / 2 - padding * 2;
    const barW = Math.max(2, availW / pts.length - 6);

    pts.forEach((p, i) => {
      const bh = (p.requests / maxVal) * availH;
      const x = padding + i * (availW / pts.length);
      const y = h / 2 - padding - bh;

      ctx.fillStyle = "#3b82f6";
      ctx.fillRect(x, y, barW, bh);

      if (pts.length < 15 || i % Math.ceil(pts.length / 10) === 0) {
        ctx.fillStyle = "#64748b";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "center";
        const day = (p.day || "").slice(0, 10);
        const date = new Date(day ? day + "T00:00:00" : p.day);
        ctx.fillText(
          date.toLocaleDateString(undefined, { month: "short", day: "numeric" }),
          x + barW / 2,
          h / 2 - 10,
        );
      }
    });
  }

  async function loadOverview() {
    try {
      const res = await fetch("/console/api/summary");
      if (!res.ok) return;
      const data = await res.json();

      $("sTotal").textContent = fmt(data.total.requests);
      const errRate = data.total.requests
        ? (1 - data.total.errors / data.total.requests) * 100
        : 100;
      $("sSuccess").textContent = fmt(errRate) + "%";
      $("sErrors").textContent = fmt(data.total.errors) + " errors";
      $("sLatency").textContent = fmtSec(data.total.avg_ms) + " s";
      $("sP95").textContent = "P95: " + fmtSec(data.total.p95_ms) + " s";
      $("sRuns").textContent = (data.run_ids || []).length;

      drawChart(data.requests_by_day);

      const mkRow = (cols) =>
        `<tr>${cols
          .map(
            (c, i) =>
              `<td class="${
                i > 0 && typeof c === "number" ? "num" : ""
              }">${c}</td>`,
          )
          .join("")}</tr>`;

      $("tRecent").querySelector("tbody").innerHTML = (data.recent_queries || [])
        .map(
          (r) => `
          <tr>
            <td class="mono" style="color:var(--text-light)">${trunc(fmtTs(r.ts, false), 16)}</td>
            <td class="mono">${esc(r.route)}</td>
            <td class="mono">${esc(r.run_id || "-")}</td>
            <td class="cell-truncate" title="${esc(r.query)}">${esc(r.query)}</td>
            <td class="num">${fmtSec(r.duration_ms)}</td>
            <td><span class="badge ${r.success ? "success" : "error"}">${r.success ? "OK" : "ERR"}</span></td>
          </tr>
        `,
        )
        .join("");

      $("tTools").querySelector("tbody").innerHTML = (data.top_tools || [])
        .map((r) => mkRow([r.tool_name, r.requests, r.errors]))
        .join("");

      $("tEndpoints").querySelector("tbody").innerHTML = (data.top_endpoints || [])
        .map((r) =>
          mkRow([
            `<span class="mono">${esc(r.route)}</span>`,
            esc(r.method),
            r.requests,
            fmtSec(r.avg_ms),
          ]),
        )
        .join("");

      $("tRunIds").querySelector("tbody").innerHTML = (data.run_ids || [])
        .map((r) =>
          mkRow([`<span class="mono">${esc(r.run_id)}</span>`, r.requests, r.errors]),
        )
        .join("");

      $("tEndpoints2").querySelector("tbody").innerHTML = (data.top_endpoints || [])
        .map(
          (r) => `
          <tr>
            <td class="mono">${esc(r.route)}</td>
            <td class="mono">${esc(r.method)}</td>
            <td class="num">${fmt(r.requests)}</td>
            <td class="num">${fmt(r.errors)}</td>
            <td class="num">${fmtSec(r.avg_ms)}</td>
            <td class="num">${fmtSec(r.p95_ms)}</td>
            <td><button class="btn secondary" type="button" style="padding:4px 10px;font-size:12px">Logs</button></td>
          </tr>
        `,
        )
        .join("");

      Array.from($("tEndpoints2").querySelectorAll("tbody tr")).forEach((tr, idx) => {
        const row = (data.top_endpoints || [])[idx];
        const btn = tr.querySelector("button");
        if (btn && row)
          btn.onclick = () => {
            $("fRoute").value = row.route || "";
            $("fMethod").value = row.method || "";
            $("fStatus").value = "";
            $("fRunId").value = "";
            $("fQuery").value = "";
            $("fOnlyQuery").checked = false;
            setTab("logs");
            loadDetailed(true);
          };
      });

      $("tRuns2").querySelector("tbody").innerHTML = (data.run_ids || [])
        .map(
          (r) => `
          <tr>
            <td class="mono">${esc(r.run_id)}</td>
            <td class="num">${fmt(r.requests)}</td>
            <td class="num">${fmt(r.errors)}</td>
            <td><button class="btn secondary" type="button" style="padding:4px 10px;font-size:12px">Logs</button></td>
          </tr>
        `,
        )
        .join("");

      Array.from($("tRuns2").querySelectorAll("tbody tr")).forEach((tr, idx) => {
        const row = (data.run_ids || [])[idx];
        const btn = tr.querySelector("button");
        if (btn && row)
          btn.onclick = () => {
            $("fRunId").value = row.run_id || "";
            $("fRunIdMode").value = "exact";
            $("fRoute").value = "";
            $("fMethod").value = "";
            $("fStatus").value = "";
            $("fQuery").value = "";
            $("fOnlyQuery").checked = false;
            setTab("logs");
            loadDetailed(true);
          };
      });
    } catch (e) {
      console.error("Overview load failed", e);
    }
  }

  let offset = 0;
  function currentRange() {
    const mode = $("fRange").value;
    if (mode === "custom") {
      const s = $("fSince").value ? new Date($("fSince").value).toISOString() : "";
      const u = $("fUntil").value ? new Date($("fUntil").value).toISOString() : "";
      return { since: s, until: u };
    }
    if (mode === "24h") {
      const since = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
      return { since };
    }
    if (mode === "7d") {
      const since = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
      return { since };
    }
    return {};
  }

  async function loadDetailed(reset = false) {
    if (reset) {
      offset = 0;
      $("tDetailed").querySelector("tbody").innerHTML = "";
    }

    const range = currentRange();
    const params = new URLSearchParams({
      limit: 50,
      offset: String(offset),
      run_id: $("fRunId").value,
      run_id_mode: $("fRunIdMode").value,
      route: $("fRoute").value,
      method: $("fMethod").value,
      status: $("fStatus").value,
      q: $("fQuery").value,
      only_query: $("fOnlyQuery").checked ? "1" : "0",
      since: range.since || "",
      until: range.until || "",
    });

    try {
      const res = await fetch("/console/api/requests?" + params);
      if (!res.ok) return;
      const data = await res.json();
      offset += data.rows.length;
      $("detailCount").textContent = `${offset} loaded`;

      const tbody = $("tDetailed").querySelector("tbody");
      (data.rows || []).forEach((r) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="mono" style="color:var(--text-light)">${trunc(fmtTs(r.ts, true), 19)}</td>
          <td class="mono">${esc(r.route)}</td>
          <td class="mono">${esc(r.method || "")}</td>
          <td><span class="badge ${r.success ? "success" : "error"}">${r.success ? "OK" : "ERR"} ${r.status_code || ""}</span></td>
          <td class="mono">${esc(r.run_id || "-")}</td>
          <td class="cell-truncate" title="${esc(r.query)}">${esc(r.query)}</td>
          <td class="num">${fmtSec(r.duration_ms)}</td>
          <td><button class="btn secondary" type="button" style="padding:4px 10px;font-size:12px">View</button></td>
        `;
        tr.querySelector("button").onclick = () => showModal(r.request_id);
        tbody.appendChild(tr);
      });

      if (data.rows.length < 50) $("btnMore").style.display = "none";
      else $("btnMore").style.display = "inline-block";
    } catch (e) {
      console.error("Detailed load failed", e);
    }
  }

  async function showModal(requestId) {
    try {
      const res = await fetch(
        "/console/api/request/" + encodeURIComponent(requestId),
      );
      if (!res.ok) throw new Error("failed");
      const row = await res.json();
      const parts = [];
      if (row.query) parts.push(row.query);
      if (row.params_json) {
        try {
          parts.push(JSON.stringify(JSON.parse(row.params_json), null, 2));
        } catch (e) {
          parts.push(row.params_json);
        }
      }
      renderCodeBlock($("mQuery"), parts.join("\n\n"));
      if (row.response_json) {
        try {
          const parsed = JSON.parse(row.response_json);
          renderCodeBlock($("mResponse"), JSON.stringify(parsed, null, 2));
        } catch (e) {
          renderCodeBlock($("mResponse"), row.response_json);
        }
      } else renderCodeBlock($("mResponse"), "(No response data)");
      $("modal").style.display = "flex";
    } catch (e) {
      renderCodeBlock($("mQuery"), "(Failed to load details)");
      renderCodeBlock($("mResponse"), "");
      $("modal").style.display = "flex";
    }
  }

  function closeModal() {
    const el = $("modal");
    if (el) el.style.display = "none";
  }

  function exportLogs() {
    const range = currentRange();
    const params = new URLSearchParams({
      format: $("fExportFmt").value,
      limit: "50000",
      run_id: $("fRunId").value,
      run_id_mode: $("fRunIdMode").value,
      route: $("fRoute").value,
      method: $("fMethod").value,
      status: $("fStatus").value,
      q: $("fQuery").value,
      only_query: $("fOnlyQuery").checked ? "1" : "0",
      since: range.since || "",
      until: range.until || "",
      include_params: $("fExportParams").checked ? "1" : "0",
      include_response: $("fExportResp").checked ? "1" : "0",
    });
    window.location.href = "/console/api/export?" + params;
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("tabOverview")?.addEventListener("click", () => setTab("overview"));
    $("tabEndpoints")?.addEventListener("click", () => setTab("endpoints"));
    $("tabRuns")?.addEventListener("click", () => setTab("runs"));
    $("tabLogs")?.addEventListener("click", () => setTab("logs"));

    $("btnLoad")?.addEventListener("click", () => loadDetailed(true));
    $("btnMore")?.addEventListener("click", () => loadDetailed(false));
    $("btnExport")?.addEventListener("click", exportLogs);
    $("btnModalClose")?.addEventListener("click", closeModal);

    $("fRange")?.addEventListener("change", () => {
      const isCustom = $("fRange").value === "custom";
      if ($("fSince")) $("fSince").style.display = isCustom ? "block" : "none";
      if ($("fUntil")) $("fUntil").style.display = isCustom ? "block" : "none";
    });

    loadOverview();
    setInterval(loadOverview, 30000);
    window.closeModal = closeModal;
    window.showModal = showModal;
  });
})();
