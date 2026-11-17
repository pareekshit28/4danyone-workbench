const byId = id => document.getElementById(id);
const labels = {
  queued: "Queued", inference: "Generating views", training: "Reconstructing 4DGS",
  complete: "Complete", interrupted: "Interrupted", failed: "Failed"
};
const active = new Set(["queued", "inference", "training"]);
let selected = null;
let sourceUrl;
let submitting = false;
let smplx = false;
let running = false;
let refreshInProgress = false;

function notice(message) {
  byId("notice").textContent = message;
  byId("notice").hidden = !message;
}

async function get(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(typeof payload.detail === "string" ? payload.detail : HTTPError(response.status));
  }
  return response.json();
}

function HTTPError(status) {
  return status === 413 ? "Upload is too large for this connection." : "Request failed (" + status + ").";
}

function upload(url, data, progress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.setRequestHeader("X-Workbench", "1");
    xhr.upload.onprogress = event => {
      if (progress && event.lengthComputable) progress.value = event.loaded / event.total;
    };
    xhr.onload = () => {
      let payload;
      try { payload = JSON.parse(xhr.responseText); } catch { payload = {}; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
      else reject(new Error(typeof payload.detail === "string" ? payload.detail : HTTPError(xhr.status)));
    };
    xhr.onerror = () => reject(new Error("Connection lost during upload."));
    xhr.send(data);
  });
}

function updateButtons() {
  byId("run").disabled = submitting || running || !smplx;
  byId("resume").disabled = submitting || running;
  byId("model-form").querySelector("button").disabled = submitting || running;
}

function drawJob(job) {
  const base = "/api/jobs/" + job.id + "/files/";
  const done = job.state === "complete";
  byId("status").hidden = false;
  byId("status").textContent = labels[job.state];
  byId("empty").hidden = done;
  byId("empty").textContent = active.has(job.state)
    ? "Keep this tab open to follow progress, or return later to check the job."
    : "The orbit preview will appear once reconstruction completes.";
  byId("failure").hidden = !job.error;
  byId("failure").textContent = job.error || "";
  byId("resume").hidden = !["failed", "interrupted"].includes(job.state);
  byId("orbit").hidden = !done;
  if (done && byId("orbit").getAttribute("src") !== base + "orbit") {
    byId("orbit").src = base + "orbit";
    byId("model-link").href = base + "model";
    byId("orbit-link").href = base + "orbit?download=true";
  } else if (!done && byId("orbit").hasAttribute("src")) {
    byId("orbit").removeAttribute("src");
    byId("orbit").load();
  }
  byId("downloads").hidden = !done;
  byId("logs").hidden = !job.log;
  byId("log").textContent = job.log || "";
  byId("log-link").href = base + "log?download=true";
}

async function refresh() {
  if (refreshInProgress) return;
  refreshInProgress = true;
  try {
    const data = await get("/api/jobs");
    smplx = data.smplx;
    running = data.jobs.some(job => active.has(job.state));
    byId("setup").hidden = smplx;
    byId("no-jobs").hidden = data.jobs.length > 0;
    selected ||= data.jobs[0]?.id;
    const list = byId("jobs");
    const signature = JSON.stringify([selected, data.jobs.map(({id, name, state}) => [id, name, state])]);
    if (signature !== list.dataset.signature) {
      list.dataset.signature = signature;
      list.replaceChildren(...data.jobs.map(job => {
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.className = "job secondary";
        button.setAttribute("aria-current", String(job.id === selected));
        const name = document.createElement("span");
        name.textContent = job.name;
        const state = document.createElement("span");
        state.textContent = labels[job.state];
        button.append(name, state);
        button.onclick = () => { selected = job.id; refresh(); };
        item.append(button);
        return item;
      }));
    }
    if (selected) {
      const id = selected;
      const job = await get("/api/jobs/" + id);
      if (id === selected) drawJob(job);
    }
  } catch (error) {
    notice(error.message);
  } finally {
    refreshInProgress = false;
    updateButtons();
  }
}

function showSource() {
  const file = byId("video").files[0];
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = file && URL.createObjectURL(file);
  byId("filename").textContent = file?.name || "Choose a video or drop it here";
  const player = byId("source-player");
  player.hidden = !file;
  byId("use-time").disabled = true;
  if (sourceUrl) player.src = sourceUrl;
  else player.removeAttribute("src");
}

byId("video").onchange = showSource;
byId("source-player").onloadedmetadata = () => { byId("use-time").disabled = false; };
byId("use-time").onclick = () => {
  byId("start").value = byId("source-player").currentTime.toFixed(3);
};
const drop = byId("drop");
drop.ondragover = event => { event.preventDefault(); drop.classList.add("drag"); };
drop.ondragleave = () => drop.classList.remove("drag");
drop.ondrop = event => {
  event.preventDefault();
  drop.classList.remove("drag");
  if (event.dataTransfer.files.length !== 1) return notice("Choose one video.");
  byId("video").files = event.dataTransfer.files;
  showSource();
};

async function submit(action) {
  if (submitting) return;
  submitting = true;
  updateButtons();
  notice("");
  try { await action(); }
  catch (error) { notice(error.message); }
  finally {
    submitting = false;
    byId("upload-progress").hidden = true;
    await refresh();
  }
}

byId("model-form").onsubmit = event => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  submit(() => upload("/api/smplx", data));
};
byId("job-form").onsubmit = event => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  submit(async () => {
    if (data.get("video").size > 90 * 1024 ** 2) throw new Error("Video exceeds the 90 MiB upload limit.");
    const progress = byId("upload-progress");
    progress.value = 0;
    progress.hidden = false;
    const job = await upload("/api/jobs", data, progress);
    selected = job.id;
  });
};
byId("resume").onclick = () => submit(async () => {
  await get("/api/jobs/" + selected + "/resume", {method: "POST", headers: {"X-Workbench": "1"}});
});
updateButtons();
refresh();
setInterval(refresh, 2500);
