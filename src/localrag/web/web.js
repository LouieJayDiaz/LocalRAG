const $ = (selector) => document.querySelector(selector);
const conversation = $("#conversation");
const message = $("#message");

function showError(error) {
  message.textContent = error instanceof Error ? error.message : String(error);
}

async function request(url, options = {}) {
  try {
    const response = await fetch(url, options);
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Request failed.");
    return body;
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error("LocalRAG is not connected. Keep the server running and reload this page.");
    }
    throw error;
  }
}

async function refreshState() {
  const state = await request("/api/state");
  $("#document-count").textContent = state.document_count;
  const list = $("#document-list");
  list.replaceChildren();
  state.documents.forEach((name) => {
    const item = document.createElement("li");
    item.title = name;
    item.textContent = name;
    list.append(item);
  });
}

async function refreshModels() {
  try {
    const body = await request("/api/models");
    const select = $("#model-select");
    const current = select.value;
    const detectedModels = body.providers.flatMap((provider) => provider.models || []);
    select.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = detectedModels.length ? "Select a model" : "No local models detected";
    select.append(placeholder);
    detectedModels.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.size ? `${model.name} - ${model.size}` : model.name;
      select.append(option);
    });
    if (detectedModels.some((model) => model.id === current)) select.value = current;
    const availableProviders = body.providers.filter((provider) => provider.available);
    $("#connection-status").textContent = detectedModels.length
      ? `${detectedModels.length} local model${detectedModels.length === 1 ? "" : "s"} found`
      : availableProviders.length ? "No local models detected" : "No local providers running";
    renderProviders(body.providers);
  } catch (error) {
    $("#connection-status").textContent = "Local provider scan failed";
    $("#connection-status").previousElementSibling.style.background = "#e99595";
  }
}

function renderProviders(providers) {
  const list = $("#provider-list");
  if (!list) return;
  list.replaceChildren();
  providers.forEach((provider) => {
    const card = document.createElement("div");
    card.className = "provider-card";
    const heading = document.createElement("div");
    heading.className = "provider-heading";
    heading.innerHTML = "<strong></strong><span></span>";
    heading.querySelector("strong").textContent = provider.name;
    heading.querySelector("span").textContent = provider.available
      ? `${(provider.models || []).length} detected`
      : "Not running";
    card.append(heading);
    const detail = document.createElement("p");
    detail.textContent = provider.available
      ? "These models are available to select."
      : provider.error || "Start this provider, then scan again.";
    card.append(detail);
    const link = document.createElement("a");
    link.href = provider.library;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = `Open ${provider.name} model library \u2197`;
    card.append(link);
    list.append(card);
  });
}

async function showProviderPanel() {
  $("#model-panel").hidden = false;
  $("#scan-status").textContent = "Scanning local model providers...";
  $("#provider-list").innerHTML = "<p>Checking local providers...</p>";
  try {
    await refreshModels();
    $("#scan-status").textContent = "Scan complete. Refresh after installing a model.";
  } catch (error) {
    $("#scan-status").textContent = error instanceof Error ? error.message : String(error);
  }
}

function addMessage(role, text, sources = []) {
  $("#welcome")?.remove();
  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  row.innerHTML = `<div class="message-label">${role === "user" ? "YOU" : "LOCALRAG"}</div><div class="bubble"></div>`;
  const bubble = row.querySelector(".bubble");
  if (text === null) {
    bubble.classList.add("thinking");
    bubble.innerHTML = '<span>.</span><span>.</span><span>.</span>';
  } else {
    bubble.textContent = text;
  }
  if (sources.length) {
    const details = document.createElement("details");
    details.className = "sources";
    details.innerHTML = `<summary>${sources.length} source${sources.length === 1 ? "" : "s"}</summary>`;
    sources.forEach((source, index) => {
      const sourceText = document.createElement("p");
      sourceText.textContent = `[${index + 1}] ${source.document} - chunk ${source.chunk} - ${source.content}`;
      details.append(sourceText);
    });
    row.append(details);
  }
  conversation.append(row);
  conversation.scrollTop = conversation.scrollHeight;
  return row;
}

async function uploadFiles(files, input) {

  if (!files.length) return;
  const form = new FormData();
  [...files].forEach((file) => form.append("uploads", file, file.name));
  message.textContent = "Reading and indexing documents...";
  try {
    const result = await request("/api/documents", { method: "POST", body: form });
    await refreshState();
    message.textContent = `Indexed ${result.documents_added} file${result.documents_added === 1 ? "" : "s"} (${result.chunks_added} chunks).`;
  } catch (error) { showError(error); }
  input.value = "";
}

$("#file-input").addEventListener("change", (event) => uploadFiles(event.target.files, event.target));
$("#folder-input").addEventListener("change", (event) => uploadFiles(event.target.files, event.target));

$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;
  const model = $("#model-select").value.trim();
  if (!model) { showError("Scan for a model and select one first."); return; }
  addMessage("user", question);
  $("#question").value = "";
  $("#question").style.height = "auto";
  $("#send-button").disabled = true;
  message.textContent = "Searching your documents and asking your local model...";
  const thinkingRow = addMessage("assistant", null);
  try {
    const result = await request("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, model }),
    });
    thinkingRow.remove();
    addMessage("assistant", result.answer, result.sources);
    message.textContent = "";
  } catch (error) {
    thinkingRow.remove();
    showError(error);
  }
  $("#send-button").disabled = false;
});


$("#clear-button").addEventListener("click", async () => {
  if (!confirm("Remove all indexed documents?")) return;
  try {
    await request("/api/clear", { method: "POST" });
    await refreshState();
    conversation.innerHTML = '<div id="welcome" class="welcome"><div class="welcome-mark">\u2726</div><h2>Index cleared.</h2><p>Upload documents to start a new chat.</p></div>';
  } catch (error) { showError(error); }
});

$("#refresh-models").addEventListener("click", refreshModels);
$("#close-model-panel").addEventListener("click", () => { $("#model-panel").hidden = true; });

$("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    if (event.ctrlKey || event.metaKey) {
      event.preventDefault();
      $("#chat-form").requestSubmit();
    }
  }
});

$("#question").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 130)}px`;
});


Promise.all([refreshState(), refreshModels()]).catch(showError);
