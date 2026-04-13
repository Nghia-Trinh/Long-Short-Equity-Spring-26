const STORAGE_KEY = "investment-theses-v1";

const messageList = document.getElementById("messageList");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const thesisForm = document.getElementById("thesisForm");
const thesisList = document.getElementById("thesisList");
const newChatBtn = document.getElementById("newChatBtn");

const fieldMap = {
  company: document.getElementById("companyInput"),
  ticker: document.getElementById("tickerInput"),
  horizon: document.getElementById("horizonInput"),
  thesis: document.getElementById("thesisInput"),
  catalysts: document.getElementById("catalystsInput"),
  risks: document.getElementById("risksInput"),
};

const state = {
  theses: loadTheses(),
};

function loadTheses() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.error("Unable to read theses from storage", error);
    return [];
  }
}

function saveTheses() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.theses));
}

function now() {
  return new Date().toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function addMessage(text, role = "assistant") {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;

  const tag = document.createElement("span");
  tag.className = "message-role";
  tag.textContent = role === "assistant" ? "Assistant" : "You";

  const content = document.createElement("p");
  content.textContent = text;

  wrapper.append(tag, content);
  messageList.appendChild(wrapper);
  messageList.scrollTop = messageList.scrollHeight;
}

function renderTheses() {
  thesisList.innerHTML = "";
  if (state.theses.length === 0) {
    const emptyState = document.createElement("p");
    emptyState.className = "empty-state";
    emptyState.textContent = "No theses yet. Add one from chat or the form.";
    thesisList.appendChild(emptyState);
    return;
  }

  state.theses.forEach((item) => {
    const card = document.createElement("article");
    card.className = "thesis-card";
    card.dataset.id = item.id;

    const heading = document.createElement("div");
    heading.className = "thesis-heading";

    const title = document.createElement("h3");
    title.textContent = item.company || "Unnamed company";

    const ticker = document.createElement("span");
    ticker.className = "ticker-pill";
    ticker.textContent = item.ticker ? item.ticker.toUpperCase() : "N/A";

    heading.append(title, ticker);

    const thesis = document.createElement("p");
    thesis.textContent = item.thesis;

    const meta = document.createElement("p");
    meta.className = "thesis-meta";
    meta.textContent = `Horizon: ${item.horizon || "Not set"} | Added: ${item.addedAt}`;

    const catalysts = document.createElement("p");
    catalysts.className = "thesis-details";
    const catalystsLabel = document.createElement("strong");
    catalystsLabel.textContent = "Catalysts:";
    catalysts.append(catalystsLabel, ` ${item.catalysts || "Not set"}`);

    const risks = document.createElement("p");
    risks.className = "thesis-details";
    const risksLabel = document.createElement("strong");
    risksLabel.textContent = "Risks:";
    risks.append(risksLabel, ` ${item.risks || "Not set"}`);

    const actions = document.createElement("div");
    actions.className = "thesis-actions";

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "thesis-delete";
    deleteBtn.textContent = "Delete";
    deleteBtn.addEventListener("click", () => {
      state.theses = state.theses.filter((entry) => entry.id !== item.id);
      saveTheses();
      renderTheses();
      addMessage(`Removed thesis for ${item.company}.`, "assistant");
    });

    actions.append(deleteBtn);
    card.append(heading, thesis, meta, catalysts, risks, actions);
    thesisList.appendChild(card);
  });
}

function normalizePayload(payload) {
  return {
    id: crypto.randomUUID(),
    company: (payload.company || "").trim(),
    ticker: (payload.ticker || "").trim().toUpperCase(),
    horizon: (payload.horizon || "").trim(),
    thesis: (payload.thesis || "").trim(),
    catalysts: (payload.catalysts || "").trim(),
    risks: (payload.risks || "").trim(),
    addedAt: now(),
  };
}

function addThesis(payload) {
  const normalized = normalizePayload(payload);
  if (!normalized.company || !normalized.thesis) {
    addMessage("To save a thesis, include at least company and thesis.", "assistant");
    return false;
  }

  state.theses.unshift(normalized);
  saveTheses();
  renderTheses();
  addMessage(
    `Added thesis for ${normalized.company}${normalized.ticker ? ` (${normalized.ticker})` : ""}.`,
    "assistant",
  );
  return true;
}

function parseAddCommand(input) {
  const command = input.replace(/^\/add/i, "").trim();
  if (!command) return null;

  const pairs = command.split(";").map((chunk) => chunk.trim()).filter(Boolean);
  if (pairs.length === 0) return null;

  const payload = {};
  pairs.forEach((pair) => {
    const [rawKey, ...rest] = pair.split(":");
    if (!rawKey || rest.length === 0) return;
    const key = rawKey.trim().toLowerCase();
    const value = rest.join(":").trim();
    payload[key] = value;
  });
  return payload;
}

function assistantResponse(text) {
  if (/^\/add/i.test(text)) {
    const payload = parseAddCommand(text);
    if (!payload) {
      addMessage(
        "Use /add with key-value fields, e.g. /add company: ACME; thesis: ...",
        "assistant",
      );
      return;
    }
    addThesis(payload);
    return;
  }

  if (/risk|downside|bear/i.test(text)) {
    addMessage(
      "For risk analysis: list thesis breakers, estimate probability-weighted downside, and define one stop-loss condition.",
      "assistant",
    );
    return;
  }

  if (/catalyst|trigger|event/i.test(text)) {
    addMessage(
      "Prioritize 2-3 dated catalysts (earnings, regulatory approvals, capacity expansions) and tie each to expected valuation impact.",
      "assistant",
    );
    return;
  }

  addMessage(
    "I can help draft or refine theses. Use /add company: ...; thesis: ... to save directly.",
    "assistant",
  );
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  addMessage(text, "user");
  chatInput.value = "";
  chatInput.style.height = "auto";
  assistantResponse(text);
});

chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 220)}px`;
});

thesisForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const payload = {
    company: fieldMap.company.value,
    ticker: fieldMap.ticker.value,
    horizon: fieldMap.horizon.value,
    thesis: fieldMap.thesis.value,
    catalysts: fieldMap.catalysts.value,
    risks: fieldMap.risks.value,
  };
  if (addThesis(payload)) {
    thesisForm.reset();
  }
});

document.querySelectorAll(".chip-btn").forEach((button) => {
  button.addEventListener("click", () => {
    const text = button.dataset.chip || "";
    chatInput.value = text;
    chatInput.dispatchEvent(new Event("input"));
    chatInput.focus();
  });
});

newChatBtn.addEventListener("click", () => {
  messageList.innerHTML = "";
  addMessage("New chat started. Ask me to draft or /add a thesis.", "assistant");
});

addMessage(
  "Welcome. Add additional investment theses from this chat or the form on the right.",
  "assistant",
);
renderTheses();
