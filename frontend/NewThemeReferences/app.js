"use strict";

const PIECES = {
  wK: "♔", wQ: "♕", wR: "♖", wB: "♗", wN: "♘", wP: "♙",
  bK: "♚", bQ: "♛", bR: "♜", bB: "♝", bN: "♞", bP: "♟",
};

const VARIANT_DESCRIPTIONS = {
  standard: "Classic chess.",
  chess960: "A randomized back rank with normal chess rules.",
  crazyhouse: "Captured pieces can be dropped back onto the board.",
  threecheck: "Win by checking the opponent three times.",
  antichess: "Captures are compulsory; lose every piece to win.",
  atomic: "Captures explode the surrounding pieces.",
  horde: "White's pawn horde tries to overwhelm Black's army.",
  kingofthehill: "Move your king into one of the four center squares.",
  racingkings: "Race your king safely to the other side first.",
};

const matchPath = window.location.pathname.match(/^\/play\/([A-Za-z0-9]+)\/?$/);
const isMatchMode = Boolean(matchPath);
const matchId = matchPath?.[1] || null;
let matchToken = isMatchMode ? sessionStorage.getItem(`dinnerbone-match-${matchId}`) : null;
document.body.classList.toggle("match-mode", isMatchMode);
const guestNameKey = "dinnerbone-guest-name";

function loadGuestName() {
  let value = "";
  try { value = localStorage.getItem(guestNameKey) || ""; } catch { /* private storage */ }
  if (!value) {
    const suffix = crypto.getRandomValues(new Uint16Array(1))[0].toString(36).toUpperCase().padStart(3, "0").slice(-3);
    value = `Guest ${suffix}`;
    try { localStorage.setItem(guestNameKey, value); } catch { /* private storage */ }
  }
  return value;
}

const $ = (selector) => document.querySelector(selector);
const boardElement = $("#board");
const moveList = $("#moveList");
const statusMessage = $("#statusMessage");
const engineMoveButton = $("#engineMoveButton");
const setupDialog = $("#setupDialog");
const setupForm = $("#setupForm");
const promotionDialog = $("#promotionDialog");
const promotionChoices = $("#promotionChoices");
const saveDialog = $("#saveDialog");
const movePanel = $(".move-panel");
const boardColumn = $(".board-column");
const boardStage = $(".board-stage");
const boardLower = $(".board-lower");
const evalTrack = $(".eval-track");
const stackedLayout = window.matchMedia("(max-width: 900px)");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

let state = null;
let eventSource = null;
let renderQueued = false;
let selectedSource = null;
let draggingSource = null;
let pendingPromotion = null;
let pendingPremove = null;
let selectedDrop = null;
let premoveSubmitting = false;
let createdMatch = null;
let setupKind = null;
let renderedBoardKey = "";
let renderedMovesKey = "";
let renderedEngineKey = "";
let renderedGuideKey = "";
let stateReceivedAt = performance.now();
let personalProfile = null;
let matchBoardFlipped = null;
let accountState = null;
let pendingOfferKind = null;
let guestName = loadGuestName();
let introFinished = false;

function shouldShowIntro(parameters) {
  return !isMatchMode
    && !parameters.has("auth_error")
    && !parameters.has("signed_in")
    && !parameters.has("invite");
}

function playSiteIntro() {
  if (introFinished) return Promise.resolve();
  introFinished = true;
  const intro = $("#siteIntro");
  if (!intro) return Promise.resolve();
  document.body.classList.add("intro-active");
  intro.classList.add("playing");
  const duration = reducedMotion.matches ? 2000 : 7200;
  return new Promise((resolve) => {
    let completed = false;
    const finish = () => {
      if (completed) return;
      completed = true;
      intro.hidden = true;
      document.body.classList.remove("intro-active");
      resolve();
    };
    intro.addEventListener("animationend", (event) => {
      if (event.animationName === "intro-garage-door") finish();
    });
    window.setTimeout(finish, duration);
  });
}

function syncBoardSize() {
  const viewportHeight = window.visualViewport?.height || window.innerHeight;
  const stageStyle = window.getComputedStyle(boardStage);
  const columnGap = Number.parseFloat(stageStyle.columnGap) || 0;
  const widthBudget = boardStage.clientWidth - evalTrack.offsetWidth - columnGap;
  const stageTop = boardStage.getBoundingClientRect().top;
  const bottomReserve = boardLower.offsetHeight + 8;
  const heightBudget = viewportHeight - stageTop - bottomReserve;
  const size = Math.floor(Math.min(widthBudget, heightBudget));
  if (size > 160) boardColumn.style.setProperty("--board-size", `${size}px`);
}

function syncSidebarHeight() {
  if (stackedLayout.matches) {
    movePanel.style.removeProperty("--board-column-height");
    return;
  }
  const height = Math.ceil(boardColumn.getBoundingClientRect().height);
  if (height > 0) movePanel.style.setProperty("--board-column-height", `${height}px`);
}

if (window.ResizeObserver) {
  const layoutObserver = new ResizeObserver(() => {
    syncBoardSize();
    syncSidebarHeight();
  });
  layoutObserver.observe(boardColumn);
  layoutObserver.observe(boardStage);
}
function syncLayout() {
  syncBoardSize();
  window.requestAnimationFrame(syncSidebarHeight);
}
stackedLayout.addEventListener("change", syncLayout);
window.addEventListener("resize", syncLayout);
window.visualViewport?.addEventListener("resize", syncLayout);
window.requestAnimationFrame(syncLayout);

function squareName(index) {
  return String.fromCharCode(97 + (index & 7)) + String((index >> 3) + 1);
}

function squareIndex(name) {
  return (Number(name[1]) - 1) * 8 + name.charCodeAt(0) - 97;
}

function formatNumber(value) {
  return new Intl.NumberFormat("en", {
    notation: value >= 1_000_000 ? "compact" : "standard",
    maximumFractionDigits: 1,
  }).format(value || 0);
}

function scoreText(cp) {
  if (Math.abs(cp) > 20_000) return cp > 0 ? "WIN" : "LOSS";
  return `${cp >= 0 ? "+" : ""}${(cp / 100).toFixed(2)}`;
}

function setConnection(online, label) {
  const element = $("#connectionState");
  element.classList.toggle("online", online);
  element.lastChild.textContent = ` ${label}`;
}

function scheduleRender(nextState) {
  if (nextState && (!state || nextState.version >= state.version)) {
    if (isMatchMode) {
      if (matchBoardFlipped === null) matchBoardFlipped = Boolean(nextState.flipped);
      nextState.flipped = matchBoardFlipped;
    }
    state = nextState;
    stateReceivedAt = performance.now();
  }
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    if (state) render();
  });
}

async function api(path, body) {
  let endpoint = path;
  let requestBody = body;
  if (isMatchMode && path === "/api/move") {
    endpoint = "/api/match/move";
    requestBody = { ...body, room_id: matchId, token: matchToken };
  } else if (isMatchMode && path === "/api/action") {
    endpoint = "/api/match/action";
    requestBody = { ...body, room_id: matchId, token: matchToken };
  }
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  scheduleRender(payload);
  return payload;
}

function displayedSquares() {
  const files = state.flipped ? [7, 6, 5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5, 6, 7];
  const ranks = state.flipped ? [0, 1, 2, 3, 4, 5, 6, 7] : [7, 6, 5, 4, 3, 2, 1, 0];
  return ranks.flatMap((rank) => files.map((file) => rank * 8 + file));
}

function legalFrom(square) {
  return interactionMoves().some((move) => move.startsWith(square));
}

function legalTargets(source) {
  return new Set(interactionMoves().filter((move) => move.startsWith(source)).map((move) => move.slice(2, 4)));
}

function interactionMoves() {
  if (!isMatchMode || state.can_move) return state.legal_moves || [];
  return state.can_premove ? (state.premove_moves || []) : [];
}

function interactionSide() {
  return isMatchMode ? state.player_side : state.side;
}

function renderBoard() {
  const last = state.moves.at(-1) || "";
  const key = [state.position_key, state.flipped, state.can_move, state.can_premove, state.side, state.player_side, last].join(":");
  if (key === renderedBoardKey) {
    updateBoardHighlights();
    return;
  }
  renderedBoardKey = key;
  selectedSource = null;
  boardElement.replaceChildren();
  const bottomRank = state.flipped ? 7 : 0;
  const leftFile = state.flipped ? 7 : 0;

  for (const index of displayedSquares()) {
    const name = squareName(index);
    const file = index & 7;
    const rank = index >> 3;
    const code = state.pieces[index];
    const movable = Boolean((state.can_move || state.can_premove) && code && code[0] === interactionSide()?.[0] && legalFrom(name));
    const square = document.createElement("button");
    square.type = "button";
    square.className = `square ${(file + rank) % 2 ? "light" : "dark"}`;
    square.dataset.square = name;
    square.setAttribute("role", "gridcell");
    square.setAttribute("aria-label", `${name}${code ? ` ${code}` : " empty"}`);
    if (movable) square.classList.add("movable");
    const isLastMoveSquare = last && (
      last.slice(0, 2) === name || last.slice(2, 4) === name
    );
    if (isLastMoveSquare) {
      square.classList.add("last-move");
      const marker = document.createElement("span");
      marker.className = "last-move-marker";
      marker.setAttribute("aria-hidden", "true");
      square.append(marker);
    }

    if (code) {
      const piece = document.createElement("span");
      piece.className = `piece ${code[0] === "w" ? "white" : "black"}`;
      piece.textContent = PIECES[code];
      piece.draggable = false;
      square.append(piece);
      square.draggable = movable;
    }
    if (rank === bottomRank) {
      const coordinate = document.createElement("span");
      coordinate.className = "coordinate file";
      coordinate.textContent = name[0];
      square.append(coordinate);
    }
    if (file === leftFile) {
      const coordinate = document.createElement("span");
      coordinate.className = "coordinate rank";
      coordinate.textContent = name[1];
      square.append(coordinate);
    }

    square.addEventListener("click", onSquareClick);
    square.addEventListener("dragstart", onDragStart);
    square.addEventListener("dragover", onDragOver);
    square.addEventListener("dragleave", () => square.classList.remove("drag-over"));
    square.addEventListener("drop", onDrop);
    square.addEventListener("dragend", onDragEnd);
    boardElement.append(square);
  }
  updateBoardHighlights();
}

function updateBoardHighlights() {
  const targets = selectedSource ? legalTargets(selectedSource) : new Set();
  for (const square of boardElement.querySelectorAll(".square")) {
    const name = square.dataset.square;
    square.classList.toggle("selected", name === selectedSource);
    square.classList.toggle("legal", targets.has(name));
    square.classList.toggle("capture", targets.has(name) && Boolean(state.pieces[squareIndex(name)]));
    square.classList.toggle("premove-source", Boolean(pendingPremove && pendingPremove.slice(0, 2) === name));
    square.classList.toggle("premove-target", Boolean(pendingPremove && pendingPremove.slice(2, 4) === name));
  }
}

function onSquareClick(event) {
  if (draggingSource) return;
  const target = event.currentTarget.dataset.square;
  if (selectedDrop) return attemptDrop(target);
  if (selectedSource && legalTargets(selectedSource).has(target)) return attemptMove(selectedSource, target);
  selectedSource = (state.can_move || state.can_premove) && legalFrom(target) ? target : null;
  updateBoardHighlights();
}

function onDragStart(event) {
  const source = event.currentTarget.dataset.square;
  if ((!state.can_move && !state.can_premove) || !legalFrom(source)) return event.preventDefault();
  draggingSource = source;
  selectedSource = source;
  event.dataTransfer.effectAllowed = "move";
  event.dataTransfer.setData("text/plain", source);
  updateBoardHighlights();
}

function onDragOver(event) {
  const source = draggingSource || event.dataTransfer.getData("text/plain");
  const target = event.currentTarget.dataset.square;
  if (source && legalTargets(source).has(target)) {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    event.currentTarget.classList.add("drag-over");
  }
}

function onDrop(event) {
  event.preventDefault();
  const source = draggingSource || event.dataTransfer.getData("text/plain");
  const target = event.currentTarget.dataset.square;
  event.currentTarget.classList.remove("drag-over");
  if (source) attemptMove(source, target);
}

function onDragEnd() {
  window.setTimeout(() => { draggingSource = null; }, 0);
  boardElement.querySelectorAll(".drag-over").forEach((square) => square.classList.remove("drag-over"));
}

function attemptMove(source, target) {
  const stem = source + target;
  const candidates = interactionMoves().filter((move) => move.startsWith(stem));
  if (!candidates.length) return;
  selectedSource = null;
  updateBoardHighlights();
  if (candidates.length === 1) return state.can_move ? submitMove(candidates[0]) : queuePremove(candidates[0]);

  pendingPromotion = stem;
  promotionChoices.replaceChildren();
  for (const suffix of ["q", "r", "b", "n"]) {
    if (!candidates.includes(stem + suffix)) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "promotion-choice";
    button.textContent = PIECES[interactionSide()[0] + suffix.toUpperCase()];
    button.addEventListener("click", () => {
      promotionDialog.close();
      if (state.can_move) submitMove(pendingPromotion + suffix);
      else queuePremove(pendingPromotion + suffix);
      pendingPromotion = null;
    });
    promotionChoices.append(button);
  }
  promotionDialog.showModal();
}

function attemptDrop(target) {
  const move = `${selectedDrop}@${target}`;
  if (!interactionMoves().includes(move)) return;
  selectedDrop = null;
  renderPockets();
  if (state.can_move) submitMove(move);
  else queuePremove(move);
}

function queuePremove(move) {
  pendingPremove = move;
  selectedSource = null;
  selectedDrop = null;
  $("#premoveText").textContent = move;
  $("#premoveBar").hidden = false;
  updateBoardHighlights();
  renderPockets();
  statusMessage.textContent = `Premove ${move} is ready.`;
}

function cancelPremove(message = "Premove cancelled.") {
  pendingPremove = null;
  $("#premoveBar").hidden = true;
  updateBoardHighlights();
  if (message) statusMessage.textContent = message;
}

function maybePlayPremove() {
  if (!pendingPremove || !state.can_move || premoveSubmitting) return;
  const move = pendingPremove;
  pendingPremove = null;
  $("#premoveBar").hidden = true;
  if (!(state.legal_moves || []).includes(move)) {
    statusMessage.textContent = "That premove is no longer legal.";
    updateBoardHighlights();
    return;
  }
  premoveSubmitting = true;
  submitMove(move).finally(() => { premoveSubmitting = false; });
}

async function submitMove(move) {
  try { await api("/api/move", { move }); }
  catch (error) { statusMessage.textContent = error.message; }
}

function renderMoves() {
  const key = JSON.stringify(state.full_moves);
  if (key === renderedMovesKey) return;
  renderedMovesKey = key;
  moveList.replaceChildren();
  if (!state.full_moves.length) {
    const empty = document.createElement("p");
    empty.className = "empty-copy";
    empty.textContent = "The score sheet is empty.";
    moveList.append(empty);
    return;
  }
  state.full_moves.forEach((row, index) => {
    const element = document.createElement("div");
    element.className = `move-row${index === state.full_moves.length - 1 ? " current" : ""}`;
    for (const [className, text] of [["number", `${row.number}.`], ["", row.white], ["", row.black]]) {
      const cell = document.createElement("span");
      cell.className = className;
      cell.textContent = text;
      element.append(cell);
    }
    moveList.append(element);
  });
  moveList.scrollTop = moveList.scrollHeight;
}

function renderEngine() {
  const latest = state.latest;
  const key = JSON.stringify([latest, state.snapshots]);
  if (key === renderedEngineKey) return;
  renderedEngineKey = key;
  $("#sourceBadge").textContent = latest?.book ? "BOOK" : latest?.cache_replay ? "TT REUSE" : (latest?.source || state.engine_name || "TREE").toUpperCase();
  const metrics = [
    ["BEST", latest?.best_move || "—"], ["DEPTH", latest ? String(latest.depth) : "—"],
    ["NODES", latest ? formatNumber(latest.nodes) : "—"], ["NPS", latest?.cache_replay ? "CACHED" : latest ? formatNumber(latest.nps) : "—"],
    ["MAX WORKERS", String(state.engine_threads || 1)],
  ];
  const metricRoot = $("#engineMetrics");
  metricRoot.replaceChildren();
  for (const [label, value] of metrics) {
    const item = document.createElement("div");
    const caption = document.createElement("span");
    const strong = document.createElement("strong");
    caption.textContent = label;
    strong.textContent = value;
    item.append(caption, strong);
    metricRoot.append(item);
  }

  const root = $("#lineList");
  root.replaceChildren();
  const lines = [];
  const seen = new Set();
  for (const snapshot of state.snapshots || []) {
    const pv = snapshot.pv?.join(" ") || "";
    if (!pv || seen.has(pv)) continue;
    seen.add(pv);
    lines.push(snapshot);
  }
  if (!lines.length && latest?.pv?.length) lines.push(latest);
  if (!lines.length) {
    const empty = document.createElement("p");
    empty.className = "empty-copy";
    empty.textContent = state.configured ? "Warming the search tree…" : "Choose a session to begin.";
    root.append(empty);
    return;
  }
  for (const line of lines) {
    const row = document.createElement("div");
    row.className = "engine-line";
    const relative = state.side === "white" ? line.score : -line.score;
    for (const [className, value] of [["depth", line.book ? "BOOK" : `D${line.depth}`], ["score", scoreText(relative)], ["pv", line.pv.join(" ")]]) {
      const span = document.createElement("span");
      span.className = className;
      span.textContent = value;
      row.append(span);
    }
    root.append(row);
  }
}

function renderGuide() {
  const guide = state.guide || {};
  const key = JSON.stringify(guide);
  if (key === renderedGuideKey) return;
  renderedGuideKey = key;
  $("#guideSummary").textContent = guide.opening || "Current position";
  const currentRoot = $("#currentMoveGuide");
  currentRoot.replaceChildren();
  const current = guide.current_move || {};
  const currentItem = document.createElement("p");
  currentItem.textContent = current.text || "No engine move has been played yet.";
  currentRoot.append(currentItem);

  const ideasRoot = $("#lineIdeaGuide");
  ideasRoot.replaceChildren();
  for (const idea of guide.line_ideas || []) {
    const item = document.createElement("p");
    item.textContent = idea;
    ideasRoot.append(item);
  }
}

function renderEvaluation() {
  const cp = state.white_score || 0;
  const whitePercent = 50 + 48 * Math.tanh(cp / 420);
  $("#evalFill").style.height = `${whitePercent}%`;
  $("#evalLabel").textContent = scoreText(cp);
}

function formatClock(seconds, unlimited = false) {
  if (unlimited) return "∞";
  const safe = Math.max(0, seconds || 0);
  if (safe < 10) return safe.toFixed(1);
  const whole = Math.ceil(safe);
  const minutes = Math.floor(whole / 60);
  return `${minutes}:${String(whole % 60).padStart(2, "0")}`;
}

function liveClock(side) {
  let seconds = state.clocks?.[side] || 0;
  if (!state.clocks?.unlimited && state.clocks?.active === side && !state.terminal) {
    seconds -= (performance.now() - stateReceivedAt) / 1000;
  }
  return Math.max(0, seconds);
}

function liveInviteSeconds() {
  if (!state?.invite?.expires_at) return 0;
  return Math.max(0, state.invite.expires_at - Date.now() / 1000);
}

function updateMatchClocks() {
  if (!isMatchMode || !state?.clocks) return;
  for (const side of ["white", "black"]) {
    const value = liveClock(side);
    $(`#${side}Clock`).textContent = formatClock(value, state.clocks.unlimited);
    $(`#${side}Player`).classList.toggle("low", !state.clocks.unlimited && value < 20);
  }
}

function renderPockets() {
  if (!isMatchMode) return;
  for (const side of ["white", "black"]) {
    const root = $(`#${side}Pocket`);
    const pieces = state.pockets?.[side] || [];
    root.hidden = state.variant !== "crazyhouse";
    root.replaceChildren();
    const counts = pieces.reduce((result, piece) => {
      result[piece] = (result[piece] || 0) + 1;
      return result;
    }, {});
    for (const [piece, count] of Object.entries(counts)) {
      const button = document.createElement("button");
      button.type = "button";
      button.title = `Drop ${piece}${count > 1 ? ` (${count} available)` : ""}`;
      button.textContent = `${PIECES[side[0] + piece]}${count > 1 ? `×${count}` : ""}`;
      button.disabled = side !== state.player_side || (!state.can_move && !state.can_premove);
      button.classList.toggle("selected", selectedDrop === piece);
      button.addEventListener("click", () => {
        selectedSource = null;
        selectedDrop = selectedDrop === piece ? null : piece;
        renderPockets();
        updateBoardHighlights();
      });
      root.append(button);
    }
  }
}

function renderMatch() {
  $("#matchPlayers").hidden = false;
  for (const side of ["white", "black"]) {
    const mine = state.player_side === side;
    const name = state.player_names?.[side] || (mine ? "You" : "Friend");
    const rating = state.player_ratings?.[side];
    const change = state.rating_updates?.[side]?.change;
    const ratingText = rating == null ? "" : ` · ${rating} BR${change == null ? "" : ` ${change >= 0 ? "+" : ""}${change}`}`;
    $(`#${side}PlayerName`).textContent = `TIME FOR ${name.toUpperCase()} · ${side.toUpperCase()}${ratingText}${mine ? " · YOU" : ""}`;
    const player = $(`#${side}Player`);
    player.hidden = false;
    player.classList.toggle("joined", Boolean(state.players?.[side]));
    player.classList.toggle("active", state.match_started && !state.terminal && state.side === side);
  }
  $("#boardTitle").textContent = state.variant_name.toUpperCase();
  $(".board-heading .section-number").textContent = "02";
  $(".interaction-hint").textContent = `${state.message} ${state.variant_description}`;
  $("#footerAbout").textContent = state.auto_saved
    ? "Completed match · saved to signed-in player history"
    : "Private casual match · no engine or analysis tools";
  $("#flipButton").textContent = "Flip";
  $("#drawButton").hidden = !state.can_resign;
  $("#resignButton").hidden = !state.can_resign;
  $("#rematchButton").hidden = !state.terminal;
  $("#rematchButton").disabled = Boolean(state.rematch?.mine && !state.rematch?.ready);
  $("#rematchButton").textContent = state.rematch?.ready
    ? "Enter rematch"
    : state.rematch?.mine
      ? "Rematch requested"
      : state.rematch?.opponent
        ? "Accept rematch"
        : "Rematch";
  $("#drawButton").textContent = state.draw_offer === "sent" ? "Cancel draw" : "Offer draw";

  const offerBar = $("#matchOfferBar");
  if (state.undo_request === "received") {
    pendingOfferKind = "undo";
    $("#matchOfferText").textContent = "Your opponent requests a takeback.";
    offerBar.hidden = false;
  } else if (state.draw_offer === "received") {
    pendingOfferKind = "draw";
    $("#matchOfferText").textContent = "Your opponent offers a draw.";
    offerBar.hidden = false;
  } else {
    pendingOfferKind = null;
    offerBar.hidden = true;
  }
  renderPockets();
  updateMatchClocks();
}

function renderHeader() {
  const outcome = state.outcome;
  const inviteOpen = Boolean(state.invite?.open && liveInviteSeconds() > 0);
  for (const element of [$("#modeBadge"), $("#turnBadge"), $("#searchBadge")]) {
    element.classList.toggle("result", Boolean(outcome));
  }
  if (isMatchMode) {
    $("#modeBadge").textContent = outcome
      ? outcome.title.toUpperCase()
      : `${state.variant_name.toUpperCase()} · ONLINE`;
    $("#turnBadge").textContent = outcome
      ? `${outcome.result} · ${outcome.reason_label.toUpperCase()}`
      : state.waiting
        ? "WAITING FOR FRIEND"
        : `${state.side.toUpperCase()} TO MOVE${state.in_check ? " · CHECK" : ""}`;
    $("#searchBadge").textContent = outcome
      ? (state.auto_saved ? "GAME OVER · SAVED" : "GAME OVER")
      : state.waiting
        ? inviteOpen
          ? `INVITE OPEN · ${Math.ceil(liveInviteSeconds() / 60)} MIN LEFT`
          : "INVITE EXPIRED"
        : state.variant_status || "LIVE MATCH";
    $("#inviteButton").textContent = inviteOpen ? "Copy invite link" : "New invite";
    statusMessage.textContent = state.message;
    $("#undoButton").disabled = !state.can_undo;
    $("#undoButton").title = state.undo_request === "sent" ? "Cancel takeback request" : "Request a takeback";
    return;
  }
  let mode = state.mode === "analysis" ? "ANALYSIS" : `YOU ${state.player_side.toUpperCase()}`;
  if (state.opponent === "personality") mode += " / PERSONALITY";
  if (state.opponent === "elo") mode += ` / ~${state.estimated_elo}`;
  if (state.engine_id !== "dinnerbone") {
    mode += ` / ${state.engine_name.toUpperCase()}`;
    if (state.estimated_elo) mode += ` ~${state.estimated_elo}`;
  }
  mode += " · GUEST";
  $("#modeBadge").textContent = outcome ? outcome.title.toUpperCase() : mode;
  $("#turnBadge").textContent = outcome
    ? `${outcome.result} · ${outcome.reason_label.toUpperCase()}`
    : `${state.side.toUpperCase()} TO MOVE${state.in_check ? " / CHECK" : ""}`;
  $("#searchBadge").textContent = outcome ? "GAME OVER" : state.latest?.book ? "BOOK READY" : state.analysis_active ? "CALCULATING" : "TREE READY";
  statusMessage.textContent = state.message;
  engineMoveButton.disabled = !state.can_play_engine || (!state.latest && !state.legal_moves.length);
  $("#undoButton").disabled = !state.can_undo;
  $("#copyFenButton").textContent = state.fen;
  updateBoardThinking();
}

function updateBoardThinking() {
  const element = $("#boardThinking");
  if (!state?.engine_thinking) {
    element.hidden = true;
    element.textContent = "";
    return;
  }
  const elapsedSinceState = (performance.now() - stateReceivedAt) / 1000;
  const remaining = Math.max(0, state.engine_move_remaining - elapsedSinceState);
  element.hidden = false;
  element.textContent = remaining > 0
    ? `${state.engine_name.toUpperCase()} MAKING A MOVE IN ${Math.ceil(remaining)}s`
    : `${state.engine_name.toUpperCase()} CHOOSING MOVE…`;
}

function render() {
  renderHeader();
  renderBoard();
  renderMoves();
  if (isMatchMode) {
    renderMatch();
    maybePlayPremove();
    return;
  }
  renderEngine();
  renderGuide();
  renderEvaluation();
  if (!$("#engineInput").options.length) refreshEngineOptions();
}

async function action(name, value = null) {
  try { await api("/api/action", { action: name, value }); }
  catch (error) { statusMessage.textContent = error.message; }
}

function showChoices() {
  setupKind = null;
  $("#sessionChoices").hidden = false;
  $("#sessionOptions").hidden = true;
  $("#setupError").textContent = "";
  personalProfile = null;
  $("#profileResult").hidden = true;
}

async function chooseSession(kind) {
  setupKind = kind;
  $("#setupError").textContent = "";
  if (kind === "invite") {
    setupDialog.close();
    openInvite();
    return;
  }
  if (kind === "white" || kind === "black") {
    const buttons = document.querySelectorAll("[data-session]");
    buttons.forEach((button) => { button.disabled = true; });
    try {
      await api("/api/configure", {
        mode: kind,
        opponent: "standard",
        think_seconds: 30,
        fen: "",
        engine_id: "dinnerbone",
      });
      setupDialog.close();
    } catch (error) {
      $("#setupError").textContent = error.message;
    } finally {
      buttons.forEach((button) => { button.disabled = false; });
    }
    return;
  }
  $("#sessionChoices").hidden = true;
  $("#sessionOptions").hidden = false;
  $("#fenField").hidden = kind !== "load";
  $("#personalityFields").hidden = kind !== "personality";
  $("#eloField").hidden = kind !== "elo";
  $("#engineFields").hidden = kind !== "engine";
  $("#profileFields").hidden = kind !== "profile";
  $("#startSessionButton").textContent = kind === "profile" ? "Analyze games" : "Start session";
  const descriptions = {
    white: "Play White against Dinnerbone at full search strength.",
    black: "Play Black; Dinnerbone will make its book move immediately.",
    load: "Paste a legal FEN and explore the position from either side.",
    personality: "Tune seven independent move-choice traits for a match as White. Dinnerbone still searches every candidate before applying the selected style.",
    elo: "Set the estimated strength for a match as White.",
    engine: "Choose an installed engine, your color, and the engine's approximate strength.",
    profile: "Import games from one color. Dinnerbone will derive and show a transparent three-metric profile.",
  };
  $("#sessionDescription").textContent = descriptions[kind];
  $("#setupError").textContent = "";
}

function refreshEngineOptions() {
  if (!state?.engines) return;
  const select = $("#engineInput");
  const previous = select.value || "stockfish";
  select.replaceChildren();
  for (const engine of state.engines) {
    const option = document.createElement("option");
    option.value = engine.id;
    option.textContent = `${engine.name}${engine.available ? "" : " (not installed)"}`;
    option.disabled = !engine.available;
    select.append(option);
  }
  if ([...select.options].some((option) => option.value === previous && !option.disabled)) select.value = previous;
  else select.selectedIndex = [...select.options].findIndex((option) => !option.disabled);
  updateEngineStrengthControl();
}

function updateEngineStrengthControl() {
  const engine = state?.engines?.find((item) => item.id === $("#engineInput").value);
  if (!engine) return;
  const slider = $("#engineEloInput");
  slider.min = engine.elo_min;
  slider.max = engine.elo_max;
  slider.value = Math.max(engine.elo_min, Math.min(engine.elo_max, Number(slider.value || 2000)));
  $("#engineEloValue").value = slider.value;
  $("#engineStrengthNote").textContent = `${engine.strength_note}.`;
}

function openSetup(required = false) {
  showChoices();
  $("#setupCloseButton").hidden = required;
  $("#setupCancelButton").hidden = required;
  if (!setupDialog.open) setupDialog.showModal();
}

function closeSetup() {
  setupDialog.close();
}

function closeDialogFromBackdrop(event) {
  if (event.target === setupDialog) closeSetup();
}

function connectEvents() {
  eventSource?.close();
  const eventPath = isMatchMode
    ? `/api/match/events?room=${encodeURIComponent(matchId)}&token=${encodeURIComponent(matchToken)}&since=${state?.version ?? -1}`
    : `/api/events?since=${state?.version ?? -1}`;
  eventSource = new EventSource(eventPath);
  eventSource.addEventListener("open", () => setConnection(true, "Live connection"));
  eventSource.addEventListener("state", (event) => scheduleRender(JSON.parse(event.data)));
  eventSource.addEventListener("error", () => setConnection(false, "Reconnecting"));
}

async function openInvite() {
  if (isMatchMode) {
    if (!state?.invite?.open || liveInviteSeconds() <= 0) {
      window.location.assign("/?invite=1");
      return;
    }
    const link = `${window.location.origin}/play/${matchId}`;
    const copied = await copyText(link, "Invite link copied.");
    if (!copied) {
      createdMatch = { room_id: matchId, path: `/play/${matchId}` };
      $("#inviteLink").value = link;
      $("#inviteOptions").hidden = true;
      $("#inviteCreated").hidden = false;
      $("#enterMatchButton").hidden = true;
      $("#inviteDialog").showModal();
    }
    return;
  }
  createdMatch = null;
  $("#inviteOptions").hidden = false;
  $("#inviteCreated").hidden = true;
  $("#inviteError").textContent = "";
  $("#enterMatchButton").hidden = false;
  $("#guestNameInput").value = accountState?.signed_in ? accountState.user.name : guestName;
  $("#guestNameField").hidden = Boolean(accountState?.signed_in);
  $("#opponentUsernameField").hidden = !accountState?.signed_in;
  $("#opponentUsernameInput").value = "";
  $("#inviteDialog").showModal();
}

function closeInvite() {
  $("#inviteDialog").close();
}

async function copyText(value, successMessage) {
  try {
    await navigator.clipboard.writeText(value);
    statusMessage.textContent = successMessage;
    if (isMatchMode) $(".interaction-hint").textContent = successMessage;
    return true;
  } catch {
    statusMessage.textContent = "Clipboard access was blocked. Select and copy the link manually.";
    return false;
  }
}

function ratingChangeText(game) {
  if (!game.rated || game.rating_change == null) return "UNRATED";
  return `${game.rating_change >= 0 ? "+" : ""}${game.rating_change} BR · ${game.rating_after}`;
}

function makeHistoryRow(game, { favorite = false, publicView = false } = {}) {
  const row = document.createElement("article");
  row.className = "account-game";
  let leading;
  if (favorite) {
    leading = document.createElement("button");
    leading.type = "button";
    leading.className = `favorite-button${game.favorite ? " active" : ""}`;
    leading.title = game.favorite ? "Remove from favorites" : "Add to favorites";
    leading.textContent = game.favorite ? "★" : "☆";
    leading.addEventListener("click", () => setFavorite(game.id, !game.favorite));
  } else {
    leading = document.createElement("span");
    leading.className = "history-mark";
    leading.textContent = game.perspective === "win" ? "W" : game.perspective === "loss" ? "L" : "D";
  }
  const details = document.createElement("div");
  const title = document.createElement("h4");
  title.textContent = `${game.variant_name} vs ${game.opponent_name}`;
  const description = document.createElement("p");
  const color = game.my_color || game.player_color;
  description.textContent = `${game.time_control} · ${color.toUpperCase()} · ${game.termination} · ${ratingChangeText(game)}`;
  const download = document.createElement("a");
  download.className = "text-button";
  download.href = `${publicView ? "/api/history/game" : "/api/account/game"}?id=${encodeURIComponent(game.id)}`;
  download.textContent = "Download PGN";
  details.append(title, description, download);
  const result = document.createElement("div");
  result.className = "game-result";
  const resultText = document.createElement("strong");
  resultText.textContent = `${game.perspective.toUpperCase()} · ${game.result}`;
  const ended = document.createElement("time");
  ended.dateTime = new Date(game.ended_at * 1000).toISOString();
  ended.textContent = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(game.ended_at * 1000);
  result.append(resultText, ended);
  row.append(leading, details, result);
  return row;
}

function renderAccount() {
  if (!accountState) return;
  $("#accountButton").textContent = accountState.signed_in ? accountState.user.name : "Sign in";
  $("#accountSignedOut").hidden = accountState.signed_in;
  $("#accountSignedIn").hidden = !accountState.signed_in;
  $("#googleSignInButton").disabled = !accountState.google_available;
  $("#googleSetupHint").textContent = accountState.google_available ? "Completed online games will be linked to this account on this Dinnerbone server." : accountState.google_setup_hint;
  if (!accountState.signed_in) return;
  $("#accountName").textContent = accountState.user.setup_required ? "Choose a username" : accountState.user.name;
  $("#accountEmail").textContent = accountState.user.email;
  $("#accountRating").textContent = `${accountState.user.bone_rating} BONE RATING · ${accountState.user.rated_games} RATED GAMES`;
  $("#accountNameInput").value = accountState.user.username || "";
  const stats = accountState.dashboard?.stats || {};
  const labels = { games: "GAMES", wins: "WINS", losses: "LOSSES", draws: "DRAWS", favorites: "FAVORITES" };
  const statsRoot = $("#accountStats");
  statsRoot.replaceChildren();
  for (const key of Object.keys(labels)) {
    const item = document.createElement("div");
    item.className = "account-stat";
    const value = document.createElement("strong");
    value.textContent = stats[key] || 0;
    const label = document.createElement("span");
    label.textContent = labels[key];
    item.append(value, label);
    statsRoot.append(item);
  }
  const gamesRoot = $("#accountGames");
  gamesRoot.replaceChildren();
  const games = accountState.dashboard?.games || [];
  if (!games.length) {
    const empty = document.createElement("p");
    empty.className = "empty-copy";
    empty.textContent = "No completed online games yet.";
    gamesRoot.append(empty);
  } else {
    for (const game of games) gamesRoot.append(makeHistoryRow(game, { favorite: true }));
  }

  const invites = accountState.invitations || [];
  $("#accountInvitationsSection").hidden = !invites.length;
  const invitationRoot = $("#accountInvitations");
  invitationRoot.replaceChildren();
  for (const invite of invites) {
    const row = document.createElement("article");
    row.className = "account-invitation";
    const copy = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = invite.from_username;
    const detail = document.createElement("span");
    detail.textContent = `${invite.from_rating} BR · ${invite.from_online ? "ONLINE" : "OFFLINE"}`;
    copy.append(name, detail);
    const enter = document.createElement("a");
    enter.className = "button primary invitation-link";
    enter.href = invite.path;
    enter.textContent = "Play";
    row.append(copy, enter);
    invitationRoot.append(row);
  }
}

async function refreshAccount() {
  const response = await fetch("/api/account");
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Could not load the account.");
  accountState = payload;
  renderAccount();
  return payload;
}

async function openAccount() {
  $("#accountError").textContent = "";
  try { await refreshAccount(); }
  catch (error) { $("#accountError").textContent = error.message; }
  if (!$("#accountDialog").open) $("#accountDialog").showModal();
}

async function setFavorite(gameId, favorite) {
  try {
    const response = await fetch("/api/account/favorite", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_id: gameId, favorite }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not update the favorite.");
    await refreshAccount();
  } catch (error) {
    $("#accountError").textContent = error.message;
  }
}

$("#accountButton").addEventListener("click", openAccount);
$("#accountCloseButton").addEventListener("click", () => $("#accountDialog").close());
$("#googleSignInButton").addEventListener("click", () => {
  const returnPath = `${window.location.pathname}${window.location.search}`;
  window.location.assign(`/api/auth/google/start?return=${encodeURIComponent(returnPath)}`);
});
$("#logoutButton").addEventListener("click", async () => {
  await fetch("/api/auth/logout", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  await refreshAccount();
});
$("#accountNameForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("#accountError").textContent = "";
  const completingSetup = Boolean(accountState?.user?.setup_required);
  try {
    const response = await fetch("/api/account/username", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: $("#accountNameInput").value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not save the username.");
    await refreshAccount();
    if (completingSetup && isMatchMode) window.location.reload();
  } catch (error) {
    $("#accountError").textContent = error.message;
  }
});

$("#playerSearchForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const root = $("#publicProfile");
  root.replaceChildren();
  $("#publicProfileError").textContent = "";
  try {
    const username = $("#playerSearchInput").value.trim();
    const response = await fetch(`/api/player?username=${encodeURIComponent(username)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not find that player.");
    const profile = payload.profile;
    const heading = document.createElement("div");
    heading.className = "public-profile-heading";
    const identity = document.createElement("div");
    const title = document.createElement("h4");
    title.textContent = profile.user.username;
    const presence = document.createElement("span");
    presence.className = `presence ${profile.user.online ? "online" : ""}`;
    presence.textContent = profile.user.online ? "ONLINE" : "OFFLINE";
    identity.append(title, presence);
    const rating = document.createElement("strong");
    rating.textContent = `${profile.user.bone_rating} BR`;
    heading.append(identity, rating);
    root.append(heading);
    const summary = document.createElement("p");
    summary.className = "public-stats";
    summary.textContent = `${profile.stats.games} games · ${profile.stats.wins} wins · ${profile.stats.losses} losses · ${profile.stats.draws} draws`;
    root.append(summary);
    if (!profile.games.length) {
      const empty = document.createElement("p");
      empty.className = "empty-copy";
      empty.textContent = "No completed online games yet.";
      root.append(empty);
    } else {
      for (const game of profile.games) root.append(makeHistoryRow(game, { publicView: true }));
    }
  } catch (error) {
    $("#publicProfileError").textContent = error.message;
  }
});

$("#inviteButton").addEventListener("click", openInvite);
$("#inviteCloseButton").addEventListener("click", closeInvite);
$("#inviteCancelButton").addEventListener("click", closeInvite);
$("#variantInput").addEventListener("change", (event) => {
  $("#variantDescription").textContent = VARIANT_DESCRIPTIONS[event.target.value] || "";
});
$("#inviteForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("#inviteError").textContent = "";
  const submit = event.submitter;
  if (submit) submit.disabled = true;
  try {
    const response = await fetch("/api/match/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        variant: $("#variantInput").value,
        base_minutes: Number($("#baseTimeInput").value),
        increment_seconds: Number($("#incrementInput").value),
        creator_color: $("#creatorColorInput").value,
        guest_name: $("#guestNameInput").value.trim(),
        opponent_username: $("#opponentUsernameInput").value.trim(),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not create the match.");
    createdMatch = payload;
    if (!accountState?.signed_in && $("#guestNameInput").value.trim()) {
      guestName = $("#guestNameInput").value.trim();
      try { localStorage.setItem(guestNameKey, guestName); } catch { /* private storage */ }
    }
    sessionStorage.setItem(`dinnerbone-match-${payload.room_id}`, payload.token);
    $("#inviteLink").value = new URL(payload.path, window.location.origin).href;
    $("#inviteCreatedDescription").textContent = payload.invited_username
      ? `Invite sent to ${payload.invited_username}. They will see it while signed in.`
      : "Your match is ready. Send this link to one player.";
    $("#inviteOptions").hidden = true;
    $("#inviteCreated").hidden = false;
  } catch (error) {
    $("#inviteError").textContent = error.message;
  } finally {
    if (submit) submit.disabled = false;
  }
});
$("#copyInviteButton").addEventListener("click", async () => {
  const copied = await copyText($("#inviteLink").value, "Invite link copied.");
  if (copied) $("#copyInviteButton").textContent = "Copied";
});
$("#enterMatchButton").addEventListener("click", () => {
  if (createdMatch) window.location.assign(createdMatch.path);
});

$("#setupButton").addEventListener("click", () => openSetup(false));
$("#setupCloseButton").addEventListener("click", closeSetup);
$("#setupCancelButton").addEventListener("click", closeSetup);
$("#sessionBackButton").addEventListener("click", showChoices);
document.querySelectorAll("[data-session]").forEach((button) => button.addEventListener("click", () => chooseSession(button.dataset.session)));
$("#flipButton").addEventListener("click", () => {
  if (!isMatchMode) return action("flip");
  matchBoardFlipped = !state.flipped;
  state.flipped = matchBoardFlipped;
  renderedBoardKey = "";
  render();
});
$("#newButton").addEventListener("click", () => action("new"));
$("#saveButton").addEventListener("click", () => saveDialog.showModal());
$("#saveCloseButton").addEventListener("click", () => saveDialog.close());
$("#undoButton").addEventListener("click", () => action("undo"));
$("#drawButton").addEventListener("click", () => action("offer_draw"));
$("#resignButton").addEventListener("click", () => {
  if (window.confirm("Resign this game?")) action("resign");
});
$("#rematchButton").addEventListener("click", () => {
  if (state?.rematch?.ready) {
    const next = state.rematch;
    sessionStorage.setItem(`dinnerbone-match-${next.room_id}`, next.token);
    window.location.assign(next.path);
    return;
  }
  action("rematch");
});
$("#acceptOfferButton").addEventListener("click", () => {
  if (pendingOfferKind) action(`accept_${pendingOfferKind}`);
});
$("#declineOfferButton").addEventListener("click", () => {
  if (pendingOfferKind) action(`decline_${pendingOfferKind}`);
});
$("#cancelPremoveButton").addEventListener("click", () => cancelPremove());
engineMoveButton.addEventListener("click", () => action("engine_move"));
setupDialog.addEventListener("click", closeDialogFromBackdrop);

for (const name of ["attack", "risk", "variety", "tenacity", "accuracy", "temperament", "endgames", "elo"]) {
  $(`#${name}Input`).addEventListener("input", (event) => { $(`#${name}Value`).value = event.target.value; });
}
$("#engineEloInput").addEventListener("input", (event) => { $("#engineEloValue").value = event.target.value; });
$("#engineInput").addEventListener("change", updateEngineStrengthControl);

document.querySelectorAll("[data-export]").forEach((button) => button.addEventListener("click", () => {
  const link = document.createElement("a");
  link.href = `/api/export?format=${encodeURIComponent(button.dataset.export)}`;
  link.download = "";
  document.body.append(link);
  link.click();
  link.remove();
  saveDialog.close();
}));

async function readProfileFiles() {
  const files = [...$("#profileFilesInput").files];
  if (!files.length) throw new Error("Choose at least one PGN or FEN file.");
  const total = files.reduce((sum, file) => sum + file.size, 0);
  if (total > 7_500_000) throw new Error("Keep the combined upload below 7.5 MB.");
  return Promise.all(files.map(async (file) => ({ name: file.name, text: await file.text() })));
}

function showPersonalProfile(profile) {
  personalProfile = profile;
  const metrics = $("#profileMetrics");
  metrics.replaceChildren();
  for (const key of ["attack", "risk", "variety"]) {
    const item = document.createElement("div");
    const value = document.createElement("b");
    const label = document.createElement("span");
    value.textContent = profile.personality[key];
    label.textContent = key.toUpperCase();
    item.append(value, label);
    metrics.append(item);
  }
  const summary = $("#profileSummary");
  summary.replaceChildren();
  const confidence = document.createElement("p");
  confidence.textContent = `${profile.confidence.toUpperCase()} CONFIDENCE · ${profile.stats.games} games · ${profile.stats.moves} modeled moves · ${profile.stats.fen_positions} FEN samples`;
  summary.append(confidence);
  for (const text of profile.summary) {
    const item = document.createElement("p");
    item.textContent = text;
    summary.append(item);
  }
  $("#profileResult").hidden = false;
}

$("#playProfileButton").addEventListener("click", async () => {
  if (!personalProfile) return;
  $("#setupError").textContent = "";
  try {
    await api("/api/configure", {
      mode: personalProfile.player_color,
      opponent: "personality",
      personality: personalProfile.personality,
      think_seconds: 30,
      engine_id: "dinnerbone",
      fen: "",
    });
    setupDialog.close();
  } catch (error) {
    $("#setupError").textContent = error.message;
  }
});

$("#copyFenButton").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(state.fen);
    statusMessage.textContent = "FEN copied to the clipboard.";
  } catch {
    statusMessage.textContent = "Clipboard access was blocked; select the FEN manually.";
  }
});

setupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!setupKind) return;
  $("#setupError").textContent = "";
  if (setupKind === "profile") {
    try {
      const files = await readProfileFiles();
      const response = await fetch("/api/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ files, modeled_color: $("#profileColorInput").value }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not analyze the games.");
      showPersonalProfile(payload.profile);
    } catch (error) {
      $("#setupError").textContent = error.message;
    }
    return;
  }
  const mode = setupKind === "load" ? "analysis" : setupKind === "engine" ? $("#engineColorInput").value : "white";
  const engineId = setupKind === "engine" ? $("#engineInput").value : "dinnerbone";
  const opponent = setupKind === "personality" ? "personality" : (setupKind === "elo" || (setupKind === "engine" && engineId === "dinnerbone")) ? "elo" : "standard";
  const estimatedElo = setupKind === "engine" ? Number($("#engineEloInput").value) : Number($("#eloInput").value);
  try {
    await api("/api/configure", {
      mode,
      opponent,
      think_seconds: 30,
      fen: setupKind === "load" ? $("#fenInput").value.trim() : "",
      engine_id: engineId,
      personality: {
        attack: Number($("#attackInput").value),
        risk: Number($("#riskInput").value),
        variety: Number($("#varietyInput").value),
        tenacity: Number($("#tenacityInput").value),
        accuracy: Number($("#accuracyInput").value),
        temperament: Number($("#temperamentInput").value),
        endgames: Number($("#endgamesInput").value),
      },
      estimated_elo: estimatedElo,
    });
    setupDialog.close();
  } catch (error) {
    $("#setupError").textContent = error.message;
  }
});

window.addEventListener("beforeunload", () => eventSource?.close());
window.setInterval(() => {
  updateBoardThinking();
  updateMatchClocks();
}, 100);
window.setInterval(() => {
  if (isMatchMode && state?.waiting) renderHeader();
}, 1000);
window.setInterval(() => {
  if (accountState?.signed_in) refreshAccount().catch(() => {});
}, 30000);

async function startApp() {
  try {
    try { await refreshAccount(); } catch { accountState = null; }
    const parameters = new URLSearchParams(window.location.search);
    if (shouldShowIntro(parameters)) playSiteIntro();
    else $("#siteIntro").hidden = true;
    if (parameters.has("auth_error")) {
      statusMessage.textContent = parameters.get("auth_error");
      window.history.replaceState({}, "", window.location.pathname);
      window.setTimeout(openAccount, 0);
    } else if (parameters.has("signed_in")) {
      window.history.replaceState({}, "", window.location.pathname);
      window.setTimeout(openAccount, 0);
    } else if (!isMatchMode && parameters.has("invite")) {
      window.history.replaceState({}, "", window.location.pathname);
      window.setTimeout(openInvite, 0);
    }
    if (accountState?.user?.setup_required) {
      window.setTimeout(async () => {
        await openAccount();
        $("#accountNameInput").focus();
      }, 0);
    }
    let payload;
    if (isMatchMode) {
      const response = await fetch("/api/match/join", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room_id: matchId, token: matchToken, guest_name: guestName }),
      });
      const joined = await response.json();
      if (!response.ok) throw new Error(joined.error || "Could not join this match.");
      matchToken = joined.token;
      sessionStorage.setItem(`dinnerbone-match-${matchId}`, matchToken);
      payload = joined.state;
    } else {
      const response = await fetch("/api/state");
      payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not load Dinnerbone.");
    }
    scheduleRender(payload);
    if (!isMatchMode && !payload.configured) {
      $(".interaction-hint").textContent = "Choose New session to play, analyze, or configure Dinnerbone.";
    }
    setConnection(true, "Live connection");
    connectEvents();
  } catch (error) {
    statusMessage.textContent = `Unable to connect: ${error.message}`;
    $(".interaction-hint").textContent = error.message;
    $("#modeBadge").textContent = isMatchMode ? "MATCH UNAVAILABLE" : "OFFLINE";
    $("#searchBadge").textContent = "OFFLINE";
    setConnection(false, "Offline");
  }
}

startApp();
