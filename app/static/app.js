"use strict";

const API = "/api/v1";
const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;
const LAST_VIDEO_KEY = "cliptrace:last-video";

const elements = {
  serviceState: document.querySelector("#serviceState"),
  serviceLabel: document.querySelector("#serviceLabel"),
  uploadForm: document.querySelector("#uploadForm"),
  fileInput: document.querySelector("#fileInput"),
  dropzone: document.querySelector("#dropzone"),
  fileCard: document.querySelector("#fileCard"),
  fileName: document.querySelector("#fileName"),
  fileSize: document.querySelector("#fileSize"),
  clearFileButton: document.querySelector("#clearFileButton"),
  titleInput: document.querySelector("#titleInput"),
  uploadButton: document.querySelector("#uploadButton"),
  uploadProgress: document.querySelector("#uploadProgress"),
  progressValue: document.querySelector("#progressValue"),
  progressBar: document.querySelector("#progressBar"),
  uploadMessage: document.querySelector("#uploadMessage"),
  playerEmpty: document.querySelector("#playerEmpty"),
  playerContent: document.querySelector("#playerContent"),
  videoPlayer: document.querySelector("#videoPlayer"),
  videoTitle: document.querySelector("#videoTitle"),
  videoMetadata: document.querySelector("#videoMetadata"),
  searchForm: document.querySelector("#searchForm"),
  queryInput: document.querySelector("#queryInput"),
  searchButton: document.querySelector("#searchButton"),
  searchMessage: document.querySelector("#searchMessage"),
  resultCard: document.querySelector("#resultCard"),
  resultStart: document.querySelector("#resultStart"),
  resultEnd: document.querySelector("#resultEnd"),
  playExcerptButton: document.querySelector("#playExcerptButton"),
  videoList: document.querySelector("#videoList"),
  libraryEmpty: document.querySelector("#libraryEmpty"),
  refreshButton: document.querySelector("#refreshButton"),
  toast: document.querySelector("#toast"),
};

const state = {
  selectedFile: null,
  currentVideo: null,
  currentResult: null,
  excerptPlaying: false,
  videos: [],
};

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 Б";
  const units = ["Б", "КБ", "МБ", "ГБ"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toFixed(index === 0 || value >= 10 ? 0 : 1)} ${units[index]}`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remaining = total % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`
    : `${minutes}:${String(remaining).padStart(2, "0")}`;
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("ru", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function humanError(code, fallback) {
  const messages = {
    UPLOAD_TOO_LARGE: "Файл превышает лимит 500 МБ.",
    UNSUPPORTED_MEDIA_TYPE: "Нужен MP4 с видео H.264/yuv420p и аудио AAC.",
    INVALID_VIDEO: "Не удалось прочитать видео. Проверьте файл и попробуйте снова.",
    VIDEO_STREAM_REQUIRED: "В файле не найдена видеодорожка.",
    VIDEO_AUDIO_TRACK_REQUIRED: "В файле не найдена AAC-аудиодорожка.",
    VIDEO_TOO_SHORT: "Видео должно длиться не менее 10 секунд.",
    VIDEO_TOO_LONG: "Видео превышает допустимую длительность.",
    VIDEO_FILE_MISSING: "Файл видео больше недоступен. Загрузите его повторно.",
    VIDEO_NOT_READY: "Видео ещё не готово к поиску.",
    VIDEO_NOT_FOUND: "Видео не найдено. Обновите список.",
    REQUEST_VALIDATION_ERROR: "Проверьте заполненные поля.",
    INTERNAL_ERROR: "Сервис столкнулся с ошибкой. Попробуйте ещё раз.",
  };
  return messages[code] || fallback || "Не удалось выполнить действие.";
}

async function responseError(response) {
  try {
    const payload = await response.json();
    return humanError(payload?.error?.code, payload?.error?.message);
  } catch {
    return `Сервис ответил с ошибкой ${response.status}.`;
  }
}

function setMessage(element, message, success = false) {
  element.textContent = message;
  element.classList.toggle("success", success);
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 3600);
}

function setServiceState(status, label) {
  elements.serviceState.classList.toggle("ready", status === "ready");
  elements.serviceState.classList.toggle("error", status === "error");
  elements.serviceLabel.textContent = label;
}

async function checkService() {
  try {
    const response = await fetch("/health/ready");
    if (!response.ok) throw new Error("not ready");
    setServiceState("ready", "Сервис готов");
  } catch {
    setServiceState("error", "Сервис недоступен");
  }
}

function setSelectedFile(file) {
  setMessage(elements.uploadMessage, "");
  if (!file) {
    state.selectedFile = null;
    elements.fileInput.value = "";
    elements.fileCard.hidden = true;
    elements.dropzone.hidden = false;
    elements.uploadButton.disabled = true;
    return;
  }
  if (!file.name.toLowerCase().endsWith(".mp4")) {
    setMessage(elements.uploadMessage, "Выберите файл с расширением .mp4.");
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    setMessage(elements.uploadMessage, "Файл превышает лимит 500 МБ.");
    return;
  }
  state.selectedFile = file;
  elements.fileName.textContent = file.name;
  elements.fileSize.textContent = formatBytes(file.size);
  elements.fileCard.hidden = false;
  elements.dropzone.hidden = true;
  elements.uploadButton.disabled = false;
  if (!elements.titleInput.value.trim()) {
    elements.titleInput.value = file.name.replace(/\.mp4$/i, "").slice(0, 200);
  }
}

function uploadVideo(event) {
  event.preventDefault();
  if (!state.selectedFile) {
    setMessage(elements.uploadMessage, "Сначала выберите видео.");
    return;
  }

  const data = new FormData();
  data.append("file", state.selectedFile, state.selectedFile.name);
  const title = elements.titleInput.value.trim();
  if (title) data.append("title", title);

  elements.uploadButton.disabled = true;
  elements.uploadProgress.hidden = false;
  setMessage(elements.uploadMessage, "");

  const request = new XMLHttpRequest();
  request.open("POST", `${API}/videos`);
  request.responseType = "json";
  request.upload.addEventListener("progress", (progress) => {
    if (!progress.lengthComputable) return;
    const percent = Math.min(99, Math.round((progress.loaded / progress.total) * 100));
    elements.progressValue.textContent = `${percent}%`;
    elements.progressBar.style.width = `${percent}%`;
  });
  request.addEventListener("load", async () => {
    elements.uploadButton.disabled = false;
    if (request.status >= 200 && request.status < 300) {
      elements.progressValue.textContent = "100%";
      elements.progressBar.style.width = "100%";
      const video = request.response;
      selectVideo(video);
      setSelectedFile(null);
      elements.titleInput.value = "";
      setMessage(elements.uploadMessage, "Видео загружено и готово к поиску.", true);
      showToast("Видео готово — получите отрывок 5–10 секунд");
      await loadVideos(false);
      window.setTimeout(() => {
        elements.uploadProgress.hidden = true;
        elements.progressBar.style.width = "0";
      }, 700);
      elements.searchButton.focus();
      return;
    }
    const payload = request.response;
    const message = humanError(payload?.error?.code, payload?.error?.message);
    setMessage(elements.uploadMessage, message);
    elements.uploadProgress.hidden = true;
  });
  request.addEventListener("error", () => {
    elements.uploadButton.disabled = false;
    elements.uploadProgress.hidden = true;
    setMessage(elements.uploadMessage, "Не удалось связаться с сервисом.");
  });
  request.send(data);
}

function selectVideo(video) {
  state.currentVideo = video;
  state.currentResult = null;
  state.excerptPlaying = false;
  window.localStorage.setItem(LAST_VIDEO_KEY, video.id);
  elements.playerEmpty.hidden = true;
  elements.playerContent.hidden = false;
  elements.resultCard.hidden = true;
  elements.resultCard.classList.remove("playing");
  setMessage(elements.searchMessage, "");

  elements.videoPlayer.pause();
  elements.videoPlayer.src = video.links.content;
  elements.videoPlayer.load();
  elements.videoTitle.textContent = video.title || video.original_filename;
  elements.videoMetadata.replaceChildren();
  const metadata = [
    formatDuration(video.media.duration_seconds),
    `${video.media.width}×${video.media.height}`,
    formatBytes(video.size_bytes),
  ];
  metadata.forEach((value) => {
    const badge = document.createElement("span");
    badge.textContent = value;
    elements.videoMetadata.append(badge);
  });
  renderVideoList();
}

async function searchVideo(event) {
  event.preventDefault();
  if (!state.currentVideo) return;
  const query = elements.queryInput.value.trim() || "Фрагмент с 5 по 10 секунду";

  elements.searchButton.disabled = true;
  elements.searchButton.textContent = "Готовим…";
  setMessage(elements.searchMessage, "");
  try {
    const response = await fetch(`${API}/videos/${state.currentVideo.id}/search`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query, limit: 5}),
    });
    if (!response.ok) throw new Error(await responseError(response));
    const payload = await response.json();
    const result = payload.results?.[0];
    if (!result) throw new Error("Сервис не вернул ни одного отрывка.");
    state.currentResult = result;
    state.excerptPlaying = false;
    elements.videoPlayer.pause();
    elements.videoPlayer.currentTime = result.start_seconds;
    elements.resultStart.textContent = result.start_timecode;
    elements.resultEnd.textContent = result.end_timecode;
    elements.resultCard.hidden = false;
    elements.resultCard.classList.remove("playing");
    elements.playExcerptButton.lastElementChild.textContent = "Смотреть отрывок";
    setMessage(elements.searchMessage, "Отрывок найден. Нажмите, чтобы посмотреть.", true);
    elements.resultCard.scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (error) {
    setMessage(elements.searchMessage, error instanceof Error ? error.message : "Поиск не выполнен.");
  } finally {
    elements.searchButton.disabled = false;
    elements.searchButton.textContent = "Получить";
  }
}

async function playExcerpt() {
  if (!state.currentResult) return;
  state.excerptPlaying = true;
  elements.resultCard.classList.remove("playing");
  void elements.resultCard.offsetWidth;
  elements.resultCard.classList.add("playing");
  elements.playExcerptButton.lastElementChild.textContent = "Идёт отрывок";
  elements.videoPlayer.currentTime = state.currentResult.start_seconds;
  try {
    await elements.videoPlayer.play();
    monitorExcerptFrames();
  } catch {
    state.excerptPlaying = false;
    elements.resultCard.classList.remove("playing");
    elements.playExcerptButton.lastElementChild.textContent = "Смотреть отрывок";
  }
}

function stopExcerpt() {
  if (!state.currentResult) return;
  elements.videoPlayer.pause();
  elements.videoPlayer.currentTime = state.currentResult.end_seconds;
  state.excerptPlaying = false;
  elements.resultCard.classList.remove("playing");
  elements.playExcerptButton.lastElementChild.textContent = "Повторить отрывок";
}

function monitorExcerptFrames(_now, metadata) {
  if (!state.excerptPlaying || !state.currentResult) return;
  const playbackTime = metadata?.mediaTime ?? elements.videoPlayer.currentTime;
  if (playbackTime >= state.currentResult.end_seconds - 0.02) {
    stopExcerpt();
    return;
  }
  if (typeof elements.videoPlayer.requestVideoFrameCallback === "function") {
    elements.videoPlayer.requestVideoFrameCallback(monitorExcerptFrames);
  }
}

function enforceExcerptBoundary() {
  if (!state.excerptPlaying || !state.currentResult) return;
  if (elements.videoPlayer.currentTime >= state.currentResult.end_seconds - 0.04) {
    stopExcerpt();
  }
}

async function loadVideos(selectLatest = true) {
  elements.refreshButton.disabled = true;
  try {
    const response = await fetch(`${API}/videos?limit=12&offset=0`);
    if (!response.ok) throw new Error(await responseError(response));
    const payload = await response.json();
    state.videos = payload.items || [];
    renderVideoList();
    if (selectLatest && !state.currentVideo && state.videos.length) {
      const rememberedId = window.localStorage.getItem(LAST_VIDEO_KEY);
      selectVideo(state.videos.find((video) => video.id === rememberedId) || state.videos[0]);
    }
  } catch (error) {
    showToast(error instanceof Error ? error.message : "Не удалось загрузить список видео.");
  } finally {
    elements.refreshButton.disabled = false;
  }
}

function renderVideoList() {
  elements.videoList.replaceChildren();
  elements.libraryEmpty.hidden = state.videos.length > 0;
  state.videos.forEach((video) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "video-list-item";
    item.classList.toggle("active", video.id === state.currentVideo?.id);
    item.setAttribute("aria-label", `Открыть видео ${video.title || video.original_filename}`);

    const thumb = document.createElement("span");
    thumb.className = "list-thumb";
    thumb.textContent = "▶";
    const copy = document.createElement("span");
    copy.className = "list-copy";
    const title = document.createElement("strong");
    title.textContent = video.title || video.original_filename;
    const details = document.createElement("span");
    details.textContent = [
      `${video.media.width}×${video.media.height}`,
      formatBytes(video.size_bytes),
      formatDate(video.created_at),
    ].filter(Boolean).join(" · ");
    copy.append(title, details);
    const duration = document.createElement("span");
    duration.className = "list-duration";
    duration.textContent = formatDuration(video.media.duration_seconds);
    item.append(thumb, copy, duration);
    item.addEventListener("click", () => {
      selectVideo(video);
      document.querySelector(".player-panel").scrollIntoView({behavior: "smooth", block: "start"});
    });
    elements.videoList.append(item);
  });
}

elements.fileInput.addEventListener("change", () => setSelectedFile(elements.fileInput.files?.[0] || null));
elements.clearFileButton.addEventListener("click", () => setSelectedFile(null));
elements.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.fileInput.click();
  }
});
["dragenter", "dragover"].forEach((type) => {
  elements.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((type) => {
  elements.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("dragging");
  });
});
elements.dropzone.addEventListener("drop", (event) => setSelectedFile(event.dataTransfer?.files?.[0] || null));
elements.uploadForm.addEventListener("submit", uploadVideo);
elements.searchForm.addEventListener("submit", searchVideo);
elements.playExcerptButton.addEventListener("click", playExcerpt);
elements.videoPlayer.addEventListener("timeupdate", enforceExcerptBoundary);
elements.videoPlayer.addEventListener("seeking", () => {
  if (!state.currentResult || !state.excerptPlaying) return;
  if (
    elements.videoPlayer.currentTime < state.currentResult.start_seconds ||
    elements.videoPlayer.currentTime > state.currentResult.end_seconds
  ) {
    state.excerptPlaying = false;
    elements.resultCard.classList.remove("playing");
    elements.playExcerptButton.lastElementChild.textContent = "Смотреть отрывок";
  }
});
elements.refreshButton.addEventListener("click", () => loadVideos(false));

checkService();
loadVideos(true);
