"use strict";

const API = "/api/v1";
const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;
const LAST_VIDEO_KEY = "cliptrace:last-video";
const INDEX_POLL_INTERVAL_MS = 2000;

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
  videoBadge: document.querySelector("#videoBadge"),
  videoTitle: document.querySelector("#videoTitle"),
  videoMetadata: document.querySelector("#videoMetadata"),
  indexStatus: document.querySelector("#indexStatus"),
  indexStatusTitle: document.querySelector("#indexStatusTitle"),
  indexStatusText: document.querySelector("#indexStatusText"),
  reindexButton: document.querySelector("#reindexButton"),
  searchForm: document.querySelector("#searchForm"),
  queryInput: document.querySelector("#queryInput"),
  searchButton: document.querySelector("#searchButton"),
  searchHint: document.querySelector("#searchHint"),
  searchMessage: document.querySelector("#searchMessage"),
  resultsSection: document.querySelector("#resultsSection"),
  resultsCount: document.querySelector("#resultsCount"),
  resultsList: document.querySelector("#resultsList"),
  resultsEmpty: document.querySelector("#resultsEmpty"),
  videoList: document.querySelector("#videoList"),
  libraryEmpty: document.querySelector("#libraryEmpty"),
  refreshButton: document.querySelector("#refreshButton"),
  toast: document.querySelector("#toast"),
};

const state = {
  selectedFile: null,
  currentVideo: null,
  currentIndex: null,
  currentResults: [],
  currentResult: null,
  currentResultElement: null,
  excerptPlaying: false,
  searchInProgress: false,
  indexPollTimer: null,
  indexRequestToken: 0,
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

function formatTimecode(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "00:00:00.000";
  const milliseconds = Math.round(seconds * 1000);
  const hours = Math.floor(milliseconds / 3600000);
  const minutes = Math.floor((milliseconds % 3600000) / 60000);
  const remainingSeconds = Math.floor((milliseconds % 60000) / 1000);
  const remainingMilliseconds = milliseconds % 1000;
  return [
    String(hours).padStart(2, "0"),
    String(minutes).padStart(2, "0"),
    `${String(remainingSeconds).padStart(2, "0")}.${String(remainingMilliseconds).padStart(3, "0")}`,
  ].join(":");
}

function formatScore(score) {
  const numericScore = Number(score);
  if (!Number.isFinite(numericScore)) return null;
  return numericScore.toFixed(3);
}

function pluralizeResults(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return `${count} результат`;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return `${count} результата`;
  return `${count} результатов`;
}

function humanError(code, fallback) {
  const messages = {
    UPLOAD_TOO_LARGE: "Файл превышает лимит 500 МБ.",
    UNSUPPORTED_MEDIA_TYPE: "Нужен MP4 с видео H.264/yuv420p и аудио AAC.",
    INVALID_VIDEO: "Не удалось прочитать видео. Проверьте файл и попробуйте снова.",
    VIDEO_STREAM_REQUIRED: "В файле не найдена видеодорожка.",
    VIDEO_AUDIO_TRACK_REQUIRED: "В файле не найдена AAC-аудиодорожка.",
    VIDEO_TOO_SHORT: "Видео слишком короткое для обработки.",
    VIDEO_TOO_LONG: "Видео превышает допустимую длительность.",
    VIDEO_FILE_MISSING: "Файл видео больше недоступен. Загрузите его повторно.",
    VIDEO_NOT_READY: "Видео ещё не готово к поиску.",
    VIDEO_INDEX_NOT_READY: "Поисковый индекс ещё не готов. Дождитесь окончания обработки.",
    VIDEO_SEARCH_INDEX_NOT_READY: "Поисковый индекс ещё не готов. Дождитесь окончания обработки.",
    SEARCH_INDEX_NOT_READY: "Поисковый индекс ещё не готов. Дождитесь окончания обработки.",
    SEARCH_INDEX_FAILED: "Индексация не завершилась. Запустите её повторно.",
    VIDEO_NOT_FOUND: "Видео не найдено. Обновите список.",
    REQUEST_VALIDATION_ERROR: "Проверьте заполненные поля.",
    INTERNAL_ERROR: "Сервис столкнулся с ошибкой. Попробуйте ещё раз.",
  };
  return messages[code] || fallback || "Не удалось выполнить действие.";
}

async function responseError(response) {
  try {
    const payload = await response.json();
    const detail = payload?.detail;
    const code = payload?.error?.code || (typeof detail === "object" ? detail?.code : null);
    const fallback = payload?.error?.message || (typeof detail === "string" ? detail : detail?.message);
    return humanError(code, fallback);
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
      setMessage(elements.uploadMessage, "Видео загружено. Готовим поисковый индекс…", true);
      showToast("Видео загружено — началась ML-индексация");
      await loadVideos(false);
      window.setTimeout(() => {
        elements.uploadProgress.hidden = true;
        elements.progressBar.style.width = "0";
      }, 700);
      elements.queryInput.focus();
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

function normalizedIndexStatus() {
  const status = String(state.currentIndex?.status || "loading").toLowerCase();
  if (["queued", "not_started"].includes(status)) return "pending";
  if (["indexing", "running"].includes(status)) return "processing";
  if (["error"].includes(status)) return "failed";
  return status;
}

function updateSearchAvailability() {
  const ready = normalizedIndexStatus() === "ready";
  const hasQuery = elements.queryInput.value.trim().length > 0;
  elements.searchButton.disabled = !ready || !hasQuery || state.searchInProgress;
}

function renderSearchIndex() {
  const status = normalizedIndexStatus();
  const statusConfig = {
    loading: {
      title: "Проверяем поисковый индекс",
      text: "Получаем актуальное состояние…",
      badge: "Проверяем индекс",
      className: "is-loading",
      hint: "Поиск станет доступен, когда расшифровка видео будет готова.",
    },
    pending: {
      title: "Видео в очереди",
      text: "ML-обработка начнётся автоматически.",
      badge: "В очереди",
      className: "is-pending",
      hint: "Можно заранее описать момент — кнопка поиска включится после индексации.",
    },
    processing: {
      title: "Распознаём речь",
      text: "Создаём расшифровку и поисковые фрагменты. Это может занять несколько минут.",
      badge: "Индексируем",
      className: "is-processing",
      hint: "Можно заранее описать момент — кнопка поиска включится после индексации.",
    },
    ready: {
      title: "Смысловой поиск готов",
      text: "Опишите сцену или произнесённую мысль обычными словами.",
      badge: "Индекс готов",
      className: "is-ready",
      hint: "Ищите по смыслу речи: точная формулировка из видео не обязательна.",
    },
    failed: {
      title: "Индексация не завершена",
      text: "Запустите обработку повторно. Само видео по-прежнему доступно.",
      badge: "Ошибка индекса",
      className: "is-failed",
      hint: "Повторите индексацию, чтобы включить поиск по видео.",
    },
    unavailable: {
      title: "Не удалось проверить индекс",
      text: "Проверьте соединение с сервисом и попробуйте ещё раз.",
      badge: "Индекс недоступен",
      className: "is-failed",
      hint: "Поиск временно недоступен: состояние индекса не получено.",
    },
  };
  const config = statusConfig[status] || statusConfig.unavailable;
  let detail = config.text;

  if (status === "ready") {
    const fragmentCount = Number(state.currentIndex?.fragment_count);
    const parts = [];
    if (Number.isFinite(fragmentCount)) {
      parts.push(`${new Intl.NumberFormat("ru").format(fragmentCount)} фрагментов`);
    }
    if (state.currentIndex?.indexed_at) parts.push(`обновлено ${formatDate(state.currentIndex.indexed_at)}`);
    if (parts.length) detail = parts.join(" · ");
  } else if (status === "failed") {
    detail = state.currentIndex?.error_message || config.text;
    const attempts = Number(state.currentIndex?.attempts);
    if (Number.isFinite(attempts) && attempts > 0) detail += ` Попыток: ${attempts}.`;
  } else if (status === "unavailable" && state.currentIndex?.error_message) {
    detail = state.currentIndex.error_message;
  }

  elements.indexStatus.className = `index-status ${config.className}`;
  elements.indexStatus.setAttribute("aria-busy", ["loading", "pending", "processing"].includes(status) ? "true" : "false");
  elements.indexStatusTitle.textContent = config.title;
  elements.indexStatusText.textContent = detail;
  elements.videoBadge.textContent = config.badge;
  elements.videoBadge.className = `video-badge ${config.className}`;
  elements.searchHint.textContent = config.hint;
  elements.reindexButton.hidden = !["ready", "failed", "unavailable"].includes(status);
  elements.reindexButton.textContent = status === "ready" ? "Обновить индекс" : status === "unavailable" ? "Проверить снова" : "Повторить";
  updateSearchAvailability();
}

function clearIndexPolling() {
  if (state.indexPollTimer !== null) {
    window.clearTimeout(state.indexPollTimer);
    state.indexPollTimer = null;
  }
}

function scheduleIndexPoll(videoId, requestToken, delay = INDEX_POLL_INTERVAL_MS) {
  clearIndexPolling();
  state.indexPollTimer = window.setTimeout(() => {
    void loadSearchIndex(videoId, requestToken);
  }, delay);
}

async function loadSearchIndex(videoId, requestToken = state.indexRequestToken) {
  if (state.currentVideo?.id !== videoId || requestToken !== state.indexRequestToken) return;
  try {
    const response = await fetch(`${API}/videos/${videoId}/search-index`);
    if (!response.ok) throw new Error(await responseError(response));
    const payload = await response.json();
    if (state.currentVideo?.id !== videoId || requestToken !== state.indexRequestToken) return;
    const previousStatus = normalizedIndexStatus();
    state.currentIndex = payload;
    renderSearchIndex();
    if (["pending", "processing"].includes(previousStatus) && normalizedIndexStatus() === "ready") {
      showToast("Поисковый индекс готов");
    }
    if (["pending", "processing"].includes(normalizedIndexStatus())) {
      scheduleIndexPoll(videoId, requestToken);
    } else {
      clearIndexPolling();
    }
  } catch (error) {
    if (state.currentVideo?.id !== videoId || requestToken !== state.indexRequestToken) return;
    state.currentIndex = {
      status: "unavailable",
      error_message: error instanceof Error ? error.message : "Не удалось получить состояние индекса.",
    };
    renderSearchIndex();
  }
}

function resetSearchResults() {
  state.currentResults = [];
  state.currentResult = null;
  state.currentResultElement = null;
  state.excerptPlaying = false;
  elements.videoPlayer.pause();
  elements.resultsSection.hidden = true;
  elements.resultsList.replaceChildren();
  elements.resultsEmpty.hidden = true;
  elements.resultsCount.textContent = "";
}

function selectVideo(video) {
  clearIndexPolling();
  state.indexRequestToken += 1;
  state.currentVideo = video;
  state.currentIndex = {status: "loading"};
  state.searchInProgress = false;
  window.localStorage.setItem(LAST_VIDEO_KEY, video.id);
  elements.playerEmpty.hidden = true;
  elements.playerContent.hidden = false;
  elements.queryInput.value = "";
  setMessage(elements.searchMessage, "");
  resetSearchResults();
  renderSearchIndex();

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
  void loadSearchIndex(video.id, state.indexRequestToken);
}

function renderSearchResults(results) {
  elements.resultsSection.hidden = false;
  elements.resultsList.replaceChildren();
  elements.resultsEmpty.hidden = results.length > 0;
  elements.resultsCount.textContent = pluralizeResults(results.length);

  results.forEach((result, index) => {
    const card = document.createElement("article");
    card.className = "result-card";

    const copy = document.createElement("div");
    copy.className = "result-copy";
    const meta = document.createElement("div");
    meta.className = "result-meta";
    const label = document.createElement("span");
    label.className = "result-label";
    label.textContent = `Отрывок ${result.rank || index + 1}`;
    meta.append(label);
    const score = formatScore(result.score);
    if (score) {
      const scoreLabel = document.createElement("span");
      scoreLabel.className = "result-score";
      scoreLabel.textContent = `Cosine score ${score}`;
      meta.append(scoreLabel);
    }
    const transcript = document.createElement("p");
    transcript.className = "result-transcript";
    transcript.textContent = result.preview_text || "Для этого отрывка текст расшифровки недоступен.";
    const timecode = document.createElement("p");
    timecode.className = "result-timecode";
    timecode.textContent = `${result.start_timecode || formatTimecode(result.start_seconds)} — ${result.end_timecode || formatTimecode(result.end_seconds)}`;
    copy.append(meta, transcript, timecode);

    const playButton = document.createElement("button");
    playButton.className = "play-button";
    playButton.type = "button";
    playButton.setAttribute("aria-label", `Смотреть отрывок ${index + 1}, ${timecode.textContent}`);
    const playIcon = document.createElement("span");
    playIcon.className = "play-icon";
    playIcon.setAttribute("aria-hidden", "true");
    playIcon.textContent = "▶";
    const playLabel = document.createElement("span");
    playLabel.textContent = "Смотреть";
    playButton.append(playIcon, playLabel);
    playButton.addEventListener("click", () => void playExcerpt(result, card));

    const track = document.createElement("div");
    track.className = "excerpt-track";
    track.setAttribute("aria-hidden", "true");
    track.append(document.createElement("span"));
    card.append(copy, playButton, track);
    elements.resultsList.append(card);
  });
}

async function searchVideo(event) {
  event.preventDefault();
  if (!state.currentVideo) return;
  if (normalizedIndexStatus() !== "ready") {
    setMessage(elements.searchMessage, "Дождитесь окончания индексации видео.");
    return;
  }
  const query = elements.queryInput.value.trim();
  if (!query) {
    setMessage(elements.searchMessage, "Опишите момент, который нужно найти.");
    elements.queryInput.focus();
    return;
  }

  state.searchInProgress = true;
  updateSearchAvailability();
  elements.searchButton.textContent = "Ищем…";
  setMessage(elements.searchMessage, "");
  try {
    const videoId = state.currentVideo.id;
    const response = await fetch(`${API}/videos/${videoId}/search`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query, limit: 5}),
    });
    if (!response.ok) throw new Error(await responseError(response));
    const payload = await response.json();
    if (state.currentVideo?.id !== videoId) return;
    const results = Array.isArray(payload.results) ? payload.results : [];
    resetSearchResults();
    state.currentResults = results;
    renderSearchResults(results);
    setMessage(
      elements.searchMessage,
      results.length ? "Нашли подходящие отрывки — выберите один для просмотра." : "Подходящих отрывков не найдено.",
      results.length > 0,
    );
    elements.resultsSection.scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (error) {
    setMessage(elements.searchMessage, error instanceof Error ? error.message : "Поиск не выполнен.");
  } finally {
    state.searchInProgress = false;
    elements.searchButton.textContent = "Найти";
    updateSearchAvailability();
  }
}

async function playExcerpt(result, resultElement) {
  if (!result || !resultElement) return;
  state.currentResult = result;
  state.currentResultElement = resultElement;
  state.excerptPlaying = true;
  elements.resultsList.querySelectorAll(".result-card").forEach((card) => {
    card.classList.remove("active", "playing");
    const label = card.querySelector(".play-button span:last-child");
    if (label) label.textContent = "Смотреть";
  });
  const duration = Math.max(0.1, Number(result.end_seconds) - Number(result.start_seconds));
  resultElement.style.setProperty("--excerpt-duration", `${duration}s`);
  resultElement.classList.add("active");
  void resultElement.offsetWidth;
  resultElement.classList.add("playing");
  const playLabel = resultElement.querySelector(".play-button span:last-child");
  if (playLabel) playLabel.textContent = "Идёт отрывок";
  elements.videoPlayer.currentTime = result.start_seconds;
  try {
    await elements.videoPlayer.play();
    monitorExcerptFrames();
  } catch {
    state.excerptPlaying = false;
    resultElement.classList.remove("playing");
    if (playLabel) playLabel.textContent = "Смотреть";
  }
}

function stopExcerpt() {
  if (!state.currentResult) return;
  elements.videoPlayer.pause();
  elements.videoPlayer.currentTime = state.currentResult.end_seconds;
  state.excerptPlaying = false;
  if (state.currentResultElement) {
    state.currentResultElement.classList.remove("playing");
    const playLabel = state.currentResultElement.querySelector(".play-button span:last-child");
    if (playLabel) playLabel.textContent = "Повторить";
  }
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

async function reindexCurrentVideo() {
  if (!state.currentVideo) return;
  const videoId = state.currentVideo.id;
  elements.reindexButton.disabled = true;
  elements.reindexButton.textContent = "Запускаем…";
  try {
    const response = await fetch(`${API}/videos/${videoId}/search-index/reindex`, {method: "POST"});
    if (!response.ok) throw new Error(await responseError(response));
    if (state.currentVideo?.id !== videoId) return;
    let payload = null;
    if (response.headers.get("content-type")?.includes("application/json")) {
      payload = await response.json();
    }
    resetSearchResults();
    state.currentIndex = payload?.status ? payload : {status: "pending"};
    setMessage(elements.searchMessage, "");
    renderSearchIndex();
    scheduleIndexPoll(videoId, state.indexRequestToken, 700);
    showToast("Повторная индексация запущена");
  } catch (error) {
    if (state.currentVideo?.id !== videoId) return;
    state.currentIndex = {
      status: "unavailable",
      error_message: error instanceof Error ? error.message : "Не удалось запустить индексацию.",
    };
    renderSearchIndex();
  } finally {
    if (state.currentVideo?.id === videoId) elements.reindexButton.disabled = false;
  }
}

function handleIndexAction() {
  if (!state.currentVideo) return;
  if (normalizedIndexStatus() === "unavailable") {
    state.currentIndex = {status: "loading"};
    renderSearchIndex();
    void loadSearchIndex(state.currentVideo.id, state.indexRequestToken);
    return;
  }
  void reindexCurrentVideo();
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
elements.queryInput.addEventListener("input", updateSearchAvailability);
elements.reindexButton.addEventListener("click", handleIndexAction);
elements.videoPlayer.addEventListener("timeupdate", enforceExcerptBoundary);
elements.videoPlayer.addEventListener("seeking", () => {
  if (!state.currentResult || !state.excerptPlaying) return;
  if (
    elements.videoPlayer.currentTime < state.currentResult.start_seconds ||
    elements.videoPlayer.currentTime > state.currentResult.end_seconds
  ) {
    state.excerptPlaying = false;
    if (state.currentResultElement) {
      state.currentResultElement.classList.remove("playing");
      const playLabel = state.currentResultElement.querySelector(".play-button span:last-child");
      if (playLabel) playLabel.textContent = "Смотреть";
    }
  }
});
elements.refreshButton.addEventListener("click", () => loadVideos(false));
window.addEventListener("beforeunload", clearIndexPolling);

checkService();
loadVideos(true);
