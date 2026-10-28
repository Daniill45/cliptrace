# ClipTrace MVP

ClipTrace загружает MP4, расшифровывает речь локальной ASR-моделью и находит
релевантные отрывки по смыслу текстового запроса. Результат содержит таймкоды,
фрагмент расшифровки и cosine score; исходное видео воспроизводится через HTTP Range.

ML-контур работает локально и не отправляет видео во внешний API:

```text
MP4 → faster-whisper → временные окна → multilingual-e5-small (ONNX)
    → PostgreSQL/pgvector → top-k отрывков → HTML5-плеер
```

## Быстрый запуск

Требуется [Podman Desktop](https://podman-desktop.io/downloads/macos) с запущенной
Podman machine и установленным Compose. Все Python-зависимости зафиксированы в
`uv.lock`.

После первого запуска Podman Desktop завершите onboarding:

1. Установите Podman engine и создайте Podman machine.
2. Откройте `Settings > Resources > Compose` и нажмите `Setup`.
3. Проверьте окружение:

```bash
podman --version
podman machine list
podman compose version
```

Затем из каталога проекта выполните:

```bash
cp .env.example .env
make up
```

Compose поднимает PostgreSQL с pgvector, применяет Alembic-миграции, запускает API
и отдельный ML-worker. Откройте:

- интерфейс — <http://localhost:8000>;
- Swagger API — <http://localhost:8000/docs>.

Проверка состояния:

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

Остановка без удаления named volumes:

```bash
make down
```

Общие логи доступны через `make logs`, только ML-worker — через `make worker-logs`,
состояние контейнеров — через `make ps`.

При первой индексации worker скачает модели в named volume `model_cache`. Ревизии
E5 и faster-whisper зафиксированы SHA, а размер Whisper задаётся alias `small`.
На Apple Silicon внутри Podman используется ARM64 CPU; Metal/MPS контейнеру не
требуется. Скорость первой обработки зависит от длины видео и ресурсов Podman machine.
Для полного удаления локальных данных явно выполните `podman compose down -v`.

Makefile использует отдельную пустую конфигурацию Compose из `.podman/config.json`.
Она не содержит credentials и не затрагивает пользовательский `~/.docker/config.json`,
в котором после удаления Docker Desktop может остаться неработающий credential helper.

### Переход с Docker Desktop

Docker Desktop и Podman machine используют разные хранилища образов и named
volumes. Старые данные не появятся в Podman автоматически. Не удаляйте прежние
Docker volumes или `Docker.raw`, пока не проверили Podman-версию. Для переноса
нужны логический `pg_dump` PostgreSQL и отдельная копия каталога uploads; если
исходный MP4 сохранился на Mac, его проще безопасно загрузить через интерфейс снова.

## Основной сценарий в интерфейсе

1. Откройте <http://localhost:8000>.
2. Перетащите MP4 в область загрузки или выберите его через диалог.
3. Дождитесь статуса «Индекс готов». Страница обновляет его автоматически.
4. Опишите момент обычной фразой, например «где обсуждают планы на следующий год».
5. Выберите один из найденных отрывков — плеер воспроизведёт только его диапазон.

Недавние загрузки остаются в списке после обновления страницы. Поддерживаются MP4
с H.264/yuv420p и AAC, размером до 500 МБ и длительностью от 1 секунды до 4 часов.

## Работа через API

Загрузка валидного H.264/yuv420p MP4 с AAC-аудиодорожкой длительностью не менее
1 секунды:

```bash
curl -i \
  -F 'title=Demo interview' \
  -F 'file=@./sample.mp4;type=video/mp4' \
  http://localhost:8000/api/v1/videos
```

Идентификатор из ответа используется во всех следующих запросах:

```bash
curl http://localhost:8000/api/v1/videos/<VIDEO_ID>

curl http://localhost:8000/api/v1/videos/<VIDEO_ID>/search-index

curl -X POST \
  -H 'Content-Type: application/json' \
  -d '{"query":"момент, где говорят о провале","limit":5}' \
  http://localhost:8000/api/v1/videos/<VIDEO_ID>/search

curl -X POST \
  http://localhost:8000/api/v1/videos/<VIDEO_ID>/search-index/reindex

curl -i \
  -H 'Range: bytes=0-1023' \
  http://localhost:8000/api/v1/videos/<VIDEO_ID>/content

curl -i -X DELETE http://localhost:8000/api/v1/videos/<VIDEO_ID>
```

До готовности индекса search возвращает `409 SEARCH_INDEX_NOT_READY`. После обработки
ответ имеет `mode="semantic-transcript"` и содержит до запрошенного `limit` результатов
в порядке модельной релевантности. `score` — ranking score, а не вероятность.

## Проверки

Команды запускаются в воспроизводимом tools-контейнере. `make test` поднимает
отдельный PostgreSQL-сервис с базой `cliptrace_test` и отдельным named volume:

```bash
make migrate
make test
make lint
make typecheck
make format
```

Тесты никогда не используют рабочую базу `cliptrace`: перед очисткой fixture проверяет
и имя БД в `TEST_DATABASE_URL`, и результат `SELECT current_database()`. При любом
несовпадении suite останавливается до `TRUNCATE`. Тестовый MP4 создаётся локально через
`ffmpeg`; сеть тестам не нужна. Обычный pytest проверяет повторный application lifespan
на той же тестовой БД и upload directory. Полный рестарт контейнеров проверяется отдельно:

```bash
podman compose up -d --build
# загрузить sample.mp4 и сохранить VIDEO_ID
podman compose restart db api
curl http://localhost:8000/api/v1/videos/<VIDEO_ID>
curl http://localhost:8000/api/v1/videos/<VIDEO_ID>/content -o /dev/null
```

## Ручная браузерная проверка

1. Откройте интерфейс, загрузите видео и выполните поиск.
2. Дождитесь готовности индекса и выполните два разных смысловых запроса.
3. Убедитесь, что результаты содержат расшифровку и разные релевантные таймкоды.
4. Нажмите результат: воспроизведение должно начаться и остановиться на его границах.

Backend отдаёт исходное видео с HTTP byte ranges. Ограничение диапазона по времени
выполняет плеер; отдельный физически обрезанный MP4 в MVP не создаётся.

## Конфигурация и режимы

Настройки читаются из environment variables через `pydantic-settings`; полный набор
есть в `.env.example`. Невалидные лимиты, model/worker settings или CORS origins
останавливают создание приложения.

Основные ML-настройки:

- `WHISPER_MODEL=small`, `WHISPER_COMPUTE_TYPE=int8` — локальное распознавание речи;
- `WHISPER_MODEL_REVISION` фиксирует snapshot весов faster-whisper;
- `WHISPER_LANGUAGE` — необязательная фиксация языка; пустое значение включает auto-detect;
- `EMBEDDING_MODEL=intfloat/multilingual-e5-small` — 384-мерные embeddings;
- `EMBEDDING_MODEL_REVISION` фиксирует конкретную ревизию ONNX-модели;
- `ML_WORKER_LEASE_SECONDS` и `ML_WORKER_MAX_ATTEMPTS` управляют безопасным захватом job;
- `SEARCH_ENGINE=mock` можно использовать при прямом запуске для изолированной отладки.

`DATABASE_URL` из `.env.example` использует host `db` и предназначен для Compose.
Для прямого запуска с host нужен отдельный URL, например:

```bash
export DATABASE_URL='postgresql+asyncpg://cliptrace:cliptrace@localhost:5432/cliptrace'
export UPLOAD_DIR="$PWD/data/uploads"
uv sync --frozen
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --no-access-log
```

В host-режиме PostgreSQL и `ffprobe` должны быть установлены отдельно. Не смешивайте
host URL с контейнерным host `db`.

## Архитектура

- Endpoint-слой адаптирует HTTP-контракт, включая Range-заголовки, и делегирует
  ресурсную логику application services.
- `VideoService` владеет транзакционными границами загрузки и tombstone-удаления.
- `VideoRepository` отвечает только за SQLAlchemy persistence.
- `LocalStorage` владеет `.part`, UUID-based final files, `.deleting` tombstones,
  безопасным `stat/open` и потоковой выдачей.
- `MediaProbe` запускает фиксированный argv `ffprobe` без shell, ограничивает время и
  объём вывода и нормализует только необходимые поля.
- `SearchService` зависит от `FragmentSearchEngine`; semantic-реализация кодирует
  запрос E5-моделью и выполняет exact cosine search внутри одного `video_id`.
- `TranscriptIndexWorker` забирает jobs через `FOR UPDATE SKIP LOCKED`, обновляет lease
  heartbeat-ом и атомарно заменяет фрагменты только после успешной индексации.
- `video_search_indexes` хранит независимый от media status жизненный цикл индекса;
  `video_fragments` — таймкоды, transcript, язык и `vector(384)`.
- OCI-образ собирается Podman из `Containerfile`; сервисы описаны стандартным
  Compose-файлом `compose.yaml`.

Приложение не вызывает `create_all()`. Схема создаётся только Alembic-миграцией.
Readiness обращается к таблице `videos`, проверяет запись в upload directory и
исполняемость `ffprobe`.

На startup удаляются старые безопасно распознанные `.part` и orphan UUID-MP4,
а `.deleting` tombstones восстанавливаются или удаляются по состоянию БД. Файлы с
неожиданными именами и symlink не затрагиваются.

## Ограничения и граница безопасности

Поддерживается только MP4-контейнер с H.264, `yuv420p` и AAC-дорожкой. Нет
транскодирования, дедупликации, авторизации и object storage. Текущая модель ищет по
произнесённой речи: чисто визуальный момент, о котором никто не говорит, она не найдёт.
Порог «ничего не найдено» пока не калиброван на продуктовом eval-наборе, поэтому
dense retrieval может вернуть слабосвязанный top result.

Текущий MVP не имеет authentication и предназначен для локальной разработки или
закрытого demo-окружения. Перед публичным развёртыванием обязательны аутентификация,
ownership-check каждого видео, quota/rate limiting, malware scanning, отдельное
object storage и политика удаления данных.

Лимит upload повторно контролируется во время фактического чтения файла. Перед
публичным развёртыванием тот же body limit необходимо продублировать на reverse
proxy/ingress, чтобы остановить слишком большой multipart body до disk spooling.
