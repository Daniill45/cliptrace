# ClipTrace MVP

ClipTrace загружает совместимые MP4, хранит метаданные в PostgreSQL, показывает
видео в удобном локальном интерфейсе и воспроизводит фиксированный demo-фрагмент
`00:00:05.000–00:00:10.000`. Настоящего ML-поиска в этой версии нет: текст запроса
пока не влияет на результат.

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

Compose поднимает PostgreSQL, одноразово применяет Alembic-миграцию и только после
её успеха запускает приложение в фоне. Откройте:

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

Логи API доступны через `make logs`, состояние контейнеров — через `make ps`.
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
3. После проверки нажмите «Получить»; описание момента в demo-режиме необязательно.
4. Нажмите «Смотреть отрывок»: плеер начнёт с 5-й секунды и остановится на 10-й.

Недавние загрузки остаются в списке после обновления страницы. Поддерживаются MP4
с H.264/yuv420p и AAC, размером до 500 МБ и длительностью не менее 10 секунд.

## Работа через API

Загрузка валидного H.264/yuv420p MP4 с AAC-аудиодорожкой длительностью не менее
10 секунд:

```bash
curl -i \
  -F 'title=Demo interview' \
  -F 'file=@./sample.mp4;type=video/mp4' \
  http://localhost:8000/api/v1/videos
```

Идентификатор из ответа используется во всех следующих запросах:

```bash
curl http://localhost:8000/api/v1/videos/<VIDEO_ID>

curl -X POST \
  -H 'Content-Type: application/json' \
  -d '{"query":"момент, где говорят о провале","limit":5}' \
  http://localhost:8000/api/v1/videos/<VIDEO_ID>/search

curl -i \
  -H 'Range: bytes=0-1023' \
  http://localhost:8000/api/v1/videos/<VIDEO_ID>/content

curl -i -X DELETE http://localhost:8000/api/v1/videos/<VIDEO_ID>
```

При default-конфигурации search возвращает ровно один результат с
`start_seconds=5.0`, `end_seconds=10.0`, `mode="mock"` и `score=null`.

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
2. Убедитесь, что карточка результата показывает `00:00:05.000–00:00:10.000`.
3. Нажмите «Смотреть отрывок»: воспроизведение должно начаться на 5-й секунде и
   остановиться на 10-й; «Повторить отрывок» снова запускает его с 5-й секунды.

Backend отдаёт исходное видео с HTTP byte ranges. Ограничение диапазона по времени
выполняет плеер; отдельный физически обрезанный MP4 в MVP не создаётся.

## Конфигурация и режимы

Настройки читаются из environment variables через `pydantic-settings`; полный набор
есть в `.env.example`. Невалидные лимиты, mock-тайминги, search engine или CORS origins
останавливают создание приложения.

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

```mermaid
flowchart LR
    UI["Web UI"] --> API["FastAPI endpoints"]
    API --> VIDEO["VideoService"]
    API --> SEARCH["SearchService"]
    VIDEO --> PROBE["MediaProbe / ffprobe"]
    VIDEO --> STORAGE["LocalStorage"]
    VIDEO --> REPO["VideoRepository"]
    REPO --> DB[("PostgreSQL")]
    SEARCH --> ENGINE["FragmentSearchEngine"]
    ENGINE --> MOCK["Current mock engine"]
    ENGINE -. "same interface" .-> ML["Future ML engine"]
```

### Поток данных

1. `POST /api/v1/videos` потоково записывает multipart-файл во временный `.part`,
   одновременно контролируя размер и вычисляя SHA-256.
2. `MediaProbe` проверяет контейнер, кодеки и длительность через `ffprobe`. После
   успешной проверки файл атомарно переименовывается в UUID-based `.mp4`, а метаданные
   сохраняются в PostgreSQL. Если запись в БД завершается ошибкой, финальный файл
   удаляется.
3. `POST /api/v1/videos/{video_id}/search` передаёт видео, запрос и лимит в
   `FragmentSearchEngine`. Текущий `MockFixedTimestampSearchEngine` намеренно не
   анализирует содержимое видео и возвращает интервал 5–10 секунд: он проверяет
   контракт API и сценарий воспроизведения, а не семантическую релевантность.
4. Браузер запрашивает исходный MP4 с HTTP Range, начинает воспроизведение с
   `start_seconds` и останавливает его на `end_seconds`; отдельный файл-фрагмент не
   создаётся.

### Компоненты

- Endpoint-слой адаптирует HTTP-контракт, включая Range-заголовки, и делегирует
  ресурсную логику application services.
- `VideoService` владеет транзакционными границами загрузки и tombstone-удаления.
- `VideoRepository` отвечает только за SQLAlchemy persistence.
- `LocalStorage` владеет `.part`, UUID-based final files, `.deleting` tombstones,
  безопасным `stat/open` и потоковой выдачей.
- `MediaProbe` запускает фиксированный argv `ffprobe` без shell, ограничивает время и
  объём вывода и нормализует только необходимые поля.
- `SearchService` зависит от `FragmentSearchEngine`; текущая реализация — явно
  именованный `MockFixedTimestampSearchEngine`.
- OCI-образ собирается Podman из `Containerfile`; сервисы описаны стандартным
  Compose-файлом `compose.yaml`.

Приложение не вызывает `create_all()`. Схема создаётся только Alembic-миграцией.
Readiness обращается к таблице `videos`, проверяет запись в upload directory и
исполняемость `ffprobe`.

На startup удаляются старые безопасно распознанные `.part` и orphan UUID-MP4,
а `.deleting` tombstones восстанавливаются или удаляются по состоянию БД. Файлы с
неожиданными именами и symlink не затрагиваются.

## Ограничения и граница безопасности

Поддерживается только MP4-контейнер с H.264, `yuv420p`, хотя бы одной AAC-дорожкой и
длительностью, позволяющей вернуть mock-фрагмент. Нет транскодирования,
дедупликации, очередей, индексации, ASR, embeddings, авторизации и object storage.
Будущая ML-реализация должна реализовать существующий `FragmentSearchEngine` и
добавить отдельный indexing pipeline и `search_status`, не меняя текущий search API.

Текущий MVP не имеет authentication и предназначен для локальной разработки или
закрытого demo-окружения. Перед публичным развёртыванием обязательны аутентификация,
ownership-check каждого видео, quota/rate limiting, malware scanning, отдельное
object storage и политика удаления данных.

Лимит upload повторно контролируется во время фактического чтения файла. Перед
публичным развёртыванием тот же body limit необходимо продублировать на reverse
proxy/ingress, чтобы остановить слишком большой multipart body до disk spooling.
