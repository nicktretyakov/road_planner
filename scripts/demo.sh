#!/usr/bin/env bash
# Демо-сценарий работы с запущенным сервером (по умолчанию http://127.0.0.1:8080).
# Сначала запустите сервер: cargo run --release
set -e
BASE="${BASE:-http://127.0.0.1:8080}"

echo "== 1. Загружаем демо-заказы Москвы =="
curl -sS -X POST "$BASE/api/demo/moscow" \
  -H 'Content-Type: application/json' -d '{"with_feedback": true}'
echo; echo

echo "== 2. Строим план на 3 курьеров с возвратом на склад =="
curl -sS -X POST "$BASE/api/plan" \
  -H 'Content-Type: application/json' \
  -d '{"couriers": 3, "start_clock": "09:00", "return_to_start": true, "use_osrm": true}'
echo; echo

echo "== 3. Курьер запросил следующий ближайший заказ =="
curl -sS -X POST "$BASE/api/nearest" \
  -H 'Content-Type: application/json' \
  -d '{"lat": 55.7539, "lon": 37.6208, "k": 3, "clock": "09:00"}'
echo; echo

echo "== 4. ML-статистика: бандит и SGD =="
curl -sS "$BASE/api/ml/stats"
echo; echo

echo "== 5. Курьер вернулся — сообщаем фактическое время поездки (обучение ML) =="
curl -sS -X POST "$BASE/api/ml/feedback" \
  -H 'Content-Type: application/json' \
  -d '{"legs": [{"from_lat": 55.7539, "from_lon": 37.6208, "to_lat": 55.7615, "to_lon": 37.6094, "actual_min": 7.5, "clock_min": 540}]}'
echo; echo
echo "Демо завершено. Все команды повторяемы — данные живут в памяти сервера."
