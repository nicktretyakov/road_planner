#!/usr/bin/env bash
# Дымовой прогон сервера: запускает сервер, проверяет все эндпоинты, останавливает.
# Использование: ./scripts/smoke.sh [порт]
set -e
PORT="${1:-8099}"
BASE="http://127.0.0.1:$PORT"
BIN="$(dirname "$0")/../target/release/route-planner"

export PORT OSRM_ENABLED="${OSRM_ENABLED:-0}"
export RUST_LOG="${RUST_LOG:-info}"

echo "=== Запуск сервера на порту $PORT (OSRM_ENABLED=$OSRM_ENABLED) ==="
"$BIN" > /tmp/route-planner-smoke.log 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT

for i in $(seq 1 50); do
  if curl -sS -m 1 "$BASE/health" > /dev/null 2>&1; then break; fi
  sleep 0.2
done

jq_pretty() { python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=1))" 2>/dev/null || cat; }

echo "=== 1. GET /health ==="
curl -sS "$BASE/health" | jq_pretty

echo "=== 2. POST /api/demo/moscow (с синтетическим ML-фидбеком) ==="
curl -sS -X POST "$BASE/api/demo/moscow" -H 'Content-Type: application/json' -d '{"with_feedback": true}' | jq_pretty

echo "=== 3. POST /api/plan: 3 курьера, замкнутые маршруты ==="
curl -sS -X POST "$BASE/api/plan" -H 'Content-Type: application/json' \
  -d '{"couriers": 3, "start": {"lat": 55.7539, "lon": 37.6208}, "start_clock": "09:00", "return_to_start": true, "use_osrm": true}' \
  | python3 -c '
import json,sys
p = json.load(sys.stdin)
print("plan_id:", p["plan_id"], "| стратегия (UCB1):", p["strategy"], "| источник матрицы:", p["distance_source"])
print("итого: %.2f км, %.1f мин, назначено %d/%d" % (p["total_distance_km"], p["total_duration_min"], p["orders_assigned"], p["orders_total"]))
for c in p["couriers"]:
    st = " -> ".join("#%s %s (%s)" % (s["order_id"], s["name"], s["arrival_clock"]) for s in c["stops"])
    ret = " -> склад (%s)" % c["return_leg"]["arrival_clock"] if c.get("return_leg") else ""
    print("курьер %s: %.2f км, %.0f мин: %s%s" % (c["courier_id"], c["distance_km"], c["duration_min"], st, ret))
print("след:")
for t in p["algorithm_trace"]: print("  -", t)
'

echo "=== 4. POST /api/tsp: замкнутый тур по 5 точкам ==="
curl -sS -X POST "$BASE/api/tsp" -H 'Content-Type: application/json' \
  -d '{"points": [{"lat": 55.7539, "lon": 37.6208}, {"lat": 55.7615, "lon": 37.6094}, {"lat": 55.7485, "lon": 37.5920}, {"lat": 55.7940, "lon": 37.5770}, {"lat": 55.7570, "lon": 37.6530}], "return_to_start": true}' \
  | jq_pretty

echo "=== 5. POST /api/nearest: ближайший заказ от текущей позиции ==="
curl -sS -X POST "$BASE/api/nearest" -H 'Content-Type: application/json' \
  -d '{"lat": 55.7539, "lon": 37.6208, "k": 3, "clock": "09:00"}' | jq_pretty

echo "=== 6. POST /api/dijkstra: кратчайший путь по сетке дорог ==="
curl -sS -X POST "$BASE/api/dijkstra" -H 'Content-Type: application/json' \
  -d '{"from": {"lat": 55.7539, "lon": 37.6208}, "to": {"lat": 55.8220, "lon": 37.6400}, "step_km": 1.0}' \
  | python3 -c '
import json,sys
d = json.load(sys.stdin)
print("путь: %.2f км по сетке vs %.2f км по прямой; узлов %d, рёбер проверено %d, шаг %.2f км, точек в пути %d" % (
    d["distance_km"], d["straight_line_km"], d["grid_nodes"], d["grid_edges_checked"], d["step_km"], len(d["path"])))'

echo "=== 7. POST /api/ml/feedback: обучение на фактической поездке ==="
curl -sS -X POST "$BASE/api/ml/feedback" -H 'Content-Type: application/json' \
  -d '{"legs": [{"from_lat": 55.7539, "from_lon": 37.6208, "to_lat": 55.8220, "to_lon": 37.6400, "actual_min": 28.0, "clock_min": 540}]}' | jq_pretty

echo "=== 8. GET /api/ml/stats ==="
curl -sS "$BASE/api/ml/stats" | jq_pretty

echo "=== 9. Ошибка: план без заказов (после очистки) ==="
curl -sS -X DELETE "$BASE/api/orders" > /dev/null
curl -sS -X POST "$BASE/api/plan" -H 'Content-Type: application/json' -d '{"couriers": 2}'
echo

echo "=== Smoke OK, сервер остановлен. Лог: /tmp/route-planner-smoke.log ==="
