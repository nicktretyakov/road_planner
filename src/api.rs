//! REST API на actix-web: заказы, планирование, TSP, ближайший заказ,
//! Дейкстра, ML-обратная связь и статистика, демо-набор Москвы.

use std::sync::atomic::Ordering;
use std::time::Instant;

use actix_web::{get, post, web, HttpResponse, Responder};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::algorithms::nearest::nearest_route;
use crate::algorithms::tsp::{solve_tsp, ImproveMode};
use crate::geo_core::{fmt_hh_mm, haversine_km, parse_hh_mm};
use crate::graph::{shortest_path, Graph, GridGraph};
use crate::ml::ClusterInfo;
use crate::models::*;
use crate::planner::build_plan;
use crate::storage::AppState;

/// DTO запроса /api/tsp.
#[derive(Debug, Deserialize)]
pub struct TspRequest {
    /// Точки маршрута (lat, lon). Первая точка — старт/депо.
    pub points: Vec<LatLng>,
    /// Замкнуть тур возвратом в старт (по умолчанию да).
    #[serde(default = "default_true")]
    pub return_to_start: bool,
    /// Стратегия улучшения: auto (бандит) или nn_2opt / nn_oropt / nn_2opt_oropt / nn_sa.
    #[serde(default)]
    pub method: Option<String>,
}

fn default_true() -> bool {
    true
}

/// DTO запроса /api/nearest.
#[derive(Debug, Deserialize)]
pub struct NearestRequest {
    pub lat: f64,
    pub lon: f64,
    /// Сколько ближайших показать (по умолчанию 5).
    #[serde(default)]
    pub k: Option<usize>,
    /// Текущее время "ЧЧ:ММ" (для ML-ETA); по умолчанию — UTC-время сервера.
    #[serde(default)]
    pub clock: Option<String>,
}

/// Элемент статистики бандита.
#[derive(Debug, Serialize)]
pub struct ArmDto {
    pub name: String,
    pub description: String,
    pub pulls: u64,
    pub mean_reward: f64,
}

#[derive(Debug, Serialize)]
pub struct SgdDto {
    /// Веса [свободный член, за 10 км, за «час пик»].
    pub weights: [f64; 3],
    pub n_samples: u64,
    pub mse_min_sq: f64,
    /// Прогноз для примера: 5 км в полдень, мин.
    pub sample_eta_5km_min: f64,
}

#[derive(Debug, Serialize)]
pub struct MlStatsResponse {
    pub sgd: SgdDto,
    pub bandit: Vec<ArmDto>,
    pub bandit_total_rounds: u64,
    pub plan_count: u64,
    pub feedback_count: u64,
    pub last_clusters: Option<ClusterInfo>,
    pub osrm: OsrmDto,
}

#[derive(Debug, Serialize)]
pub struct OsrmDto {
    pub enabled: bool,
    pub base: String,
    pub last_status: String,
}

#[derive(Debug, Serialize)]
pub struct DemoResponse {
    pub loaded: usize,
    pub synthetic_feedback: usize,
    pub note: String,
}

fn error_400(msg: String) -> HttpResponse {
    HttpResponse::BadRequest().json(serde_json::json!({ "error": msg }))
}

/// Стратегия из имени (None — не распознано, Some — даже для auto).
fn mode_from_name(name: &str) -> Option<ImproveMode> {
    match name {
        "nn_2opt" => Some(ImproveMode::TwoOpt),
        "nn_oropt" => Some(ImproveMode::OrOpt),
        "nn_2opt_oropt" => Some(ImproveMode::TwoOptOrOpt),
        "nn_sa" => Some(ImproveMode::Annealing),
        _ => None,
    }
}

// ---------- health ----------

#[get("/health")]
async fn health(state: web::Data<AppState>) -> impl Responder {
    let orders = state.orders.read().unwrap().len();
    let plans = state.plans.read().unwrap().len();
    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "service": "route-planner",
        "orders": orders,
        "plans": plans,
        "osrm_enabled": state.osrm.enabled,
    }))
}

// ---------- заказы ----------

#[post("/api/orders")]
async fn create_orders(state: web::Data<AppState>, body: web::Json<OneOrManyOrders>) -> impl Responder {
    let inputs: Vec<OrderInput> = match body.into_inner() {
        OneOrManyOrders::One(o) => vec![o],
        OneOrManyOrders::Many(v) => v,
    };
    if inputs.is_empty() {
        return error_400("Передан пустой список заказов".into());
    }
    let mut created = Vec::with_capacity(inputs.len());
    for (i, input) in inputs.into_iter().enumerate() {
        if input.name.trim().is_empty() {
            return error_400(format!("Заказ #{i}: пустое имя"));
        }
        if let Err(e) = validate_latlng(input.lat, input.lon) {
            return error_400(format!("Заказ '{}': {e}", input.name));
        }
        if input.service_min < 0.0 || input.weight_kg < 0.0 {
            return error_400(format!(
                "Заказ '{}': service_min и weight_kg должны быть ≥ 0",
                input.name
            ));
        }
        let id = input
            .id
            .unwrap_or_else(|| state.next_order_id.fetch_add(1, Ordering::Relaxed));
        created.push(Order {
            id,
            name: input.name.trim().to_string(),
            lat: input.lat,
            lon: input.lon,
            window: input.time_window,
            service_min: input.service_min,
            weight_kg: input.weight_kg,
        });
    }
    {
        let mut o = state.orders.write().unwrap();
        // если id переданы явно — заменяем существующие с тем же id
        for c in &created {
            o.retain(|x| x.id != c.id);
        }
        o.extend(created.clone());
    }
    HttpResponse::Created().json(serde_json::json!({ "created": created.len(), "orders": created }))
}

#[get("/api/orders")]
async fn list_orders(state: web::Data<AppState>) -> impl Responder {
    let orders = state.orders.read().unwrap().clone();
    HttpResponse::Ok().json(serde_json::json!({ "count": orders.len(), "orders": orders }))
}

#[get("/api/plans")]
async fn list_plans(state: web::Data<AppState>) -> impl Responder {
    let plans = state.plans.read().unwrap().clone();
    HttpResponse::Ok().json(serde_json::json!({ "count": plans.len(), "plans": plans }))
}

#[actix_web::delete("/api/orders/{id}")]
async fn delete_order(state: web::Data<AppState>, path: web::Path<u64>) -> impl Responder {
    let id = path.into_inner();
    let mut o = state.orders.write().unwrap();
    let before = o.len();
    o.retain(|x| x.id != id);
    if o.len() == before {
        return HttpResponse::NotFound()
            .json(serde_json::json!({ "error": format!("Заказ {id} не найден") }));
    }
    HttpResponse::Ok().json(serde_json::json!({ "deleted": id, "remaining": o.len() }))
}

#[actix_web::delete("/api/orders")]
async fn clear_orders(state: web::Data<AppState>) -> impl Responder {
    state.orders.write().unwrap().clear();
    HttpResponse::Ok().json(serde_json::json!({ "cleared": true }))
}

// ---------- планирование ----------

#[post("/api/plan")]
async fn plan(state: web::Data<AppState>, body: web::Json<PlanRequest>) -> impl Responder {
    match build_plan(&state, &body) {
        Ok(p) => {
            state.plans.write().unwrap().push(p.clone());
            HttpResponse::Ok().json(p)
        }
        Err(e) => error_400(e),
    }
}

// ---------- TSP ----------

#[post("/api/tsp")]
async fn tsp(state: web::Data<AppState>, body: web::Json<TspRequest>) -> impl Responder {
    let req = body.into_inner();
    if req.points.len() < 2 {
        return error_400("Для TSP нужно минимум 2 точки (первая — старт)".into());
    }
    for (i, p) in req.points.iter().enumerate() {
        if let Err(e) = validate_latlng(p.lat, p.lon) {
            return error_400(format!("Точка #{i}: {e}"));
        }
    }

    // выбор стратегии: явное имя или через UCB1-бандита
    let (mode, bandit_arm) = match req.method.as_deref() {
        Some(name) if name != "auto" => match mode_from_name(name) {
            Some(m) => (m, None),
            None => {
                return error_400(format!(
                    "Неизвестная стратегия '{name}'. Доступны: auto, nn_2opt, nn_oropt, nn_2opt_oropt, nn_sa"
                ));
            }
        },
        _ => {
            let (arm, name) = {
                let ml = state.ml.lock().unwrap();
                let arm = ml.bandit.select();
                (arm, ml.bandit.arm_name(arm).to_string())
            };
            let mode = mode_from_name(&name).unwrap_or(ImproveMode::TwoOpt);
            (mode, Some(arm))
        }
    };

    let pts: Vec<(f64, f64)> = req.points.iter().map(LatLng::as_tuple).collect();
    let t0 = Instant::now();
    let res = solve_tsp(&pts, req.return_to_start, mode, 42);
    let elapsed = t0.elapsed().as_millis();

    let improvement = if res.baseline_km > 1e-9 {
        ((res.baseline_km - res.length_km) / res.baseline_km * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };
    // обучение бандита и на TSP-вызовах (награда — % сокращения тура)
    if let Some(arm) = bandit_arm {
        let mut ml = state.ml.lock().unwrap();
        ml.bandit.update(arm, improvement);
    }

    let tour: Vec<LatLng> = res.seq.iter().map(|&i| req.points[i]).collect();
    HttpResponse::Ok().json(TspResponse {
        tour,
        length_km: res.length_km,
        baseline_nn_km: res.baseline_km,
        improvement_pct: improvement,
        method: res.mode.name().to_string(),
        iterations: res.iterations,
        closed: req.return_to_start,
        elapsed_ms: elapsed,
    })
}

// ---------- ближайший заказ ----------

#[post("/api/nearest")]
async fn nearest(state: web::Data<AppState>, body: web::Json<NearestRequest>) -> impl Responder {
    let req = body.into_inner();
    if let Err(e) = validate_latlng(req.lat, req.lon) {
        return error_400(e);
    }
    let orders = state.orders.read().unwrap().clone();
    if orders.is_empty() {
        return error_400(
            "Заказов нет: добавьте через POST /api/orders или POST /api/demo/moscow".into(),
        );
    }
    let clock_min = match &req.clock {
        Some(s) => match parse_hh_mm(s) {
            Ok(m) => m,
            Err(e) => return error_400(e),
        },
        None => utc_minutes_now(),
    };
    let pts: Vec<(f64, f64)> = orders.iter().map(Order::latlng).collect();
    let k = req.k.unwrap_or(5).clamp(1, orders.len());

    // жадный «ближайший сосед» от текущего местоположения:
    // каждый следующий пункт — ближайший к предыдущей позиции
    let seq = nearest_route((req.lat, req.lon), &pts);
    let items: Vec<NearestItem> = seq
        .iter()
        .take(k)
        .map(|&i| {
            let o = &orders[i];
            let d = haversine_km((req.lat, req.lon), o.latlng());
            let eta = {
                let ml = state.ml.lock().unwrap();
                ml.sgd.predict(d, clock_min)
            };
            NearestItem {
                order_id: o.id,
                name: o.name.clone(),
                lat: o.lat,
                lon: o.lon,
                distance_km: d,
                eta_min: eta,
                window: o.window,
            }
        })
        .collect();

    HttpResponse::Ok().json(NearestResponse {
        from: LatLng { lat: req.lat, lon: req.lon },
        clock: fmt_hh_mm(clock_min),
        items,
    })
}

// ---------- Дейкстра ----------

#[post("/api/dijkstra")]
async fn dijkstra(body: web::Json<DijkstraRequest>) -> impl Responder {
    let req = body.into_inner();
    if let Err(e) = validate_latlng(req.from.lat, req.from.lon) {
        return error_400(format!("from: {e}"));
    }
    if let Err(e) = validate_latlng(req.to.lat, req.to.lon) {
        return error_400(format!("to: {e}"));
    }
    if req.step_km <= 0.0 || req.step_km > 50.0 {
        return error_400("step_km должен быть в диапазоне (0; 50]".into());
    }
    let from = req.from.as_tuple();
    let to = req.to.as_tuple();
    let grid = GridGraph::between(from, to, req.step_km);
    let s = grid.nearest_node(from);
    let t = grid.nearest_node(to);
    match shortest_path(&grid, s, t) {
        Some(res) => {
            let path: Vec<LatLng> = res
                .path
                .iter()
                .map(|&u| {
                    let (lat, lon) = grid.coords(u);
                    LatLng { lat, lon }
                })
                .collect();
            HttpResponse::Ok().json(DijkstraResponse {
                distance_km: res.distance,
                path,
                grid_nodes: grid.num_nodes(),
                grid_edges_checked: res.edges_checked,
                step_km: grid.step,
                straight_line_km: haversine_km(from, to),
            })
        }
        None => error_400("Путь не найден".into()),
    }
}

// ---------- ML ----------

#[post("/api/ml/feedback")]
async fn ml_feedback(state: web::Data<AppState>, body: web::Json<FeedbackRequest>) -> impl Responder {
    let req = body.into_inner();
    if req.legs.is_empty() {
        return error_400("Передан пустой список поездок".into());
    }
    {
        let mut ml = state.ml.lock().unwrap();
        for leg in &req.legs {
            if let Err(e) = validate_latlng(leg.from_lat, leg.from_lon) {
                return error_400(format!("from: {e}"));
            }
            if let Err(e) = validate_latlng(leg.to_lat, leg.to_lon) {
                return error_400(format!("to: {e}"));
            }
            if leg.actual_min <= 0.0 || leg.actual_min > 1440.0 {
                return error_400(format!(
                    "actual_min должен быть в (0; 1440], получено {}",
                    leg.actual_min
                ));
            }
            let d = haversine_km((leg.from_lat, leg.from_lon), (leg.to_lat, leg.to_lon));
            ml.sgd.update(d, leg.clock_min, leg.actual_min);
            ml.feedback_count += 1;
        }
    }
    let ml = state.ml.lock().unwrap();
    HttpResponse::Ok().json(serde_json::json!({
        "updated": req.legs.len(),
        "sgd_weights": ml.sgd.w,
        "n_samples": ml.sgd.n_samples,
        "mse": ml.sgd.mse,
    }))
}

#[get("/api/ml/stats")]
async fn ml_stats(state: web::Data<AppState>) -> impl Responder {
    let ml = state.ml.lock().unwrap();
    let arms: Vec<ArmDto> = ml
        .bandit
        .stats()
        .into_iter()
        .map(|s| {
            let name = s.name.clone();
            ArmDto {
                description: crate::ml::MlEngine::arm_description(&name).to_string(),
                name,
                pulls: s.pulls,
                mean_reward: s.mean_reward,
            }
        })
        .collect();
    HttpResponse::Ok().json(MlStatsResponse {
        sgd: SgdDto {
            weights: ml.sgd.w,
            n_samples: ml.sgd.n_samples,
            mse_min_sq: ml.sgd.mse,
            sample_eta_5km_min: ml.sgd.predict(5.0, 720),
        },
        bandit: arms,
        bandit_total_rounds: ml.bandit.total_rounds(),
        plan_count: ml.plan_count,
        feedback_count: ml.feedback_count,
        last_clusters: ml.last_clusters.clone(),
        osrm: OsrmDto {
            enabled: state.osrm.enabled,
            base: state.osrm.base.clone(),
            last_status: state.osrm.last_status.lock().unwrap().clone(),
        },
    })
}

// ---------- демо Москвы ----------

#[post("/api/demo/moscow")]
async fn demo_moscow(state: web::Data<AppState>, body: web::Json<DemoRequest>) -> impl Responder {
    let want_feedback = body.with_feedback;
    let demo: Vec<(&str, f64, f64, &str, f64, f64)> = vec![
        ("Тверская, 7", 55.7615, 37.6094, "09:00-11:00", 8.0, 2.0),
        ("Арбат, 24", 55.7485, 37.5920, "09:30-12:30", 10.0, 1.5),
        ("Кутузовский просп., 32", 55.7401, 37.5306, "10:00-13:00", 12.0, 3.0),
        ("Ленинский просп., 109", 55.6602, 37.4927, "12:00-16:00", 15.0, 4.0),
        ("Профсоюзная ул., 41", 55.6700, 37.5600, "11:00-14:00", 10.0, 2.5),
        ("Ленинградский просп., 52", 55.7940, 37.5770, "09:00-12:00", 9.0, 2.0),
        ("Ленинградский просп., 36", 55.7840, 37.5650, "09:00-13:00", 8.0, 1.0),
        ("Волоколамское ш., 15", 55.8050, 37.5140, "10:00-14:00", 11.0, 2.5),
        ("Земляной Вал, 33", 55.7570, 37.6530, "13:00-17:00", 7.0, 1.0),
        ("Земляной Вал, 24", 55.7590, 37.6570, "12:00-15:00", 8.0, 1.5),
        ("Просп. Мира, 119 (ВДНХ)", 55.8220, 37.6400, "14:00-18:00", 10.0, 3.0),
        ("Русаковская ул., 24", 55.7860, 37.6800, "15:00-19:00", 9.0, 2.0),
    ];
    let orders: Vec<Order> = demo
        .iter()
        .enumerate()
        .map(|(i, (name, lat, lon, window, service, weight))| Order {
            id: (i + 1) as u64,
            name: name.to_string(),
            lat: *lat,
            lon: *lon,
            window: crate::geo_core::parse_window(window).expect("демо-окна корректны"),
            service_min: *service,
            weight_kg: *weight,
        })
        .collect();

    let mut synthetic = 0usize;
    if want_feedback {
        // Синтетическая история поездок по Москве: время ≈ 4 + 2.2·км + 8·пик ± 12%.
        // ЯВНО помечено как демо-данные — служит для демонстрации онлайн-обучения SGD.
        let mut rng = StdRng::seed_from_u64(777);
        let mut ml = state.ml.lock().unwrap();
        for _ in 0..400 {
            let a = rng.gen_range(0..orders.len());
            let b = rng.gen_range(0..orders.len());
            if a == b {
                continue;
            }
            let d = haversine_km(orders[a].latlng(), orders[b].latlng());
            if d < 0.2 {
                continue;
            }
            let clock = rng.gen_range(7 * 60..20 * 60);
            let peak = crate::ml::sgd::SgdModel::peak_factor(clock);
            let noise = 0.88 + rng.gen::<f64>() * 0.24;
            let actual = ((4.0 + 2.2 * d + 8.0 * peak) * noise).max(d * 0.8);
            ml.sgd.update(d, clock, actual);
            ml.feedback_count += 1;
            synthetic += 1;
        }
    }

    {
        let mut o = state.orders.write().unwrap();
        *o = orders;
    }
    state.next_order_id.store(13, Ordering::Relaxed);
    HttpResponse::Ok().json(DemoResponse {
        loaded: 12,
        synthetic_feedback: synthetic,
        note: "Загружены 12 заказов по Москве (склад — Красная площадь). Синтетические поездки помечены как демо-данные.".into(),
    })
}

/// Текущее время сервера в минутах от начала суток (UTC).
fn utc_minutes_now() -> u32 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    ((secs % 86_400) / 60) as u32
}

/// Конфигурация приложения: разделяемое состояние + все маршруты.
pub fn configure(cfg: &mut web::ServiceConfig, state: web::Data<AppState>) {
    cfg.app_data(state)
        .service(health)
        .service(create_orders)
        .service(list_orders)
        .service(clear_orders)
        .service(delete_order)
        .service(list_plans)
        .service(plan)
        .service(tsp)
        .service(nearest)
        .service(dijkstra)
        .service(ml_feedback)
        .service(ml_stats)
        .service(demo_moscow);
}
