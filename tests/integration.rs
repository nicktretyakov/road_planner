//! Интеграционные тесты REST API (офлайн: OSRM выключен, состояние в памяти).

use actix_http::Request;
use actix_web::dev::{Service, ServiceResponse};
use actix_web::test;
use actix_web::App;
use actix_web::web::Data;
use route_planner::api;
use route_planner::storage::AppState;
use serde_json::{json, Value};

/// Сервис поверх ЕДИНОГО состояния — все вызовы теста видят одни данные.
async fn app(state: Data<AppState>) -> impl Service<Request, Response = ServiceResponse, Error = actix_web::Error> {
    test::init_service(App::new().configure(move |c| api::configure(c, state.clone()))).await
}

async fn setup() -> Data<AppState> {
    Data::new(AppState::new_offline())
}

async fn post_json(state: Data<AppState>, uri: &str, body: Value) -> Value {
    let svc = app(state).await;
    let req = test::TestRequest::post().uri(uri).set_json(body).to_request();
    let resp = test::call_service(&svc, req).await;
    assert!(
        resp.status().is_success(),
        "POST {uri} должен быть успешен, статус {}",
        resp.status()
    );
    test::read_body_json(resp).await
}

async fn post_json_fail(state: Data<AppState>, uri: &str, body: Value) -> (u16, Value) {
    let svc = app(state).await;
    let req = test::TestRequest::post().uri(uri).set_json(body).to_request();
    let resp = test::call_service(&svc, req).await;
    let status = resp.status().as_u16();
    let body: Value = test::read_body_json(resp).await;
    (status, body)
}

async fn get_json(state: Data<AppState>, uri: &str) -> (u16, Value) {
    let svc = app(state).await;
    let req = test::TestRequest::get().uri(uri).to_request();
    let resp = test::call_service(&svc, req).await;
    let status = resp.status().as_u16();
    let body: Value = test::read_body_json(resp).await;
    (status, body)
}

#[actix_web::test]
async fn health_ok() {
    let state = setup().await;
    let (status, body) = get_json(state, "/health").await;
    assert_eq!(status, 200);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["service"], "route-planner");
}

#[actix_web::test]
async fn orders_crud_flow() {
    let state = setup().await;
    let created = post_json(
        state.clone(),
        "/api/orders",
        json!({
            "name": "Тверская, 7",
            "lat": 55.7615,
            "lon": 37.6094,
            "time_window": "09:00-11:00"
        }),
    )
    .await;
    assert_eq!(created["created"], 1);
    let id = created["orders"][0]["id"].as_u64().unwrap();
    assert_eq!(created["orders"][0]["window"], "09:00-11:00");

    let (_, list) = get_json(state.clone(), "/api/orders").await;
    assert_eq!(list["count"], 1);

    let svc = app(state.clone()).await;
    let req = test::TestRequest::delete()
        .uri(&format!("/api/orders/{id}"))
        .to_request();
    let resp = test::call_service(&svc, req).await;
    assert!(resp.status().is_success());

    let svc = app(state).await;
    let req = test::TestRequest::delete().uri("/api/orders/999").to_request();
    let resp = test::call_service(&svc, req).await;
    assert_eq!(resp.status().as_u16(), 404);
}

#[actix_web::test]
async fn bad_coordinates_rejected() {
    let state = setup().await;
    let (status, body) = post_json_fail(
        state,
        "/api/orders",
        json!({
            "name": "Плохой заказ",
            "lat": 123.0,
            "lon": 37.0,
            "time_window": "09:00-11:00"
        }),
    )
    .await;
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("Широта"));
}

#[actix_web::test]
async fn bad_time_window_rejected() {
    let state = setup().await;
    let svc = app(state).await;
    let req = test::TestRequest::post()
        .uri("/api/orders")
        .set_json(json!({
            "name": "Окно вверх ногами",
            "lat": 55.0,
            "lon": 37.0,
            "time_window": "15:00-09:00"
        }))
        .to_request();
    let resp = test::call_service(&svc, req).await;
    assert_eq!(resp.status().as_u16(), 400);
    // тело ошибки может быть plain-text (дефолтный обработчик JSON-ошибок actix)
    let body = test::read_body(resp).await;
    assert!(!body.is_empty());
}

#[actix_web::test]
async fn demo_and_full_plan() {
    let state = setup().await;
    let body = post_json(state.clone(), "/api/demo/moscow", json!({"with_feedback": true})).await;
    assert_eq!(body["loaded"], 12);
    assert!(body["synthetic_feedback"].as_u64().unwrap() > 200);

    let plan = post_json(
        state.clone(),
        "/api/plan",
        json!({
            "couriers": 3,
            "start": {"lat": 55.7539, "lon": 37.6208},
            "start_clock": "09:00",
            "return_to_start": true,
            "use_osrm": false
        }),
    )
    .await;

    assert_eq!(plan["orders_total"], 12);
    assert_eq!(plan["orders_assigned"], 12);
    assert_eq!(plan["distance_source"], "haversine+ml");
    assert_eq!(plan["unassigned"].as_array().unwrap().len(), 0);
    let couriers = plan["couriers"].as_array().unwrap();
    assert_eq!(couriers.len(), 3);
    for c in couriers {
        assert!(c["all_on_time"].as_bool().unwrap(), "все доставки должны быть в окнах");
        assert!(c["return_leg"].is_object(), "маршрут замкнут на склад");
    }
    assert!(plan["total_distance_km"].as_f64().unwrap() > 0.0);
    assert!(!plan["strategy"].as_str().unwrap().is_empty());
    assert!(!plan["algorithm_trace"].as_array().unwrap().is_empty());

    let plan2 = post_json(state.clone(), "/api/plan", json!({"couriers": 2, "use_osrm": false})).await;
    assert_eq!(plan2["orders_assigned"], 12);

    // история планов копится
    let (_, plans) = get_json(state, "/api/plans").await;
    assert_eq!(plans["count"], 2);
}

#[actix_web::test]
async fn plan_without_orders_fails() {
    let state = setup().await;
    let (status, body) = post_json_fail(state, "/api/plan", json!({"couriers": 1})).await;
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("Нет заказов"));
}

#[actix_web::test]
async fn tsp_closed_tour() {
    let state = setup().await;
    let body = post_json(
        state,
        "/api/tsp",
        json!({
            "points": [
                {"lat": 55.7539, "lon": 37.6208},
                {"lat": 55.7615, "lon": 37.6094},
                {"lat": 55.7485, "lon": 37.5920},
                {"lat": 55.7940, "lon": 37.5770},
                {"lat": 55.7570, "lon": 37.6530}
            ],
            "return_to_start": true,
            "method": "nn_2opt_oropt"
        }),
    )
    .await;
    assert_eq!(body["tour"].as_array().unwrap().len(), 4);
    assert_eq!(body["method"], "nn_2opt_oropt");
    assert_eq!(body["closed"], true);
    assert!(body["length_km"].as_f64().unwrap() > 0.0);
    assert!(
        body["baseline_nn_km"].as_f64().unwrap() >= body["length_km"].as_f64().unwrap() - 1e-6,
        "улучшенный тур не хуже жадного"
    );
}

#[actix_web::test]
async fn tsp_unknown_method_rejected() {
    let state = setup().await;
    let (status, body) = post_json_fail(
        state,
        "/api/tsp",
        json!({
            "points": [{"lat": 55.0, "lon": 37.0}, {"lat": 55.1, "lon": 37.1}],
            "method": "магия"
        }),
    )
    .await;
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("Неизвестная стратегия"));
}

#[actix_web::test]
async fn nearest_endpoint_flow() {
    let state = setup().await;
    post_json(state.clone(), "/api/demo/moscow", json!({})).await;
    let body = post_json(
        state,
        "/api/nearest",
        json!({"lat": 55.7539, "lon": 37.6208, "k": 3, "clock": "09:00"}),
    )
    .await;
    let items = body["items"].as_array().unwrap();
    assert_eq!(items.len(), 3);
    // первый — ближайший: Тверская 7 (≈0.8 км), а не ВДНХ (≈8 км)
    assert_eq!(items[0]["name"], "Тверская, 7");
    assert!(items[0]["distance_km"].as_f64().unwrap() < 2.0);
    assert!(items[0]["eta_min"].as_f64().unwrap() > 0.0);
    let d0 = items[0]["distance_km"].as_f64().unwrap();
    let d1 = items[1]["distance_km"].as_f64().unwrap();
    assert!(d0 <= d1, "элементы упорядочены по расстоянию");
}

#[actix_web::test]
async fn dijkstra_finds_grid_path() {
    let state = setup().await;
    let body = post_json(
        state,
        "/api/dijkstra",
        json!({
            "from": {"lat": 55.7539, "lon": 37.6208},
            "to": {"lat": 55.8000, "lon": 37.7000},
            "step_km": 1.0
        }),
    )
    .await;
    assert!(
        body["distance_km"].as_f64().unwrap() >= body["straight_line_km"].as_f64().unwrap(),
        "сетевой путь не короче прямой линии"
    );
    assert!(body["path"].as_array().unwrap().len() >= 2);
    assert!(body["grid_nodes"].as_u64().unwrap() > 0);
}

#[actix_web::test]
async fn ml_feedback_trains_and_stats_reflects() {
    let state = setup().await;
    let fb = post_json(
        state.clone(),
        "/api/ml/feedback",
        json!({
            "legs": [
                {"from_lat": 55.7539, "from_lon": 37.6208, "to_lat": 55.7615, "to_lon": 37.6094, "actual_min": 6.0, "clock_min": 540},
                {"from_lat": 55.7615, "from_lon": 37.6094, "to_lat": 55.7940, "to_lon": 37.5770, "actual_min": 20.0, "clock_min": 1080}
            ]
        }),
    )
    .await;
    assert_eq!(fb["updated"], 2);
    assert_eq!(fb["n_samples"], 2);

    let (_, stats) = get_json(state, "/api/ml/stats").await;
    assert_eq!(stats["feedback_count"], 2);
    assert_eq!(stats["bandit"].as_array().unwrap().len(), 4);
    assert_eq!(stats["sgd"]["n_samples"], 2);
    assert_eq!(stats["osrm"]["enabled"], false);
}

#[actix_web::test]
async fn bandit_learns_across_plans() {
    let state = setup().await;
    post_json(state.clone(), "/api/demo/moscow", json!({})).await;
    for _ in 0..6 {
        post_json(state.clone(), "/api/plan", json!({"couriers": 2, "use_osrm": false})).await;
    }
    let (_, stats) = get_json(state, "/api/ml/stats").await;
    assert_eq!(stats["plan_count"], 6);
    assert!(stats["bandit_total_rounds"].as_u64().unwrap() >= 4);
}
