//! Модели данных: заказы, курьеры, планы маршрутов и DTO REST API.

use crate::geo_core::TimeWindow;
use serde::{Deserialize, Serialize};

/// Заказ на доставку: географические координаты + временное окно.
#[derive(Debug, Clone, Serialize)]
pub struct Order {
    pub id: u64,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub window: TimeWindow,
    pub service_min: f64,
    pub weight_kg: f64,
}

impl Order {
    pub fn latlng(&self) -> (f64, f64) {
        (self.lat, self.lon)
    }
}

/// DTO приёма заказа. Если `id` не передан, сервер назначает его сам.
#[derive(Debug, Deserialize)]
pub struct OrderInput {
    pub id: Option<u64>,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub time_window: TimeWindow,
    #[serde(default = "default_service_min")]
    pub service_min: f64,
    #[serde(default = "default_weight_kg")]
    pub weight_kg: f64,
}

fn default_service_min() -> f64 {
    10.0
}
fn default_weight_kg() -> f64 {
    1.0
}

/// Один заказ или массив заказов — принимаем оба варианта.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OneOrManyOrders {
    One(OrderInput),
    Many(Vec<OrderInput>),
}

/// Географическая точка (ширина/долгота) для запросов.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LatLng {
    pub lat: f64,
    pub lon: f64,
}

impl LatLng {
    pub fn as_tuple(&self) -> (f64, f64) {
        (self.lat, self.lon)
    }
}

/// Валидация координат: lat ∈ [-90, 90], lon ∈ [-180, 180].
pub fn validate_latlng(lat: f64, lon: f64) -> Result<(), String> {
    if !(-90.0..=90.0).contains(&lat) {
        return Err(format!("Широта {lat} вне диапазона [-90; 90]"));
    }
    if !(-180.0..=180.0).contains(&lon) {
        return Err(format!("Долгота {lon} вне диапазона [-180; 180]"));
    }
    Ok(())
}

/// Запрос планирования доставки (кластеризация + VRPTW + улучшение).
#[derive(Debug, Deserialize)]
pub struct PlanRequest {
    /// Количество курьеров. Если не задано — эвристика: 1 курьер на ~6 заказов.
    pub couriers: Option<u32>,
    /// Склад/точка старта. По умолчанию — Красная площадь, Москва.
    pub start: Option<LatLng>,
    /// Время выезда "ЧЧ:ММ". По умолчанию "09:00".
    #[serde(default = "default_start_clock")]
    pub start_clock: String,
    /// Замкнуть маршруты возвратом на склад (постановка TSP).
    #[serde(default)]
    pub return_to_start: bool,
    /// Использовать OSRM для матрицы времён/расстояний (при недоступности — фолбэк Haversine+ML).
    #[serde(default)]
    pub use_osrm: bool,
}

fn default_start_clock() -> String {
    "09:00".into()
}

/// Останов в маршруте курьера.
#[derive(Debug, Clone, Serialize)]
pub struct Stop {
    pub seq: usize,
    pub order_id: u64,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    /// Расчётное время прибытия от начала рейса, мин.
    pub eta_min: f64,
    /// Часы прибытия с учётом ожидания окна ("ЧЧ:ММ").
    pub arrival_clock: String,
    /// Ожидание открытия окна, мин.
    pub wait_min: f64,
    pub window: TimeWindow,
    /// Плечо до этой остановки.
    pub leg_distance_km: f64,
    pub leg_duration_min: f64,
}

/// Плечо возврата на склад (если маршрут замкнут).
#[derive(Debug, Clone, Serialize)]
pub struct ReturnLeg {
    pub distance_km: f64,
    pub duration_min: f64,
    pub arrival_clock: String,
}

/// Маршрут одного курьера (кластера заказов).
#[derive(Debug, Clone, Serialize)]
pub struct CourierRoute {
    pub courier_id: u64,
    pub stops: Vec<Stop>,
    pub distance_km: f64,
    pub duration_min: f64,
    pub load_kg: f64,
    pub return_to_start: bool,
    pub return_leg: Option<ReturnLeg>,
    pub all_on_time: bool,
}

/// Итоговый план доставки.
#[derive(Debug, Clone, Serialize)]
pub struct RoutePlan {
    pub plan_id: u64,
    pub created_at: String,
    /// Стратегия улучшения, выбранная UCB1-бандитом.
    pub strategy: String,
    /// Источник матрицы расстояний/времён: "osrm" или "haversine+ml".
    pub distance_source: String,
    pub start_clock: String,
    pub return_to_start: bool,
    pub couriers: Vec<CourierRoute>,
    pub total_distance_km: f64,
    pub total_duration_min: f64,
    pub orders_total: usize,
    pub orders_assigned: usize,
    pub unassigned: Vec<UnassignedOrder>,
    /// След работы алгоритмов (русские пояснения по шагам).
    pub algorithm_trace: Vec<String>,
}

/// Заказ, который не удалось разместить в маршруте (окно недостижимо).
#[derive(Debug, Clone, Serialize)]
pub struct UnassignedOrder {
    pub order_id: u64,
    pub reason: String,
}

/// Ответ TSP-решателя.
#[derive(Debug, Serialize)]
pub struct TspResponse {
    /// Тур по точкам в порядке посещения (без старта в ответе).
    pub tour: Vec<LatLng>,
    pub length_km: f64,
    /// Длина жадного тура «ближайший сосед» до улучшения.
    pub baseline_nn_km: f64,
    pub improvement_pct: f64,
    pub method: String,
    pub iterations: usize,
    pub closed: bool,
    pub elapsed_ms: u128,
}

/// Элемент ответа «ближайший заказ».
#[derive(Debug, Serialize)]
pub struct NearestItem {
    pub order_id: u64,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub distance_km: f64,
    /// Прогноз ML времени в пути до заказа, мин.
    pub eta_min: f64,
    pub window: TimeWindow,
}

#[derive(Debug, Serialize)]
pub struct NearestResponse {
    pub from: LatLng,
    pub clock: String,
    pub items: Vec<NearestItem>,
}

/// Запрос Дейкстры по сетке дорог.
#[derive(Debug, Deserialize)]
pub struct DijkstraRequest {
    pub from: LatLng,
    pub to: LatLng,
    /// Шаг сетки в км (по умолчанию 1.0).
    #[serde(default = "default_step_km")]
    pub step_km: f64,
}

fn default_step_km() -> f64 {
    1.0
}

#[derive(Debug, Serialize)]
pub struct DijkstraResponse {
    pub distance_km: f64,
    pub path: Vec<LatLng>,
    pub grid_nodes: usize,
    pub grid_edges_checked: usize,
    pub step_km: f64,
    /// Напрямую по гаверсинусу — для сравнения с сетевым путём.
    pub straight_line_km: f64,
}

/// Обратная связь для обучения: фактическое время в пути по плечу.
#[derive(Debug, Deserialize)]
pub struct LegFeedback {
    pub from_lat: f64,
    pub from_lon: f64,
    pub to_lat: f64,
    pub to_lon: f64,
    /// Фактическое время в пути, мин.
    pub actual_min: f64,
    /// Время выезда "ЧЧ:ММ" (для признака «час пик»). Если не задано — 12:00.
    #[serde(default = "default_clock_min")]
    pub clock_min: u32,
}

fn default_clock_min() -> u32 {
    720
}

#[derive(Debug, Deserialize)]
pub struct FeedbackRequest {
    pub legs: Vec<LegFeedback>,
}

/// Запрос демо-набора Москвы.
#[derive(Debug, Default, Deserialize)]
pub struct DemoRequest {
    /// Сгенерировать синтетическую историю поездок и обучить на ней SGD-модель.
    #[serde(default)]
    pub with_feedback: bool,
}
