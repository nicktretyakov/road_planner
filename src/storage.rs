//! Хранилище в памяти: заказы, планы и ML-состояние (по выбору пользователя
//! данные не persisted'ятся — живут в процессе сервера).

use std::sync::atomic::AtomicU64;
use std::sync::{Mutex, RwLock};

use crate::ml::MlEngine;
use crate::models::{Order, RoutePlan};
use crate::osrm::OsrmClient;

/// Общее состояние приложения (разделено между воркерами actix-web).
pub struct AppState {
    pub orders: RwLock<Vec<Order>>,
    pub plans: RwLock<Vec<RoutePlan>>,
    pub next_order_id: AtomicU64,
    pub next_plan_id: AtomicU64,
    /// ML-движок под мьютексом: SGD-модель + UCB1-бандит.
    pub ml: Mutex<MlEngine>,
    pub osrm: OsrmClient,
}

impl AppState {
    /// Конфигурация из окружения (OSRM_URL / OSRM_ENABLED).
    pub fn new() -> Self {
        Self {
            orders: RwLock::new(Vec::new()),
            plans: RwLock::new(Vec::new()),
            next_order_id: AtomicU64::new(1),
            next_plan_id: AtomicU64::new(1),
            ml: Mutex::new(MlEngine::new()),
            osrm: OsrmClient::from_env(),
        }
    }

    /// Офлайн-режим: OSRM принудительно выключен (тесты и детерминизм).
    pub fn new_offline() -> Self {
        Self {
            osrm: OsrmClient::disabled(),
            ..Self::new()
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
