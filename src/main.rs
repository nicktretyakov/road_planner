//! Точка входа: запуск actix-web сервера планирования маршрутов доставки.
//!
//! Переменные окружения:
//! - `PORT` — порт HTTP (по умолчанию 8080);
//! - `OSRM_URL` — базовый URL OSRM (по умолчанию https://router.project-osrm.org);
//! - `OSRM_ENABLED` — 0/false выключает OSRM (только Haversine + ML);
//! - `RUST_LOG` — уровень логов (по умолчанию info).

use actix_web::{middleware::Logger, web, App, HttpServer};
use route_planner::{api, storage::AppState};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let state = web::Data::new(AppState::new());

    log::info!("Сервер планирования маршрутов: http://0.0.0.0:{port}");
    log::info!("OSRM: {} (enabled={})", state.osrm.base, state.osrm.enabled);
    log::info!("Эндпоинты: /health, /api/orders, /api/plan, /api/tsp, /api/nearest, /api/dijkstra, /api/ml/feedback, /api/ml/stats, /api/demo/moscow, /api/plans");

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .configure(|c| api::configure(c, state.clone()))
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
