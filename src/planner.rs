//! Оркестратор планирования: K-means кластеризация → матрицы расстояний/времён
//! (OSRM или Haversine+ML) → жадная вставка VRPTW → улучшение стратегией,
//! выбранной UCB1-бандитом → расписание ETA. Уникальные бизнес-элементы:
//! учет срочных окон, ML-ETA и бандит-самонастройка.

use std::sync::atomic::Ordering;

use log::info;

use crate::algorithms::kmeans::{default_cluster_count, kmeans};
use crate::algorithms::tsp::{ImproveMode, DistFn};
use crate::algorithms::vrptw::{greedy_insertion, improve_with_windows, route_len_open, simulate, StopNode};
use crate::geo_core::{fmt_hh_mm, haversine_km, parse_hh_mm, to_local_km};
use crate::ml::ClusterInfo;
use crate::models::*;
use crate::storage::AppState;

/// Склад по умолчанию: Москва, Красная площадь.
pub const DEFAULT_DEPOT: (f64, f64) = (55.7539, 37.6208);

/// Построение полного плана доставки по запросу.
pub fn build_plan(state: &AppState, req: &PlanRequest) -> Result<RoutePlan, String> {
    let orders: Vec<Order> = {
        let o = state.orders.read().unwrap();
        o.clone()
    };
    if orders.is_empty() {
        return Err("Нет заказов для планирования: добавьте заказы (POST /api/orders) или загрузите демо (POST /api/demo/moscow)".into());
    }

    let depot = req.start.map(|p| p.as_tuple()).unwrap_or(DEFAULT_DEPOT);
    let start_min = parse_hh_mm(&req.start_clock)? as f64;
    let n = orders.len();

    // ---- 1. Выбор числа курьеров и кластеризация K-means ----
    let k = req
        .couriers
        .map(|c| c as usize)
        .unwrap_or_else(|| default_cluster_count(n))
        .clamp(1, n);

    let origin = centroid_of(&orders, depot);
    let pts_km: Vec<(f64, f64)> = orders
        .iter()
        .map(|o| to_local_km(origin, o.latlng()))
        .collect();
    let clustering = kmeans(&pts_km, k, 42, 100);
    let mut trace: Vec<String> = vec![format!(
        "K-means: {} заказ(ов) → {} кластер(ов) за {} итераций (инерция {:.2} км²)",
        n, clustering.k, clustering.iterations, clustering.inertia
    )];

    // ---- 2. Матрицы и маршруты по кластерам ----
    let mut courier_routes: Vec<CourierRoute> = Vec::new();
    let mut unassigned: Vec<UnassignedOrder> = Vec::new();
    let mut cluster_sizes = vec![0usize; clustering.k];
    for &l in &clustering.labels {
        cluster_sizes[l] += 1;
    }

    let mut total_km = 0.0;
    let mut total_min = 0.0;
    let mut assigned_total = 0usize;
    let mut distance_source = "haversine+ml".to_string();

    // Бандит выбирает стратегию один раз на весь план (награда — суммарная по всем кластерам)
    let (arm_index, arm_name) = {
        let ml = state.ml.lock().unwrap();
        let arm = ml.bandit.select();
        (arm, ml.bandit.arm_name(arm).to_string())
    };
    let mode = mode_from_name(&arm_name);
    trace.push(format!(
        "UCB1-бандит выбрал стратегию улучшения '{}' ({}): попыток {}",
        arm_name,
        mode.description(),
        {
            let ml = state.ml.lock().unwrap();
            ml.bandit.stats()[arm_index].pulls
        }
    ));

    let mut sum_insertion_km = 0.0;
    let mut sum_improved_km = 0.0;

    for cluster in 0..clustering.k {
        let members: Vec<usize> = (0..n).filter(|&i| clustering.labels[i] == cluster).collect();
        if members.is_empty() {
            continue;
        }
        // узлы: 0 — депо, 1..=m — заказы кластера
        let m = members.len();
        let mut stops: Vec<StopNode> = Vec::with_capacity(m);
        for &oi in &members {
            let o = &orders[oi];
            stops.push(StopNode {
                order_id: o.id,
                order_vec_idx: oi,
                lat: o.lat,
                lon: o.lon,
                window: o.window,
                service_min: o.service_min,
                weight_kg: o.weight_kg,
            });
        }

        // координаты для матриц: [депо, заказы кластера]
        let pts: Vec<(f64, f64)> =
            std::iter::once(depot).chain(stops.iter().map(|s| (s.lat, s.lon))).collect();

        // ---- 2a. матрица расстояний/времён: OSRM или Haversine+ML ----
        let (km_mat, min_mat, source) = build_matrices(state, &pts, &req.start_clock, req.use_osrm, &mut trace);
        if cluster == 0 {
            distance_source = source;
        }

        let tt_closure = |a: usize, b: usize| min_mat[a][b];
        let km_closure = |a: usize, b: usize| km_mat[a][b];
        let tt: DistFn = &tt_closure;
        let dkm: DistFn = &km_closure;

        // ---- 2b. жадная вставка с временными окнами ----
        let cand: Vec<usize> = (1..=m).collect();
        let (mut seq, unassigned_nodes) =
            greedy_insertion(start_min, &stops, &cand, tt, dkm, 0.5, 50);
        let insertion_km = route_len_open(&seq, dkm);
        sum_insertion_km += insertion_km;

        for node in unassigned_nodes {
            let s = &stops[node - 1];
            unassigned.push(UnassignedOrder {
                order_id: s.order_id,
                reason: "не удалось уложить во временное окно с текущим временем выезда".into(),
            });
        }

        // ---- 2c. улучшение выбранной бандитом стратегией (окна сохраняются) ----
        let seed = 1000 * state.next_plan_id.load(Ordering::Relaxed) as u64 + cluster as u64;
        let improved_km = improve_with_windows(start_min, &stops, &mut seq, tt, dkm, mode, seed);
        sum_improved_km += improved_km;

        // ---- 2d. финальное расписание ----
        let sched = simulate(start_min, &stops, &seq, tt, dkm);
        let mut stops_out: Vec<Stop> = Vec::with_capacity(seq.len());
        let mut load_kg = 0.0;
        for (i, &node) in seq.iter().enumerate() {
            let s = &stops[node - 1];
            let leg = &sched.legs[i];
            let o = &orders[s.order_vec_idx];
            load_kg += s.weight_kg;
            stops_out.push(Stop {
                seq: i + 1,
                order_id: s.order_id,
                name: o.name.clone(),
                lat: s.lat,
                lon: s.lon,
                eta_min: leg.arrival_min,
                arrival_clock: fmt_hh_mm(leg.arrival_min as u32),
                wait_min: leg.wait_min,
                window: s.window,
                leg_distance_km: leg.travel_km,
                leg_duration_min: leg.travel_min,
            });
        }

        // возврат на склад (постановка TSP — замкнутый тур)
        let mut return_leg = None;
        let mut route_km = sched.total_km;
        let mut route_min = sched.total_min;
        if req.return_to_start {
            if let Some(&last) = seq.last() {
                let back_km = km_mat[last][0];
                let back_min = min_mat[last][0];
                route_km += back_km;
                route_min += back_min;
                return_leg = Some(ReturnLeg {
                    distance_km: back_km,
                    duration_min: back_min,
                    arrival_clock: fmt_hh_mm((start_min + route_min) as u32),
                });
            }
        }
        total_km += route_km;
        total_min += route_min;
        assigned_total += stops_out.len();

        courier_routes.push(CourierRoute {
            courier_id: cluster as u64 + 1,
            stops: stops_out,
            distance_km: route_km,
            duration_min: route_min,
            load_kg,
            return_to_start: req.return_to_start,
            return_leg,
            all_on_time: sched.feasible,
        });
        trace.push(format!(
            "Курьер {} (кластер из {} заказов): {:.2} км, {:.1} мин, нарушение окон: {}",
            cluster + 1,
            m,
            route_km,
            route_min,
            if sched.feasible { "нет" } else { "есть" }
        ));
    }

    // ---- 3. Награда бандита: % сокращения маршрута относительно жадной вставки ----
    let improvement_pct = if sum_insertion_km > 1e-9 {
        ((sum_insertion_km - sum_improved_km) / sum_insertion_km * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };
    {
        let mut ml = state.ml.lock().unwrap();
        ml.bandit.update(arm_index, improvement_pct);
        ml.plan_count += 1;
        ml.last_clusters = Some(ClusterInfo {
            k: clustering.k,
            sizes: cluster_sizes,
            inertia: clustering.inertia,
        });
    }
    trace.push(format!(
        "Награда бандита: {improvement_pct:.2}% сокращения маршрута ({:.2} км → {:.2} км)",
        sum_insertion_km, sum_improved_km
    ));

    let plan = RoutePlan {
        plan_id: state.next_plan_id.fetch_add(1, Ordering::Relaxed),
        created_at: chrono_utc_now(),
        strategy: arm_name,
        distance_source,
        start_clock: req.start_clock.clone(),
        return_to_start: req.return_to_start,
        couriers: courier_routes,
        total_distance_km: total_km,
        total_duration_min: total_min,
        orders_total: n,
        orders_assigned: assigned_total,
        unassigned,
        algorithm_trace: trace,
    };
    info!(
        "План #{}: {} заказов, {:.1} км, {:.0} мин, стратегия {}",
        plan.plan_id, plan.orders_total, plan.total_distance_km, plan.total_duration_min, plan.strategy
    );
    Ok(plan)
}

/// Имя стратегии бандита → режим улучшения.
fn mode_from_name(name: &str) -> ImproveMode {
    match name {
        "nn_oropt" => ImproveMode::OrOpt,
        "nn_2opt_oropt" => ImproveMode::TwoOptOrOpt,
        "nn_sa" => ImproveMode::Annealing,
        _ => ImproveMode::TwoOpt,
    }
}

/// Матрицы расстояний и времён для набора точек (lat, lon).
/// OSRM (если включён и доступен) → иначе Haversine + ML-прогноз.
fn build_matrices(
    state: &AppState,
    pts: &[(f64, f64)],
    start_clock: &str,
    use_osrm: bool,
    trace: &mut Vec<String>,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, String) {
    let n = pts.len();
    let clock_min = parse_hh_mm(start_clock).unwrap_or(540);

    if use_osrm {
        if let Ok(table) = state.osrm.table(pts) {
            // поверх времён OSRM применяем learned traffic factor
            let feedback = { let ml = state.ml.lock().unwrap(); ml.feedback_count };
            let minutes: Vec<Vec<f64>> = if feedback > 0 {
                let ml = state.ml.lock().unwrap();
                table
                    .minutes
                    .iter()
                    .enumerate()
                    .map(|(i, row)| {
                        row.iter()
                            .enumerate()
                            .map(|(j, &mn)| {
                                mn * ml.sgd.traffic_factor(table.km[i][j], clock_min)
                            })
                            .collect()
                    })
                    .collect()
            } else {
                table.minutes.clone()
            };
            trace.push("Матрица времён/расстояний получена от OSRM (OpenStreetMap)".into());
            if feedback == 0 {
                trace.push("Пока нет фактических поездок — traffic-фактор ML не применяется".into());
            }
            return (table.km, minutes, "osrm".into());
        }
    }

    // фолбэк: Haversine (crate geo) + прогноз SGD
    let mut km = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                km[i][j] = haversine_km(pts[i], pts[j]);
            }
        }
    }
    let minutes = {
        let ml = state.ml.lock().unwrap();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { 0.0 } else { ml.sgd.predict(km[i][j], clock_min) })
                    .collect()
            })
            .collect()
    };
    if use_osrm {
        trace.push("OSRM недоступен — фолбэк: Haversine (geo) + ML-прогноз времени".into());
    } else {
        trace.push("Матрица: Haversine (geo) + ML-прогноз времени в пути".into());
    }
    (km, minutes, "haversine+ml".into())
}

/// Центроид заказов (или депо, если заказов нет) для локальной проекции.
fn centroid_of(orders: &[Order], fallback: (f64, f64)) -> (f64, f64) {
    if orders.is_empty() {
        return fallback;
    }
    let (sl, so) = orders.iter().fold((0.0, 0.0), |acc, o| (acc.0 + o.lat, acc.1 + o.lon));
    (sl / orders.len() as f64, so / orders.len() as f64)
}

/// Текущее UTC-время в RFC-подобной строке (без внешних крейтов).
fn chrono_utc_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86_400;
    let (y, mo, d) = civil_from_days(days as i64);
    let hm = secs % 86_400;
    format!(
        "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}Z",
        hm / 3600,
        (hm % 3600) / 60,
        hm % 60
    )
}

/// Алгоритм Говарда Антессона: дни с эпохи → (год, месяц, день).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_date_epoch() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1)); // 2024-01-01
    }

    #[test]
    fn centroid_fallback() {
        let c = centroid_of(&[], (55.0, 37.0));
        assert_eq!(c, (55.0, 37.0));
    }
}
