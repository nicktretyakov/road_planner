//! VRPTW — расширение TSP временными окнами: для каждого заказа задан
//! интервал [ready, due], когда возможна доставка. Считается допустимым
//! прибытие раньше открытия окна (курьер ждёт), но опоздание запрещено.
//!
//! Построение маршрута — жадная вставка в духе Solomon I1: кандидаты
//! сортируются по срочности (конец окна), для каждого перебираются позиции
//! вставки, выбирается минимальная стоимость = добавленное расстояние +
//! взвешенное ожидание. Улучшение — локальные 2-opt/Or-opt/отжиг с
//! обязательной проверкой допустимости всех окон.

use serde::Serialize;

use crate::algorithms::tsp::{improve, DistFn, ImproveMode};
use crate::geo_core::TimeWindow;

/// Останов маршрута в узловых координатах (узел i в seq соответствует stops[i-1]).
#[derive(Debug, Clone)]
pub struct StopNode {
    pub order_id: u64,
    /// Индекс в исходном списке заказов (для вывода).
    pub order_vec_idx: usize,
    pub lat: f64,
    pub lon: f64,
    pub window: TimeWindow,
    pub service_min: f64,
    pub weight_kg: f64,
}

/// План одного плеча маршрута.
#[derive(Debug, Clone, Serialize)]
pub struct LegPlan {
    pub order_vec_idx: usize,
    pub arrival_min: f64,
    pub wait_min: f64,
    pub depart_min: f64,
    pub travel_min: f64,
    pub travel_km: f64,
    pub on_time: bool,
}

/// Результат симуляции маршрута: расписание прибытий по всем остановкам.
#[derive(Debug, Clone)]
pub struct RouteSchedule {
    pub legs: Vec<LegPlan>,
    pub feasible: bool,
    pub total_km: f64,
    /// Общая длительность рейса (в пути + ожидание + сервис), мин.
    pub total_min: f64,
    pub pure_travel_min: f64,
    pub wait_min_total: f64,
    pub violations: Vec<String>,
}

/// Прокрутка расписания: узел 0 — депо, seq — узлы остановок.
pub fn simulate(
    start_min: f64,
    stops: &[StopNode],
    seq: &[usize],
    tt: DistFn,
    dkm: DistFn,
) -> RouteSchedule {
    let mut legs = Vec::with_capacity(seq.len());
    let mut violations = Vec::new();
    let mut feasible = true;
    let mut t = start_min;
    let mut prev = 0usize;
    let mut total_km = 0.0;
    let mut travel_total = 0.0;
    let mut wait_total = 0.0;

    for &node in seq {
        let s = &stops[node - 1];
        let travel = tt(prev, node);
        let km = dkm(prev, node);
        let arrival = t + travel;
        let wait = (s.window.start_min as f64 - arrival).max(0.0);
        let svc_start = arrival + wait;
        let on_time = svc_start <= s.window.end_min as f64 + 1e-9;
        if !on_time {
            feasible = false;
            violations.push(format!(
                "заказ {} прибытие в {}, окно до {}",
                s.order_id,
                crate::geo_core::fmt_hh_mm(svc_start as u32),
                s.window
            ));
        }
        legs.push(LegPlan {
            order_vec_idx: s.order_vec_idx,
            arrival_min: arrival,
            wait_min: wait,
            depart_min: svc_start + s.service_min,
            travel_min: travel,
            travel_km: km,
            on_time,
        });
        t = svc_start + s.service_min;
        total_km += km;
        travel_total += travel;
        wait_total += wait;
        prev = node;
    }

    RouteSchedule {
        legs,
        feasible,
        total_km,
        total_min: t - start_min,
        pure_travel_min: travel_total,
        wait_min_total: wait_total,
        violations,
    }
}

/// Длина открытого маршрута по узлам (без возврата в депо).
pub fn route_len_open(seq: &[usize], dkm: DistFn) -> f64 {
    if seq.is_empty() {
        return 0.0;
    }
    let mut len = dkm(0, seq[0]);
    for w in seq.windows(2) {
        len += dkm(w[0], w[1]);
    }
    len
}

/// Жадная вставка Solomon-типа. Кандидаты рассматриваются по срочности
/// (чем раньше закрывается окно, тем раньше вставка). Стоимость вставки:
/// добавленное расстояние + `wait_weight` × добавленное ожидание.
/// Приоритет отдается минимальной стоимости среди всех (кандидат, позиция).
///
/// Возвращает (последовательность узлов, нераспределённые узлы).
pub fn greedy_insertion(
    start_min: f64,
    stops: &[StopNode],
    candidates: &[usize],
    tt: DistFn,
    dkm: DistFn,
    wait_weight: f64,
    max_candidates_per_round: usize,
) -> (Vec<usize>, Vec<usize>) {
    // сортировка по срочности: сначала заказы с ранним концом окна
    let mut remaining: Vec<usize> = candidates.to_vec();
    remaining.sort_by_key(|&n| (stops[n - 1].window.end_min, stops[n - 1].window.start_min));
    let mut route: Vec<usize> = Vec::new();
    let mut unassigned = Vec::new();

    while !remaining.is_empty() {
        let base = simulate(start_min, stops, &route, tt, dkm);
        let considered = if remaining.len() > max_candidates_per_round {
            &remaining[..max_candidates_per_round]
        } else {
            &remaining[..]
        };
        let mut best: Option<(f64, usize, usize)> = None; // (cost, node, pos)
        for &cand in considered {
            for pos in 0..=route.len() {
                let mut seq2 = Vec::with_capacity(route.len() + 1);
                seq2.extend_from_slice(&route[..pos]);
                seq2.push(cand);
                seq2.extend_from_slice(&route[pos..]);
                let sched = simulate(start_min, stops, &seq2, tt, dkm);
                if !sched.feasible {
                    continue;
                }
                let extra_km = sched.total_km - base.total_km;
                let extra_wait = sched.wait_min_total - base.wait_min_total;
                let cost = extra_km + wait_weight * extra_wait;
                if best.map(|b| cost < b.0 - 1e-12).unwrap_or(true) {
                    best = Some((cost, cand, pos));
                }
            }
        }
        match best {
            Some((_, node, pos)) => {
                route.insert(pos, node);
                remaining.retain(|&r| r != node);
            }
            None => {
                // ни один кандидат не встаёт — первый (самый срочный) остаётся нераспределённым
                let cand = remaining.remove(0);
                unassigned.push(cand);
            }
        }
    }
    (route, unassigned)
}

/// Улучшение маршрута с сохранением допустимости всех временных окон.
/// Возвращает итоговую длину открытого маршрута.
pub fn improve_with_windows(
    start_min: f64,
    stops: &[StopNode],
    seq: &mut Vec<usize>,
    tt: DistFn,
    dkm: DistFn,
    mode: ImproveMode,
    seed: u64,
) -> f64 {
    let feasible = |s: &[usize]| simulate(start_min, stops, s, tt, dkm).feasible;
    improve(seq, dkm, false, mode, seed, Some(&feasible));
    route_len_open(seq, dkm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo_core::parse_window;

    fn euclid(a: usize, b: usize) -> f64 {
        let pts: Vec<(f64, f64)> = vec![
            (0.0, 0.0),  // депо
            (2.0, 0.0),  // узел 1
            (5.0, 0.0),  // узел 2
            (9.0, 0.0),  // узел 3
        ];
        let (pa, pb) = (pts[a], pts[b]);
        ((pa.0 - pb.0).powi(2) + (pa.1 - pb.1).powi(2)).sqrt()
    }

    fn stops_3() -> Vec<StopNode> {
        vec![
            StopNode {
                order_id: 1,
                order_vec_idx: 0,
                lat: 55.0,
                lon: 37.0,
                window: parse_window("08:00-18:00").unwrap(),
                service_min: 10.0,
                weight_kg: 1.0,
            },
            StopNode {
                order_id: 2,
                order_vec_idx: 1,
                lat: 55.1,
                lon: 37.1,
                window: parse_window("08:00-18:00").unwrap(),
                service_min: 10.0,
                weight_kg: 1.0,
            },
            StopNode {
                order_id: 3,
                order_vec_idx: 2,
                lat: 55.2,
                lon: 37.2,
                window: parse_window("08:00-18:00").unwrap(),
                service_min: 10.0,
                weight_kg: 1.0,
            },
        ]
    }

    #[test]
    fn simulate_respects_windows() {
        // скорость 1 км/мин: депо→1 = 2 мин, прибытие в 09:02 при старте 09:00
        let stops = stops_3();
        let sched = simulate(540.0, &stops, &[1, 2, 3], &euclid, &euclid);
        assert!(sched.feasible);
        assert!((sched.legs[0].arrival_min - 542.0).abs() < 1e-9);
        assert!((sched.legs[0].travel_min - 2.0).abs() < 1e-9);
        // общий путь 9 км, сервис 3×10 мин
        assert!((sched.total_km - 9.0).abs() < 1e-9);
        assert!((sched.total_min - (9.0 + 30.0)).abs() < 1e-9);
    }

    #[test]
    fn simulate_marks_late_arrival() {
        let stops = stops_3();
        // узел 3 на 9 км; старт в 17:55 → прибытие 18:04, окно до 18:00 → нарушение
        let sched = simulate((17 * 60 + 55) as f64, &stops, &[3], &euclid, &euclid);
        assert!(!sched.feasible);
        assert!(sched.violations.len() == 1);
    }

    #[test]
    fn waiting_for_window_open() {
        let mut stops = stops_3();
        stops[0].window = parse_window("10:00-11:00").unwrap();
        // старт 09:00, до узла 1 — 2 мин → прибытие 09:02, ожидание до 10:00 = 58 мин
        let sched = simulate(540.0, &stops, &[1], &euclid, &euclid);
        assert!(sched.feasible);
        assert!((sched.legs[0].wait_min - 58.0).abs() < 1e-9, "ожидание должно быть 58 мин");
    }

    #[test]
    fn insertion_visits_all_when_feasible() {
        let stops = stops_3();
        let (route, unassigned) =
            greedy_insertion(540.0, &stops, &[1, 2, 3], &euclid, &euclid, 0.5, 50);
        assert_eq!(route.len(), 3);
        assert!(unassigned.is_empty());
        let sched = simulate(540.0, &stops, &route, &euclid, &euclid);
        assert!(sched.feasible);
    }

    #[test]
    fn infeasible_order_goes_unassigned() {
        let mut stops = stops_3();
        // узел 3 — только 09:00-09:05, доехать за 5+ мин невозможно при 1 км/мин (9 км)
        stops[2].window = parse_window("09:00-09:05").unwrap();
        let (route, unassigned) =
            greedy_insertion(540.0, &stops, &[1, 2, 3], &euclid, &euclid, 0.5, 50);
        assert_eq!(route.len(), 2);
        assert_eq!(unassigned, vec![3]);
    }

    #[test]
    fn improvement_keeps_windows() {
        let stops = stops_3();
        let mut seq = vec![3, 2, 1]; // дальний первым — субоптимально
        let before = route_len_open(&seq, &euclid);
        let after = improve_with_windows(540.0, &stops, &mut seq, &euclid, &euclid, ImproveMode::TwoOptOrOpt, 5);
        assert!(after <= before);
        let sched = simulate(540.0, &stops, &seq, &euclid, &euclid);
        assert!(sched.feasible, "улучшение не должно ломать окна");
    }
}
