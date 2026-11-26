//! Задача коммивояжёра (TSP): посетить все точки ровно один раз и (опционально)
//! вернуться в исходную. Собственная реализация:
//!
//! 1. Построение — жадный «ближайший сосед»;
//! 2. Улучшение — 2-opt, Or-opt (перенос отрезков длины 1–3) и
//!    имитация отжига (метод, имитирующий природный процесс управляемого
//!    охлаждения металла: шаги принимаются вероятностно с убывающей температурой);
//! 3. Всё расстояние — Haversine из библиотеки `geo`.
//!
//! Узлы: 0 — старт/депо, 1..n — заказы. Функция расстояния передаётся снаружи,
//! поэтому тот же код работает и для матриц OSRM.

use rand::prelude::*;

/// Функция расстояния между узлами (0 — депо).
pub type DistFn<'a> = &'a dyn Fn(usize, usize) -> f64;

/// Режим локального улучшения (именуется также для UCB1-бандита).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImproveMode {
    /// 2-opt: разворот отрезков маршрута.
    TwoOpt,
    /// Or-opt: перенос отрезков длины 1–3 в другие позиции.
    OrOpt,
    /// 2-opt + Or-opt последовательно.
    TwoOptOrOpt,
    /// Имитация отжига + полировка 2-opt.
    Annealing,
}

impl ImproveMode {
    pub fn name(&self) -> &'static str {
        match self {
            ImproveMode::TwoOpt => "nn_2opt",
            ImproveMode::OrOpt => "nn_oropt",
            ImproveMode::TwoOptOrOpt => "nn_2opt_oropt",
            ImproveMode::Annealing => "nn_sa",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            ImproveMode::TwoOpt => "2-opt: развороты отрезков до сходимости",
            ImproveMode::OrOpt => "Or-opt: перенос отрезков длины 1-3",
            ImproveMode::TwoOptOrOpt => "2-opt + Or-opt комбинированно",
            ImproveMode::Annealing => "имитация отжига + полировка 2-opt",
        }
    }
}

/// Длина маршрута: депо → seq → (депо, если замкнут).
pub fn tour_length(seq: &[usize], d: DistFn, closed: bool) -> f64 {
    if seq.is_empty() {
        return 0.0;
    }
    let mut len = d(0, seq[0]);
    for w in seq.windows(2) {
        len += d(w[0], w[1]);
    }
    if closed {
        len += d(*seq.last().unwrap(), 0);
    }
    len
}

/// Построение жадным «ближайшим соседом» по узлам 1..=n от депо (узел 0).
pub fn nearest_construction(d: DistFn, n: usize) -> Vec<usize> {
    let mut used = vec![false; n + 1];
    let mut seq = Vec::with_capacity(n);
    let mut cur = 0usize;
    for _ in 0..n {
        let mut best = 0usize;
        let mut bd = f64::INFINITY;
        for v in 1..=n {
            if !used[v] {
                let dd = d(cur, v);
                if dd < bd {
                    bd = dd;
                    best = v;
                }
            }
        }
        used[best] = true;
        seq.push(best);
        cur = best;
    }
    seq
}

/// Проходы 2-opt до локальной сходимости. Если задана проверка `feasible`,
/// разворот принимается только когда маршрут остаётся допустимым (VRPTW).
pub fn two_opt_passes(
    seq: &mut Vec<usize>,
    d: DistFn,
    closed: bool,
    feasible: Option<&dyn Fn(&[usize]) -> bool>,
    max_passes: usize,
) -> usize {
    let mut passes = 0;
    loop {
        passes += 1;
        let mut improved = false;
        let n = seq.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let prev = if i == 0 { 0 } else { seq[i - 1] };
                let a = seq[i];
                let b = seq[j];
                let next = if j + 1 < n { Some(seq[j + 1]) } else if closed { Some(0) } else { None };
                let delta = d(prev, b) + next.map(|nx| d(a, nx)).unwrap_or(0.0)
                    - d(prev, a) - next.map(|nx| d(b, nx)).unwrap_or(0.0);
                if delta < -1e-9 {
                    let mut cand = seq.clone();
                    cand[i..=j].reverse();
                    if feasible.map(|f| f(&cand)).unwrap_or(true) {
                        *seq = cand;
                        improved = true;
                    }
                }
            }
        }
        if !improved || passes >= max_passes {
            break;
        }
    }
    passes
}

/// Проходы Or-opt: перенос отрезков длины 1–3 в лучшие позиции.
pub fn or_opt_passes(
    seq: &mut Vec<usize>,
    d: DistFn,
    closed: bool,
    feasible: Option<&dyn Fn(&[usize]) -> bool>,
    max_passes: usize,
) -> usize {
    let mut passes = 0;
    loop {
        passes += 1;
        let mut improved = false;
        let base_len = tour_length(seq, d, closed);
        let n = seq.len();
        'outer: for seg_len in 1..=3.min(n) {
            for i in 0..=(n - seg_len) {
                let seg: Vec<usize> = seq[i..i + seg_len].to_vec();
                let mut rest = Vec::with_capacity(n - seg_len);
                rest.extend_from_slice(&seq[..i]);
                rest.extend_from_slice(&seq[i + seg_len..]);
                for pos in 0..=rest.len() {
                    if pos == i {
                        continue;
                    }
                    let mut cand = Vec::with_capacity(n);
                    cand.extend_from_slice(&rest[..pos]);
                    cand.extend_from_slice(&seg);
                    cand.extend_from_slice(&rest[pos..]);
                    let len = tour_length(&cand, d, closed);
                    if len < base_len - 1e-9 {
                        if feasible.map(|f| f(&cand)).unwrap_or(true) {
                            *seq = cand;
                            improved = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
        if !improved || passes >= max_passes {
            break;
        }
    }
    passes
}

/// Имитация отжига: случайные шаги 2-opt/Or-opt принимаются вероятностно
/// с геометрически остывающей температурой; недопустимые (по окнам) шаги отклоняются.
pub fn annealing(
    seq: &mut Vec<usize>,
    d: DistFn,
    closed: bool,
    seed: u64,
    iters: usize,
    feasible: Option<&dyn Fn(&[usize]) -> bool>,
) -> f64 {
    let n = seq.len();
    if n < 2 {
        return tour_length(seq, d, closed);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cur = seq.clone();
    let mut cur_len = tour_length(&cur, d, closed);
    let mut best = cur.clone();
    let mut best_len = cur_len;
    let t0 = (cur_len / n as f64).max(0.5);
    let t1 = t0 * 1e-4;
    let alpha = (t1 / t0).powf(1.0 / iters.max(1) as f64);
    let mut t = t0;

    for _ in 0..iters {
        let mut cand = cur.clone();
        if rng.gen::<f64>() < 0.5 {
            // случайный 2-opt
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            let (i, j) = (i.min(j), i.max(j));
            if i == j {
                t *= alpha;
                continue;
            }
            cand[i..=j].reverse();
        } else {
            // случайный Or-opt
            let seg_len = rng.gen_range(1..=3.min(n));
            let i = rng.gen_range(0..=(n - seg_len));
            let seg: Vec<usize> = cand[i..i + seg_len].to_vec();
            cand.drain(i..i + seg_len);
            let pos = rng.gen_range(0..=cand.len());
            for (k, s) in seg.into_iter().enumerate() {
                cand.insert(pos + k, s);
            }
        }
        if let Some(f) = feasible {
            if !f(&cand) {
                t *= alpha;
                continue;
            }
        }
        let cand_len = tour_length(&cand, d, closed);
        let delta = cand_len - cur_len;
        if delta < 0.0 || rng.gen::<f64>() < (-delta / t).exp() {
            cur = cand;
            cur_len = cand_len;
            if cur_len < best_len - 1e-9 {
                best = cur.clone();
                best_len = cur_len;
            }
        }
        t *= alpha;
    }
    *seq = best;
    best_len
}

/// Общая точка входа улучшения: применяет выбранный режим, гарантируя
/// допустимость (если `feasible` задан — недопустимый результат откатывается).
pub fn improve(
    seq: &mut Vec<usize>,
    d: DistFn,
    closed: bool,
    mode: ImproveMode,
    seed: u64,
    feasible: Option<&dyn Fn(&[usize]) -> bool>,
) -> usize {
    let mut work = seq.clone();
    if let Some(f) = feasible {
        if !f(&work) {
            return 0; // вход недопустим — улучшение не выполняем
        }
    }
    let mut iterations = 0;
    match mode {
        ImproveMode::TwoOpt => {
            iterations = two_opt_passes(&mut work, d, closed, feasible, 60);
        }
        ImproveMode::OrOpt => {
            iterations = or_opt_passes(&mut work, d, closed, feasible, 60);
        }
        ImproveMode::TwoOptOrOpt => {
            iterations += two_opt_passes(&mut work, d, closed, feasible, 60);
            iterations += or_opt_passes(&mut work, d, closed, feasible, 60);
            iterations += two_opt_passes(&mut work, d, closed, feasible, 60);
        }
        ImproveMode::Annealing => {
            let iters = (work.len() * 3000).clamp(2000, 60_000);
            annealing(&mut work, d, closed, seed, iters, feasible);
            iterations = iters + two_opt_passes(&mut work, d, closed, feasible, 60);
        }
    }
    if let Some(f) = feasible {
        if !f(&work) {
            return iterations; // страховка: недопустимый результат не применяем
        }
    }
    *seq = work;
    iterations
}

/// Решение TSP для географических точек: узел 0 — первая точка (старт).
pub struct TspSolve {
    pub seq: Vec<usize>,
    pub length_km: f64,
    pub baseline_km: f64,
    pub iterations: usize,
    pub mode: ImproveMode,
}

pub fn solve_tsp(pts: &[(f64, f64)], closed: bool, mode: ImproveMode, seed: u64) -> TspSolve {
    let n = pts.len().saturating_sub(1); // pts[0] — старт
    let d = |a: usize, b: usize| -> f64 {
        let pa = if a == 0 { pts[0] } else { pts[a] };
        let pb = if b == 0 { pts[0] } else { pts[b] };
        crate::geo_core::haversine_km(pa, pb)
    };
    let mut seq = nearest_construction(&d, n);
    let baseline = tour_length(&seq, &d, closed);
    let iterations = improve(&mut seq, &d, closed, mode, seed, None);
    let length = tour_length(&seq, &d, closed);
    TspSolve { seq, length_km: length, baseline_km: baseline, iterations, mode }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo_core::haversine_km;

    /// Синтетические точки в км: равносторонняя решётка; расстояние — евклид.
    fn euclid_pts() -> Vec<(f64, f64)> {
        vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (20.0, 0.0),
            (10.0, 8.0),
            (20.0, 8.0),
            (0.0, 8.0),
        ]
    }

    fn euclid_d(a: usize, b: usize) -> f64 {
        let pts = euclid_pts();
        let pa = pts[a];
        let pb = pts[b];
        ((pa.0 - pb.0).powi(2) + (pa.1 - pb.1).powi(2)).sqrt()
    }

    #[test]
    fn two_opt_improves_crossed_tour() {
        // заведомо «крестовой» порядок: 1 -> 5 -> 3 -> 4 -> 2 (узлы 0..5)
        let mut seq = vec![1, 5, 3, 4, 2];
        let before = tour_length(&seq, &euclid_d, true);
        two_opt_passes(&mut seq, &euclid_d, true, None, 50);
        let after = tour_length(&seq, &euclid_d, true);
        assert!(after < before - 1e-6, "2-opt не улучшил тур: {before} -> {after}");
    }

    #[test]
    fn nn_construction_visits_all_once() {
        let d = |a: usize, b: usize| euclid_d(a, b);
        let seq = nearest_construction(&d, 5);
        assert_eq!(seq.len(), 5);
        let mut seen = vec![false; 6];
        for v in &seq {
            assert_ne!(*v, 0);
            assert!(!seen[*v]);
            seen[*v] = true;
        }
    }

    #[test]
    fn annealing_not_worse_than_baseline() {
        let d = |a: usize, b: usize| euclid_d(a, b);
        let mut seq = nearest_construction(&d, 5);
        let base = tour_length(&seq, &d, true);
        annealing(&mut seq, &d, true, 123, 5000, None);
        let after = tour_length(&seq, &d, true);
        assert!(after <= base + 1e-9, "отжиг ухудшил тур: {base} -> {after}");
    }

    #[test]
    fn solve_tsp_closed_roundtrip() {
        // Москва-центр → 5 точек в радиусе ~5 км
        let pts = vec![
            (55.7539, 37.6208),
            (55.7615, 37.6094),
            (55.7485, 37.5920),
            (55.7401, 37.5306),
            (55.7940, 37.5770),
            (55.7570, 37.6530),
        ];
        let r = solve_tsp(&pts, true, ImproveMode::TwoOptOrOpt, 11);
        assert_eq!(r.seq.len(), 5);
        let mut seen = vec![false; 6];
        for v in &r.seq {
            assert!(!seen[*v]);
            seen[*v] = true;
        }
        let d = |a: usize, b: usize| -> f64 {
            haversine_km(pts[if a == 0 { 0 } else { a }], pts[if b == 0 { 0 } else { b }])
        };
        let recomputed = tour_length(&r.seq, &d, true);
        assert!((recomputed - r.length_km).abs() < 1e-9);
        assert!(r.length_km <= r.baseline_km + 1e-9);
    }

    #[test]
    fn open_tour_no_return_leg() {
        let d = |a: usize, b: usize| euclid_d(a, b);
        let seq = vec![1, 2, 3];
        let open = tour_length(&seq, &d, false);
        let closed = tour_length(&seq, &d, true);
        assert!(closed > open);
        assert!((closed - open - euclid_d(3, 0)).abs() < 1e-9);
    }
}
