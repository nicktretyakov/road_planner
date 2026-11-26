//! Жадная стратегия «ближайший сосед»: на каждом шаге выбирается ближайший
//! ещё не посещённый заказ относительно текущего местоположения.
//!
//! Это базовая эвристика сервера: курьер в произвольной точке получает
//! подсказку «какой заказ ехать везти следующим» (эндпоинт /api/nearest),
//! а также стартовое решение для TSP перед локальным улучшением.

use crate::geo_core::haversine_km;

/// Индекс ближайшей неиспользованной точки к `from`.
pub fn nearest_index(from: (f64, f64), pts: &[(f64, f64)], used: &[bool]) -> Option<usize> {
    let mut best = None;
    let mut best_d = f64::INFINITY;
    for (i, p) in pts.iter().enumerate() {
        if i >= used.len() || used[i] {
            continue;
        }
        let d = haversine_km(from, *p);
        if d < best_d {
            best_d = d;
            best = Some(i);
        }
    }
    best
}

/// Полный жадный маршрут: старт → последовательность индексов точек.
pub fn nearest_route(start: (f64, f64), pts: &[(f64, f64)]) -> Vec<usize> {
    let mut used = vec![false; pts.len()];
    let mut cur = start;
    let mut seq = Vec::with_capacity(pts.len());
    for _ in 0..pts.len() {
        if let Some(i) = nearest_index(cur, pts, &used) {
            used[i] = true;
            seq.push(i);
            cur = pts[i];
        }
    }
    seq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_closest_first() {
        let start = (55.7539, 37.6208);
        // точки: север (~8 км), юг (~4 км), восток (~9 км)
        let pts = [(55.8260, 37.6400), (55.7180, 37.6000), (55.7540, 37.7100)];
        let seq = nearest_route(start, &pts);
        assert_eq!(seq[0], 1, "первым должен идти самый близкий южный заказ");
    }

    #[test]
    fn empty_pts() {
        assert!(nearest_route((55.0, 37.0), &[]).is_empty());
        assert!(nearest_index((55.0, 37.0), &[], &[]).is_none());
    }

    #[test]
    fn used_points_skipped() {
        let pts = [(55.0, 37.0), (55.1, 37.0)];
        let used = [true, false];
        assert_eq!(nearest_index((55.0, 37.0), &pts, &used), Some(1));
    }
}
