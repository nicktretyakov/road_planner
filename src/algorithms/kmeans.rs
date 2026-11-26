//! Кластеризация заказов K-means (с инициализацией k-means++) на локальной
//! плоскости в километрах. Заказы разбиваются на группы по географической
//! близости, затем внутри каждого кластера планируется свой маршрут
//! (по одному курьеру на кластер).
//!
//! Детерминирована при фиксированном seed — поведение воспроизводимо и тестируемо.

use rand::prelude::*;
use serde::Serialize;

/// Результат кластеризации.
#[derive(Debug, Clone, Serialize)]
pub struct KMeans {
    pub k: usize,
    /// Центроиды в локальных километрах (x — восток, y — север).
    pub centers: Vec<(f64, f64)>,
    /// Метка кластера для каждой точки.
    pub labels: Vec<usize>,
    /// Сумма квадратов расстояний до центроидов.
    pub inertia: f64,
    pub iterations: usize,
}

fn dist2(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}

/// K-means с инициализацией k-means++ и Ллойд-итерациями.
/// Точки заданы в километрах (см. `geo_core::to_local_km`).
pub fn kmeans(pts: &[(f64, f64)], k: usize, seed: u64, max_iter: usize) -> KMeans {
    let n = pts.len();
    if n == 0 {
        return KMeans { k: 0, centers: vec![], labels: vec![], inertia: 0.0, iterations: 0 };
    }
    let k = k.clamp(1, n);
    let mut rng = StdRng::seed_from_u64(seed);

    // --- инициализация k-means++ ---
    let mut centers: Vec<(f64, f64)> = Vec::with_capacity(k);
    centers.push(pts[rng.gen_range(0..n)]);
    while centers.len() < k {
        // для каждой точки — квадрат расстояния до ближайшего центра
        let d2: Vec<f64> = pts
            .iter()
            .map(|p| centers.iter().map(|c| dist2(*p, *c)).fold(f64::INFINITY, f64::min))
            .collect();
        let total: f64 = d2.iter().sum();
        let next = if total <= 1e-12 {
            // все точки совпадают — берём первую не дублирующую
            (0..n).find(|i| d2[*i] > 0.0).unwrap_or(0)
        } else {
            let mut r = rng.gen::<f64>() * total;
            let mut idx = n - 1;
            for (i, d) in d2.iter().enumerate() {
                r -= d;
                if r <= 0.0 {
                    idx = i;
                    break;
                }
            }
            idx
        };
        centers.push(pts[next]);
    }

    // --- Ллойд-итерации ---
    let mut labels = vec![0usize; n];
    let mut iterations = 0;
    for _ in 0..max_iter {
        iterations += 1;
        let mut moved = false;
        for (i, p) in pts.iter().enumerate() {
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for (ci, c) in centers.iter().enumerate() {
                let d = dist2(*p, *c);
                if d < bd {
                    bd = d;
                    best = ci;
                }
            }
            if labels[i] != best {
                labels[i] = best;
                moved = true;
            }
        }
        // обновление центроидов
        let mut sums = vec![(0.0_f64, 0.0_f64); k];
        let mut cnt = vec![0usize; k];
        for (i, p) in pts.iter().enumerate() {
            let l = labels[i];
            sums[l].0 += p.0;
            sums[l].1 += p.1;
            cnt[l] += 1;
        }
        for ci in 0..k {
            if cnt[ci] == 0 {
                // пустой кластер: переносим туда самую дальнюю точку от своего центра
                let far = (0..n)
                    .map(|i| (i, dist2(pts[i], centers[labels[i]])))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0;
                labels[far] = ci;
                centers[ci] = pts[far];
            } else {
                centers[ci] = (sums[ci].0 / cnt[ci] as f64, sums[ci].1 / cnt[ci] as f64);
            }
        }
        if !moved {
            break;
        }
    }

    let inertia: f64 = pts
        .iter()
        .zip(&labels)
        .map(|(p, l)| dist2(*p, centers[*l]))
        .sum();

    KMeans { k, centers, labels, inertia, iterations }
}

/// Эвристика числа кластеров: ~6 заказов на курьера, не более 8.
pub fn default_cluster_count(n: usize) -> usize {
    if n <= 5 {
        1
    } else {
        ((n as f64 / 6.0).ceil() as usize).clamp(1, 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_two_blobs() {
        // два облака: вокруг (0,0) и вокруг (100,100) км
        let mut pts = vec![];
        for i in 0..8 {
            pts.push(((i % 3) as f64, (i / 3) as f64));
        }
        for i in 0..8 {
            pts.push((100.0 + (i % 3) as f64, 100.0 + (i / 3) as f64));
        }
        let r = kmeans(&pts, 2, 42, 100);
        // все точки каждого облака должны получить одну метку
        let l0 = r.labels[0];
        for i in 0..8 {
            assert_eq!(r.labels[i], l0);
        }
        for i in 8..16 {
            assert_ne!(r.labels[i], l0);
        }
        assert_eq!(r.k, 2);
    }

    #[test]
    fn k_clamped_to_n() {
        let pts = [(1.0, 1.0), (2.0, 2.0)];
        let r = kmeans(&pts, 10, 1, 50);
        assert_eq!(r.k, 2);
    }

    #[test]
    fn empty_input() {
        let r = kmeans(&[], 3, 1, 10);
        assert_eq!(r.k, 0);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let pts: Vec<(f64, f64)> = (0..20).map(|i| ((i * 7 % 13) as f64, (i * 3 % 11) as f64)).collect();
        let a = kmeans(&pts, 3, 7, 50);
        let b = kmeans(&pts, 3, 7, 50);
        assert_eq!(a.labels, b.labels);
        assert!((a.inertia - b.inertia).abs() < 1e-12);
    }

    #[test]
    fn default_cluster_count_heuristic() {
        assert_eq!(default_cluster_count(3), 1);
        assert_eq!(default_cluster_count(12), 2);
        assert_eq!(default_cluster_count(60), 8);
    }
}
