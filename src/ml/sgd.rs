//! Онлайн-линейная регрессия времени в пути методом стохастического
//! градиентного спуска (SGD). Модель дообучается на каждой фактической
//! поездке из /api/ml/feedback и используется для:
//! - прогноза ETA между точками (когда OSRM недоступен);
//! - коэффициента дорожной загрузки (traffic factor) поверх времён OSRM.
//!
//! Признаки: [1, расстояние (декакилометры), «час пик»].
//! Целевая величина — фактическое время в пути, минуты.
//!
//! Это простая, но настоящая ML-модель: онлайн-обучение, L2-регуляризация,
//! скользящий MSE, физические ограничения прогноза.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SgdModel {
    /// Веса [свободный член, за 10 км, за пик].
    pub w: [f64; 3],
    pub lr: f64,
    pub l2: f64,
    pub n_samples: u64,
    /// Скользящая среднеквадратичная ошибка (мин²).
    pub mse: f64,
}

impl SgdModel {
    /// Априорные веса: 4 мин базово, 2 мин/км (~30 км/ч городская скорость),
    /// +8 мин в час пик. Модель сразу полезна, дообучение уточняет веса.
    pub fn new() -> Self {
        Self { w: [4.0, 20.0, 8.0], lr: 0.05, l2: 1e-4, n_samples: 0, mse: 0.0 }
    }

    /// Коэффициент «час пик»: 1.0 в утренние (07–09) и вечерние (17–19) часы,
    /// минимальный ночью, промежуточный в остальное время.
    pub fn peak_factor(clock_min: u32) -> f64 {
        let h = clock_min / 60;
        match h {
            7..=9 => 1.0,
            17..=19 => 1.0,
            10 => 0.6,
            16 => 0.6,
            20 => 0.5,
            0..=5 => 0.05,
            _ => 0.3,
        }
    }

    /// Вектор признаков: [1, dist_km / 10, пик].
    pub fn features(dist_km: f64, clock_min: u32) -> [f64; 3] {
        [1.0, dist_km / 10.0, Self::peak_factor(clock_min)]
    }

    /// Сырое предсказание (линейная комбинация).
    pub fn predict_raw(&self, dist_km: f64, clock_min: u32) -> f64 {
        let x = Self::features(dist_km, clock_min);
        self.w[0] * x[0] + self.w[1] * x[1] + self.w[2] * x[2]
    }

    /// Предсказание с физическим ограничением снизу: не быстрее 60 км/ч и
    /// не меньше 1 минуты на любой переезд.
    pub fn predict(&self, dist_km: f64, clock_min: u32) -> f64 {
        let floor = (dist_km * 1.0).max(1.0);
        self.predict_raw(dist_km, clock_min).max(floor)
    }

    /// Один шаг SGD по фактической поездке. Возвращает абсолютную ошибку, мин.
    pub fn update(&mut self, dist_km: f64, clock_min: u32, actual_min: f64) -> f64 {
        let x = Self::features(dist_km, clock_min);
        let pred = self.predict_raw(dist_km, clock_min);
        let err = pred - actual_min;
        for i in 0..3 {
            self.w[i] -= self.lr * (err * x[i] + self.l2 * self.w[i]);
        }
        self.n_samples += 1;
        let sq = err * err;
        self.mse = if self.n_samples == 1 { sq } else { self.mse + (sq - self.mse) / self.n_samples as f64 };
        err.abs()
    }

    /// Коэффициент дорожной загрузки: прогноз модели относительно базовой
    /// скорости 30 км/ч. Применяется как множитель к временам OSRM,
    /// когда накоплена реальная история поездок.
    pub fn traffic_factor(&self, dist_km: f64, clock_min: u32) -> f64 {
        if dist_km < 0.05 {
            return 1.0;
        }
        let base = dist_km * 2.0; // 30 км/ч
        (self.predict_raw(dist_km, clock_min) / base).clamp(0.6, 2.5)
    }
}

impl Default for SgdModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_to_linear_target() {
        // целевая зависимость: 4 + 3 мин/км, «пик» не влияет
        let mut m = SgdModel::new();
        m.lr = 0.02;
        let clocks = [540u32, 720, 900, 1080, 480, 1060]; // разные факторы пика
        for i in 0..6000 {
            let dist = 0.5 + ((i * 37) % 90) as f64 / 10.0; // 0.5..9.5 км
            let actual = 4.0 + 3.0 * dist;
            m.update(dist, clocks[i % clocks.len()], actual);
        }
        assert!((m.w[1] - 30.0).abs() < 2.0, "вес за 10 км должен стремиться к 30, получено {}", m.w[1]);
        assert!((m.w[0] - 4.0).abs() < 3.0, "свободный член должен стремиться к 4, получено {}", m.w[0]);
        assert!(m.w[2].abs() < 3.0, "вес пика должен уйти к нулю, получено {}", m.w[2]);
        assert!(m.mse < 25.0, "MSE должно упасть, получено {}", m.mse);
    }

    #[test]
    fn predict_respects_floor() {
        let m = SgdModel::new();
        assert!(m.predict(0.01, 720) >= 1.0);
        // 10 км не может занимать меньше 10 минут (60 км/ч)
        assert!(m.predict(10.0, 720) >= 10.0);
    }

    #[test]
    fn peak_factor_profile() {
        assert!((SgdModel::peak_factor(8 * 60) - 1.0).abs() < 1e-9);
        assert!((SgdModel::peak_factor(18 * 60) - 1.0).abs() < 1e-9);
        assert!(SgdModel::peak_factor(3 * 60) < 0.1);
        assert!(SgdModel::peak_factor(13 * 60) < 1.0);
    }

    #[test]
    fn traffic_factor_sane() {
        let m = SgdModel::new();
        let f = m.traffic_factor(10.0, 8 * 60); // час пик
        let n = m.traffic_factor(10.0, 3 * 60); // ночь
        assert!(f > n, "в час пик фактор должен быть выше ночного");
        assert!((0.6..=2.5).contains(&f));
    }
}
