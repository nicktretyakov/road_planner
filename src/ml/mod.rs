//! Машинное обучение сервера:
//! - `sgd` — онлайн-регрессия времени в пути (стochastic gradient descent);
//! - `bandit` — UCB1-бандит, выбирающий стратегию улучшения маршрута по
//!   накопленной награде (обучение с подкреплением без модели).

pub mod bandit;
pub mod sgd;

pub use bandit::Bandit;
pub use sgd::SgdModel;

use serde::Serialize;

/// Имена «рычагов» бандита — стратегии улучшения маршрутов.
pub const IMPROVEMENT_ARMS: [&str; 4] = ["nn_2opt", "nn_oropt", "nn_2opt_oropt", "nn_sa"];

/// Информация о последней кластеризации (для /api/ml/stats).
#[derive(Debug, Clone, Serialize)]
pub struct ClusterInfo {
    pub k: usize,
    pub sizes: Vec<usize>,
    pub inertia: f64,
}

/// Движок ML: SGD-модель + бандит + счётчики обучения.
#[derive(Debug)]
pub struct MlEngine {
    pub sgd: SgdModel,
    pub bandit: Bandit,
    /// Сколько фактических поездок получено через /api/ml/feedback.
    pub feedback_count: u64,
    /// Сколько раз запускалось планирование.
    pub plan_count: u64,
    pub last_clusters: Option<ClusterInfo>,
}

impl MlEngine {
    pub fn new() -> Self {
        Self {
            sgd: SgdModel::new(),
            bandit: Bandit::new(&IMPROVEMENT_ARMS),
            feedback_count: 0,
            plan_count: 0,
            last_clusters: None,
        }
    }

    /// Человекочитаемое описание рычага бандита.
    pub fn arm_description(name: &str) -> &'static str {
        match name {
            "nn_2opt" => "жадный ближайший сосед + 2-opt",
            "nn_oropt" => "жадный ближайший сосед + Or-opt",
            "nn_2opt_oropt" => "жадный ближайший сосед + 2-opt и Or-opt",
            "nn_sa" => "жадный ближайший сосед + имитация отжига",
            _ => "неизвестная стратегия",
        }
    }
}

impl Default for MlEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_boots_with_four_arms() {
        let e = MlEngine::new();
        assert_eq!(e.bandit.arms(), 4);
        assert_eq!(e.sgd.n_samples, 0);
    }
}
