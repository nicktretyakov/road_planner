//! UCB1 — бандит с верхней доверительной границей (Auer et al., 2002).
//! Выбирает стратегию улучшения маршрута: сначала каждый «рычаг» по разу,
//! затем argmax(средняя награда + c·√(ln T / n_i)).
//!
//! Награда — процент сокращения маршрута относительно жадной вставки,
//! нормированный на время работы шага (обе составляющие стимулируют и
//! качество, и скорость).

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ArmStat {
    pub name: String,
    pub pulls: u64,
    pub mean_reward: f64,
}

#[derive(Debug)]
pub struct Bandit {
    names: Vec<String>,
    pulls: Vec<u64>,
    rewards: Vec<f64>,
    c: f64,
}

impl Bandit {
    pub fn new(names: &[&str]) -> Self {
        let names: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        let n = names.len();
        Self {
            names,
            pulls: vec![0; n],
            rewards: vec![0.0; n],
            c: std::f64::consts::SQRT_2,
        }
    }

    pub fn arms(&self) -> usize {
        self.names.len()
    }

    /// Выбор рычага: все по разу, затем UCB1.
    pub fn select(&self) -> usize {
        if let Some(i) = self.pulls.iter().position(|&p| p == 0) {
            return i;
        }
        let total: u64 = self.pulls.iter().sum();
        let ln_t = (total as f64).ln().max(1e-9);
        let ucb = |i: usize| {
            self.rewards[i] / self.pulls[i] as f64 + self.c * (ln_t / self.pulls[i] as f64).sqrt()
        };
        (0..self.pulls.len())
            .max_by(|a, b| ucb(*a).partial_cmp(&ucb(*b)).unwrap())
            .unwrap()
    }

    /// Регистрация награды выбранного рычага.
    pub fn update(&mut self, arm: usize, reward: f64) {
        self.pulls[arm] += 1;
        self.rewards[arm] += reward;
    }

    pub fn stats(&self) -> Vec<ArmStat> {
        (0..self.names.len())
            .map(|i| ArmStat {
                name: self.names[i].clone(),
                pulls: self.pulls[i],
                mean_reward: if self.pulls[i] > 0 {
                    self.rewards[i] / self.pulls[i] as f64
                } else {
                    0.0
                },
            })
            .collect()
    }

    pub fn total_rounds(&self) -> u64 {
        self.pulls.iter().sum()
    }

    pub fn arm_name(&self, i: usize) -> &str {
        &self.names[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARMS: [&str; 4] = ["a", "b", "c", "d"];

    #[test]
    fn explores_each_arm_once() {
        let b = Bandit::new(&ARMS);
        assert_eq!(b.select(), 0);
        let mut b = b;
        for i in 0..4 {
            let a = b.select();
            assert_eq!(a, i);
            b.update(a, 1.0);
        }
    }

    #[test]
    fn exploits_best_arm() {
        let mut b = Bandit::new(&ARMS);
        // прогрев: каждый рычаг один раз со своей «истинной» наградой
        let true_rewards = [2.0, 10.0, 1.0, 5.0];
        for i in 0..4 {
            b.update(i, true_rewards[i]);
        }
        // детерминированная фаза эксплуатации
        for _ in 0..500 {
            let a = b.select();
            b.update(a, true_rewards[a]);
        }
        let st = b.stats();
        assert_eq!(st[1].name, "b");
        assert!(st[1].pulls > 400, "лучший рычаг должен тянуться чаще всего: {:?}", st);
        assert!(st[2].pulls <= st[0].pulls + 1, "худший рычаг не должен тянуться часто: {:?}", st);
        assert!((st[1].mean_reward - 10.0).abs() < 1e-9);
    }

    #[test]
    fn stats_zero_before_use() {
        let b = Bandit::new(&ARMS);
        assert_eq!(b.total_rounds(), 0);
        for s in b.stats() {
            assert_eq!(s.mean_reward, 0.0);
        }
    }
}
