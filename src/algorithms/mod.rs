//! Алгоритмы маршрутизации: ближайший сосед, кластеризация K-means,
//! TSP (2-opt / Or-opt / имитация отжига) и VRPTW с временными окнами.

pub mod kmeans;
pub mod nearest;
pub mod tsp;
pub mod vrptw;
