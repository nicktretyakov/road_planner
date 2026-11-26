//! Графы и поиск кратчайшего пути: представление географических точек как
//! вершин, рёбра — географические расстояния. Алгоритм Дейкстры (двоичная куча).
//!
//! Реализованы два графа:
//! - `GridGraph` — синтетическая дорожная сетка поверх ограничивающего прямоугольника
//!   (bbox) между двумя точками: узлы — узлы решётки, рёбра — 4-связность,
//!   вес ребра = Haversine × коэффициент извилистости дорог;
//! - `CompleteGraph` — полный граф «каждый с каждым» (используется неявно
//!   матрицами расстояний в TSP/VRPTW).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::geo_core::haversine_km;

/// Абстракция графа для поиска кратчайшего пути.
pub trait Graph {
    fn num_nodes(&self) -> usize;
    /// Список смежности: (сосед, вес ребра).
    fn neighbors(&self, u: usize) -> Vec<(usize, f64)>;
    /// Географические координаты узла (lat, lon).
    fn coords(&self, u: usize) -> (f64, f64);
}

/// Результат Дейкстры: длина пути, последовательность узлов и число
/// просмотренных рёбер (для статистики).
#[derive(Debug, Clone)]
pub struct DijkstraResult {
    pub distance: f64,
    pub path: Vec<usize>,
    pub edges_checked: usize,
}

/// Элемент кучи: (приоритет-расстояние, узел) с Nan-безопасным порядком
/// (f64 не реализует Ord, поэтому делаем обёртку).
#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapItem(f64, usize);

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // в куче нет NaN — расстояния всегда конечны
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.1.cmp(&other.1))
    }
}

/// Классический алгоритм Дейкстры с двоичной кучей, O((V+E) log V).
pub fn shortest_path(g: &dyn Graph, src: usize, dst: usize) -> Option<DijkstraResult> {
    let n = g.num_nodes();
    if src >= n || dst >= n {
        return None;
    }
    let mut dist = vec![f64::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    let mut heap = BinaryHeap::new();
    let mut edges_checked = 0usize;
    dist[src] = 0.0;
    heap.push(Reverse(HeapItem(0.0, src)));

    while let Some(Reverse(HeapItem(d, u))) = heap.pop() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        if u == dst {
            break;
        }
        for (v, w) in g.neighbors(u) {
            edges_checked += 1;
            let nd = d + w;
            if nd < dist[v] {
                dist[v] = nd;
                prev[v] = u;
                heap.push(Reverse(HeapItem(nd, v)));
            }
        }
    }
    if dist[dst].is_infinite() {
        return None;
    }
    let mut path = vec![dst];
    let mut cur = dst;
    while cur != src {
        cur = prev[cur];
        path.push(cur);
    }
    path.reverse();
    Some(DijkstraResult { distance: dist[dst], path, edges_checked })
}

/// Синтетическая дорожная сетка (решётка) над ограничивающим прямоугольником.
/// Узлы — вершины решётки с шагом `step`; рёбра — 4-связность,
/// вес = расстояние Haversine × `detour` (коэффициент извилистости улиц).
pub struct GridGraph {
    pub lat_min: f64,
    pub lon_min: f64,
    pub lat_max: f64,
    pub lon_max: f64,
    pub nx: usize,
    pub ny: usize,
    pub step: f64,
    pub detour: f64,
}

impl GridGraph {
    const MAX_NODES: usize = 4096;
    const MAX_SIDE: usize = 80;

    /// Сетка между двумя точками с шагом `step_km` (запас 2 км вокруг bbox).
    /// Если узлов слишком много, шаг автоматически увеличивается.
    pub fn between(from: (f64, f64), to: (f64, f64), step_km: f64) -> Self {
        let margin = 2.0 / 111.32; // ~2 км в градусах широты
        let lat_min = from.0.min(to.0) - margin;
        let lat_max = from.0.max(to.0) + margin;
        let lon_min = from.1.min(to.1) - margin;
        let lon_max = from.1.max(to.1) + margin;
        let mid_lat = (lat_min + lat_max) / 2.0;
        let km_per_deg_lon = 111.320 * mid_lat.to_radians().cos();
        let span_lat_km = (lat_max - lat_min) * 110.574;
        let span_lon_km = (lon_max - lon_min) * km_per_deg_lon;

        let mut step = step_km.max(0.05);
        let mut nx = ((span_lon_km / step).ceil() as usize + 1).max(2);
        let mut ny = ((span_lat_km / step).ceil() as usize + 1).max(2);
        if nx * ny > Self::MAX_NODES {
            let scale = ((nx as f64 * ny as f64) / Self::MAX_NODES as f64).sqrt();
            step *= scale;
            nx = ((span_lon_km / step).ceil() as usize + 1).min(Self::MAX_SIDE).max(2);
            ny = ((span_lat_km / step).ceil() as usize + 1).min(Self::MAX_SIDE).max(2);
        }
        Self {
            lat_min,
            lon_min,
            lat_max,
            lon_max,
            nx,
            ny,
            step,
            detour: 1.25, // средняя извилистость городских дорог
        }
    }

    fn km_per_deg_lon(&self) -> f64 {
        111.320 * (((self.lat_min + self.lat_max) / 2.0).to_radians().cos())
    }

    fn node_latlon(&self, i: usize, j: usize) -> (f64, f64) {
        (
            self.lat_min + j as f64 * self.step / 110.574,
            self.lon_min + i as f64 * self.step / self.km_per_deg_lon(),
        )
    }

    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Ближайший узел сетки к точке (с ограничением границами).
    pub fn nearest_node(&self, p: (f64, f64)) -> usize {
        let i = (((p.1 - self.lon_min) * self.km_per_deg_lon() / self.step).round() as isize)
            .clamp(0, self.nx as isize - 1) as usize;
        let j = (((p.0 - self.lat_min) * 110.574 / self.step).round() as isize)
            .clamp(0, self.ny as isize - 1) as usize;
        self.idx(i, j)
    }
}

impl Graph for GridGraph {
    fn num_nodes(&self) -> usize {
        self.nx * self.ny
    }

    fn coords(&self, u: usize) -> (f64, f64) {
        let (i, j) = (u % self.nx, u / self.nx);
        self.node_latlon(i, j)
    }

    fn neighbors(&self, u: usize) -> Vec<(usize, f64)> {
        let (i, j) = (u % self.nx, u / self.nx);
        let cu = self.node_latlon(i, j);
        let mut out = Vec::with_capacity(4);
        let mut push = |ii: usize, jj: usize| {
            let cv = self.node_latlon(ii, jj);
            out.push((self.idx(ii, jj), haversine_km(cu, cv) * self.detour));
        };
        if i > 0 {
            push(i - 1, j);
        }
        if i + 1 < self.nx {
            push(i + 1, j);
        }
        if j > 0 {
            push(i, j - 1);
        }
        if j + 1 < self.ny {
            push(i, j + 1);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Маленький рукотворный граф: 0-1 = 1, 1-2 = 2, 0-2 = 10.
    struct Tiny;
    impl Graph for Tiny {
        fn num_nodes(&self) -> usize {
            3
        }
        fn neighbors(&self, u: usize) -> Vec<(usize, f64)> {
            match u {
                0 => vec![(1, 1.0), (2, 10.0)],
                1 => vec![(0, 1.0), (2, 2.0)],
                _ => vec![(1, 2.0), (0, 10.0)],
            }
        }
        fn coords(&self, u: usize) -> (f64, f64) {
            (55.0 + u as f64, 37.0)
        }
    }

    #[test]
    fn dijkstra_finds_shorter_indirect_path() {
        let r = shortest_path(&Tiny, 0, 2).unwrap();
        assert!((r.distance - 3.0).abs() < 1e-9, "ожидали 3 км, получено {}", r.distance);
        assert_eq!(r.path, vec![0, 1, 2]);
    }

    #[test]
    fn dijkstra_same_node() {
        let r = shortest_path(&Tiny, 1, 1).unwrap();
        assert_eq!(r.distance, 0.0);
        assert_eq!(r.path, vec![1]);
    }

    #[test]
    fn grid_between_close_points() {
        let g = GridGraph::between((55.75, 37.62), (55.80, 37.70), 1.0);
        assert!(g.num_nodes() <= GridGraph::MAX_NODES);
        let s = g.nearest_node((55.75, 37.62));
        let t = g.nearest_node((55.80, 37.70));
        let r = shortest_path(&g, s, t).unwrap();
        let straight = haversine_km((55.75, 37.62), (55.80, 37.70));
        assert!(r.distance >= straight * 0.99, "сетевой путь короче прямой линии");
        assert!(r.distance <= straight * 2.0, "сетевой путь подозрительно длинный: {} vs {straight}", r.distance);
        assert_eq!(r.path.first(), Some(&s));
        assert_eq!(r.path.last(), Some(&t));
    }

    #[test]
    fn nearest_node_clamps_to_bounds() {
        let g = GridGraph::between((55.75, 37.62), (55.80, 37.70), 1.0);
        let outside = g.nearest_node((60.0, 40.0));
        assert!(outside < g.num_nodes());
    }
}
