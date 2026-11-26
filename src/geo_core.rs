//! Гео-ядро: расстояния по гаверсинусу через библиотеку `geo`,
//! проекция координат в локальную плоскость (км) и работа с временными окнами.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// Расстояние по гаверсинусу между точками (lat, lon), км.
/// Реализация — стандартный метод библиотеки `geo` (трейт HaversineDistance).
/// ВАЖНО: geo принимает Point::new(x=долгота, y=широта), поэтому внутри
/// выполняется перестановка компонентов.
pub fn haversine_km(a: (f64, f64), b: (f64, f64)) -> f64 {
    use geo::{HaversineDistance, Point};
    let p1 = Point::new(a.1, a.0); // (lon, lat)
    let p2 = Point::new(b.1, b.0);
    p1.haversine_distance(&p2) / 1000.0
}

/// Проекция (lat, lon) в локальную плоскость (x — восток, y — север) в километрах
/// относительно origin. Равнопромежуточная проекция: корректна на масштабах города/агломерации.
pub fn to_local_km(origin: (f64, f64), p: (f64, f64)) -> (f64, f64) {
    let km_per_deg_lat = 110.574;
    let km_per_deg_lon = 111.320 * origin.1.to_radians().cos();
    (
        (p.1 - origin.1) * km_per_deg_lon, // восток (по долготе)
        (p.0 - origin.0) * km_per_deg_lat, // север (по широте)
    )
}

/// Обратное преобразование локальных километров (x — восток, y — север) в (lat, lon).
pub fn from_local_km(origin: (f64, f64), v: (f64, f64)) -> (f64, f64) {
    let km_per_deg_lat = 110.574;
    let km_per_deg_lon = 111.320 * origin.1.to_radians().cos();
    (
        origin.0 + v.1 / km_per_deg_lat, // lat
        origin.1 + v.0 / km_per_deg_lon, // lon
    )
}

/// Разбор времени "ЧЧ:ММ" в минуты от начала суток.
pub fn parse_hh_mm(s: &str) -> Result<u32, String> {
    let parts: Vec<&str> = s.trim().split(':').collect();
    if parts.len() != 2 {
        return Err(format!("Ожидался формат ЧЧ:ММ, получено '{s}'"));
    }
    let h: u32 = parts[0].trim().parse().map_err(|_| format!("Некорректный час в '{s}'"))?;
    let m: u32 = parts[1].trim().parse().map_err(|_| format!("Некорректные минуты в '{s}'"))?;
    if h > 23 || m > 59 {
        return Err(format!("Время вне диапазона 00:00-23:59: '{s}'"));
    }
    Ok(h * 60 + m)
}

/// Форматирование минут от начала суток в "ЧЧ:ММ".
pub fn fmt_hh_mm(minutes: u32) -> String {
    let m = minutes % (24 * 60);
    format!("{:02}:{:02}", m / 60, m % 60)
}

/// Временное окно доставки: [start_min, end_min] — минуты от начала суток.
/// В JSON сериализуется строкой "ЧЧ:ММ-ЧЧ:ММ" (например "09:00-12:30").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeWindow {
    pub start_min: u32,
    pub end_min: u32,
}

impl TimeWindow {
    pub fn new(start_min: u32, end_min: u32) -> Result<Self, String> {
        if start_min >= end_min {
            return Err(format!(
                "Начало окна {} должно быть меньше конца {}",
                fmt_hh_mm(start_min),
                fmt_hh_mm(end_min)
            ));
        }
        if end_min > 24 * 60 {
            return Err("Конец окна не может выходить за 24:00".into());
        }
        Ok(Self { start_min, end_min })
    }

    /// Попадает ли момент времени (в минутах) в окно (учитываем открытие; закрытие — жёсткий дедлайн начала сервиса).
    pub fn contains_start(&self, minute: u32) -> bool {
        minute >= self.start_min
    }
}

impl fmt::Display for TimeWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", fmt_hh_mm(self.start_min), fmt_hh_mm(self.end_min))
    }
}

impl Serialize for TimeWindow {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for TimeWindow {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        parse_window(&s).map_err(serde::de::Error::custom)
    }
}

/// Разбор окна из строки "ЧЧ:ММ-ЧЧ:ММ".
pub fn parse_window(s: &str) -> Result<TimeWindow, String> {
    let parts: Vec<&str> = s.trim().split('-').collect();
    if parts.len() != 2 {
        return Err(format!(
            "Ожидался формат временного окна 'ЧЧ:ММ-ЧЧ:ММ', получено '{s}'"
        ));
    }
    let start = parse_hh_mm(parts[0])?;
    let end = parse_hh_mm(parts[1])?;
    TimeWindow::new(start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haversine_moscow_spb() {
        // Москва (Красная площадь) — Санкт-Петербург (Дворцовая площадь) ≈ 634 км
        let d = haversine_km((55.7539, 37.6208), (59.9398, 30.3146));
        assert!(d > 620.0 && d < 645.0, "ожидали ~634 км, получено {d}");
        assert!((haversine_km((55.0, 37.0), (55.0, 37.0)) - 0.0).abs() < 1e-9);
        // симметричность
        let d2 = haversine_km((59.9398, 30.3146), (55.7539, 37.6208));
        assert!((d - d2).abs() < 1e-9);
    }

    #[test]
    fn haversine_short_distance() {
        // ~1 км по широте
        let d = haversine_km((55.0, 37.0), (55.009, 37.0));
        assert!(d > 0.95 && d < 1.05, "получено {d}");
    }

    #[test]
    fn local_projection_roundtrip() {
        let origin = (55.7539, 37.6208);
        let p = (55.8, 37.7);
        let v = to_local_km(origin, p);
        let back = from_local_km(origin, v);
        assert!((back.0 - p.0).abs() < 1e-9 && (back.1 - p.1).abs() < 1e-9);
    }

    #[test]
    fn time_parse_and_format() {
        assert_eq!(parse_hh_mm("09:30").unwrap(), 570);
        assert_eq!(fmt_hh_mm(570), "09:30");
        assert!(parse_hh_mm("24:00").is_err());
        assert!(parse_hh_mm("abc").is_err());
        assert!(parse_hh_mm("09:99").is_err());
    }

    #[test]
    fn window_parse_and_serde() {
        let w = parse_window("09:00-12:30").unwrap();
        assert_eq!((w.start_min, w.end_min), (540, 750));
        assert!(parse_window("12:30-09:00").is_err());
        assert!(parse_window("просто строка").is_err());
        let json = serde_json::to_string(&w).unwrap();
        assert_eq!(json, "\"09:00-12:30\"");
        let w2: TimeWindow = serde_json::from_str(&json).unwrap();
        assert_eq!(w, w2);
    }
}
