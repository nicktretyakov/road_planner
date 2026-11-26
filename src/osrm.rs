//! Клиент готового сервиса маршрутизации OSRM (OpenStreetMap-based engine).
//! Сервис совместим по URL и с GraphHopper-подобными развертываниями через
//! переменную окружения OSRM_URL (Google Maps API требует ключ и другой
//! протокол — см. README; здесь выбран открытый OSRM).
//!
//! Матрица времён/расстояний: GET {base}/table/v1/driving/{lon,lat;...}?annotations=duration,distance
//! При недоступности сервиса планировщик откатывается на Haversine + ML-прогноз.

use serde_json::Value;
use std::sync::Mutex;
use std::time::Duration;

/// Матрица времён (минуты) и расстояний (км) между точками.
#[derive(Debug, Clone)]
pub struct OsrmTable {
    pub minutes: Vec<Vec<f64>>,
    pub km: Vec<Vec<f64>>,
}

#[derive(Debug)]
pub struct OsrmClient {
    pub base: String,
    pub enabled: bool,
    agent: ureq::Agent,
    /// Последний статус вызова (для /api/ml/stats и диагностики).
    pub last_status: Mutex<String>,
}

impl OsrmClient {
    pub fn new(base: impl Into<String>, enabled: bool) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(4))
            .build();
        Self {
            base: base.into(),
            enabled,
            agent,
            last_status: Mutex::new("не использовался".into()),
        }
    }

    /// Конфигурация из окружения: OSRM_URL, OSRM_ENABLED (0/false — выключить).
    pub fn from_env() -> Self {
        let base =
            std::env::var("OSRM_URL").unwrap_or_else(|_| "https://router.project-osrm.org".into());
        let enabled = std::env::var("OSRM_ENABLED")
            .map(|v| {
                let v = v.to_lowercase();
                v != "0" && v != "false" && v != "off"
            })
            .unwrap_or(true);
        Self::new(base, enabled)
    }

    /// Клиент с выключенным OSRM (тесты, офлайн-режим).
    pub fn disabled() -> Self {
        Self::new("отключён", false)
    }

    /// Точки (lat, lon) → строка координат OSRM. ВНИМАНИЕ: OSRM ожидает "lon,lat".
    fn coords_string(pts: &[(f64, f64)]) -> String {
        pts.iter()
            .map(|p| format!("{:.6},{:.6}", p.1, p.0))
            .collect::<Vec<_>>()
            .join(";")
    }

    fn set_status(&self, s: String) {
        if let Ok(mut st) = self.last_status.lock() {
            *st = s;
        }
    }

    /// Матрица по всем парам точек. Ошибка — текст по-русски для клиента API.
    pub fn table(&self, pts: &[(f64, f64)]) -> Result<OsrmTable, String> {
        if !self.enabled {
            return Err("OSRM отключён (OSRM_ENABLED=0)".into());
        }
        if pts.len() < 2 {
            return Err("Для таблицы нужно минимум 2 точки".into());
        }
        let url = format!(
            "{}/table/v1/driving/{}?annotations=duration,distance",
            self.base.trim_end_matches('/'),
            Self::coords_string(pts)
        );
        let json: Value = match self.agent.get(&url).call() {
            Ok(r) => match r.into_json::<Value>() {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("Не удалось разобрать ответ OSRM: {e}");
                    self.set_status(msg.clone());
                    return Err(msg);
                }
            },
            Err(e) => {
                let msg = format!("OSRM недоступен: {e}");
                self.set_status(msg.clone());
                return Err(msg);
            }
        };
        match parse_table(&json) {
            Ok(t) => {
                self.set_status(format!("ok ({} точек, {})", pts.len(), self.base));
                Ok(t)
            }
            Err(e) => {
                self.set_status(format!("ошибка: {e}"));
                Err(e)
            }
        }
    }
}

/// Разбор ответа OSRM table (unit-тестируется без сети).
pub fn parse_table(v: &Value) -> Result<OsrmTable, String> {
    let code = v.get("code").and_then(Value::as_str).unwrap_or("");
    if code != "Ok" {
        return Err(format!("OSRM вернул код '{code}'"));
    }
    let dur = v
        .get("durations")
        .and_then(Value::as_array)
        .ok_or("в ответе нет массива durations")?;
    let dst = v
        .get("distances")
        .and_then(Value::as_array)
        .ok_or("в ответе нет массива distances")?;
    let n = dur.len();
    if dst.len() != n {
        return Err("размеры матриц durations/distances не совпадают".into());
    }
    let mut minutes = vec![vec![0.0; n]; n];
    let mut km = vec![vec![0.0; n]; n];
    for i in 0..n {
        let di = dur[i].as_array().ok_or("durations: строка не массив")?;
        let si = dst[i].as_array().ok_or("distances: строка не массив")?;
        if di.len() != n || si.len() != n {
            return Err("матрица не квадратная".into());
        }
        for j in 0..n {
            minutes[i][j] = di[j].as_f64().unwrap_or(f64::INFINITY) / 60.0;
            km[i][j] = si[j].as_f64().unwrap_or(f64::INFINITY) / 1000.0;
        }
    }
    Ok(OsrmTable { minutes, km })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_ok_response() {
        let v = json!({
            "code": "Ok",
            "durations": [[0.0, 600.0], [600.0, 0.0]],
            "distances": [[0.0, 5000.0], [5000.0, 0.0]]
        });
        let t = parse_table(&v).unwrap();
        assert!((t.minutes[0][1] - 10.0).abs() < 1e-9);
        assert!((t.km[0][1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_bad_code() {
        let v = json!({"code": "NoTable", "durations": [], "distances": []});
        assert!(parse_table(&v).is_err());
    }

    #[test]
    fn disabled_client_errors() {
        let c = OsrmClient::disabled();
        let pts = [(55.7, 37.6), (55.8, 37.7)];
        assert!(c.table(&pts).is_err());
        let st = c.last_status.lock().unwrap().clone();
        assert_eq!(st, "не использовался");
    }

    #[test]
    fn coords_string_is_lon_lat() {
        let s = OsrmClient::coords_string(&[(55.7539, 37.6208), (55.8, 37.7)]);
        assert_eq!(s, "37.620800,55.753900;37.700000,55.800000");
    }
}
