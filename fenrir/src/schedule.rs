use std::collections::HashMap;

use crate::agent::Temperature;

pub type LrScheduleTable = HashMap<String, fn(usize) -> f64>;
pub type TemperatureScheduleForTargetPolicy = HashMap<String, fn(usize) -> Temperature>;
pub type TemperatureScheduleForMoveSelection = HashMap<String, fn(usize) -> Temperature>;


pub fn lr_sch_initialize() -> LrScheduleTable {
    let mut hm: HashMap<String, fn(usize) -> f64> =  HashMap::new();

    // Below keep adding the schedule as you need it along with its name
    const fn agz_lr_schedule(n_steps: usize) -> f64 {
        match n_steps {
            0..400000 => 0.01f64,
            400000..600000 => 0.001f64,
            600000.. => 0.0001f64,
        }
    }
    hm.insert("AlphaGoZero".to_string(), agz_lr_schedule);

    hm
}

pub fn temp_sch_tp_initialize() -> TemperatureScheduleForTargetPolicy{
    let mut hm: HashMap<String, fn(usize) -> Temperature> = HashMap::new();

    const fn agz(n_steps: usize) -> Temperature{
        Temperature::Temp(1.0f32)
    }

    hm.insert("AlphaGoZero".to_string(), agz);

    hm
}

pub fn temp_sch_ms_initialize() -> TemperatureScheduleForMoveSelection {
    let mut hm: HashMap<String, fn(usize) -> Temperature > = HashMap::new();

    const fn agz(n_steps: usize) -> Temperature{
        match n_steps{
            0..30 => Temperature::Temp(1.0),
            30.. => Temperature::Zero,
        }
    }

    const fn evaluation(n_steps: usize) -> Temperature{
        Temperature::Zero
    }

    hm.insert("AlphaGoZero".to_string(), agz);
    hm.insert("evaluation".to_string(), evaluation);

    hm
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn initialize_works() {
        let table = lr_sch_initialize();
        assert!(table.contains_key("AlphaGoZero"));
        let agz = table["AlphaGoZero"];
        assert_eq!(agz(500000), 0.001f64);
    }
}
