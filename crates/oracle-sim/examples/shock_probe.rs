use oracle_ingest::data;
use oracle_sim::{simulate, ShockModel, SimConfig};

fn main() {
    let t = data::world_cup_2026();
    let model = data::fit_baseline_model(42);
    let _inputs = oracle_sim::LiveInputs {
        venue: data::matchup_adjustments(&t),
        shootout_rating: data::shootout_ratings(),
        knockout_pedigree: data::knockout_pedigree(),
        ..Default::default()
    };
    println!(
        "{:>6} {:>6} {:>9} {:>9} {:>9} {:>10}",
        "rho", "w", "top1", "top5", "entropy", "eff_teams"
    );
    for (rho, w) in [
        (0.0, 0.0),
        (0.3, 0.0),
        (0.6, 0.0),
        (0.9, 0.0),
        (-0.5, 0.0),
        (0.0, 0.5),
        (0.0, 0.9),
        (0.6, 0.3),
    ] {
        let f = simulate(
            &t,
            &model,
            SimConfig {
                iterations: 60_000,
                seed: 42,
                shocks: ShockModel {
                    attack_defence: rho,
                    environment: w,
                },
                ..Default::default()
            },
        );
        let mut p: Vec<f64> = f.teams.iter().map(|x| x.p_champion).collect();
        p.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let ent: f64 = -p
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|x| x * x.ln())
            .sum::<f64>();
        println!(
            "{:>6.2} {:>6.2} {:>8.2}% {:>8.2}% {:>9.4} {:>10.2}",
            rho,
            w,
            p[0] * 100.0,
            p[..5].iter().sum::<f64>() * 100.0,
            ent,
            ent.exp()
        );
    }
}
