# Processed dataset notes
- Source cells B0005, B0006, B0007, B0018 (.mat), discharge cycles only (2 A CC discharge, ambient 24 °C per cycle metadata).
- Discharge cycles kept: 168, 168, 168, 132; samples: 50,285 / 50,285 / 50,285 / 34,866 (185,721 total). No rows dropped for NaN/inf.
- Temperature range (°C): B0005 23.2–41.5, B0006 23.2–42.0, B0007 23.0–42.3, B0018 22.4–38.9.
- Features (per sample, computed within each cycle, using only current/past samples of that cycle):
  V (Voltage_measured), I (Current_measured), P = V·|I| (W), t = time since cycle start (s),
  E_cum = Σ P·Δt /3600 (Wh), Q_cum = Σ |I|·Δt /3600 (Ah), dV = V(0) − V(t) (V),
  amb = ambient temperature from cycle metadata (constant 24 °C here; kept for generality), cycle_idx = discharge-cycle count (ageing proxy).
- No temperature history is used as input (no lagged temperature), so the task is pure electrical-to-thermal estimation (virtual sensor), not autoregressive forecasting.
- Charge cycles not used. Sampling interval is non-uniform; Δt from the Time vector.
