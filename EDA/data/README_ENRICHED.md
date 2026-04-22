# HYSPLIT ConvLSTM dataset enriched with GDAS meteorology

Added broadcast meteorology channels to input and target tensors:
- u10m_ms
- v10m_ms
- wspd10_ms
- wdir_sin
- wdir_cos
- pblh_m
- sfcp_hpa
- rh2m_pct
- t02m_k

Channel order is original plume channels first, then the meteorology channels above.
