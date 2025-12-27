import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Waterflood Data
# Response to injection with lag
dates = [datetime(2020, 1, 1) + timedelta(days=x) for x in range(1000)]
inj = np.random.normal(500, 50, 1000) # Injection
inj[300:400] = 0 # Injector shut-in
# Oil responds to Inj with lag of 30 days + decline
oil = np.zeros(1000)
decline = 1000 * np.exp(-0.002 * np.arange(1000))
response = np.convolve(inj, np.ones(30)/30, mode='same') * 0.5 # Connectivity 0.5
oil = decline + response
df_wf = pd.DataFrame({'date': dates, 'oil_rate': oil, 'water_injection': inj})
df_wf.to_csv("data/sample_waterflood.csv", index=False)

# 2. Pressure Data (Build-up)
# Flowing then shut-in
t_p = np.arange(100)
rate_p = 500 * np.exp(-0.01 * t_p)
rate_p[50:60] = 0 # Shut-in
pres_p = 2000 - 0.5 * rate_p # Pwf = Pres - q/PI
# During shut-in, pressure builds up
pres_p[50:60] = 2000 - (2000 - pres_p[49]) * np.exp(-0.5 * (np.arange(10)))
df_p = pd.DataFrame({'date': dates[:100], 'oil_rate': rate_p, 'pressure': pres_p})
df_p.to_csv("data/sample_pressure.csv", index=False)

# 3. Downtime Data (Random shut-ins)
rate_d = 1000 * np.exp(-0.0015 * np.arange(1000))
# Add random noise
rate_d += np.random.normal(0, 20, 1000)
# Add random 0s
mask = np.random.rand(1000) > 0.95
rate_d[mask] = 0
df_d = pd.DataFrame({'date': dates, 'oil_rate': rate_d})
df_d.to_csv("data/sample_downtime.csv", index=False)

# 4. Field Data (Spatial)
lats = np.random.uniform(31.0, 32.0, 20)
lons = np.random.uniform(-103.0, -102.0, 20)
eurs = np.random.uniform(100000, 500000, 20)
# Simple spatial trend: Higher EUR in NE corner
eurs += (lats - 31.0) * 100000 + (lons + 103) * 100000
df_f = pd.DataFrame({'well_id': [f'Well_{i}' for i in range(20)], 'lat': lats, 'lon': lons, 'eur': eurs})
df_f.to_csv("data/sample_field.csv", index=False)
