from src.data.fred_client import FredClient
from dotenv import load_dotenv
load_dotenv()   # 强制从项目根目录读取 .env

client = FredClient.from_env()

# 1) 拉最终版 INDPRO
df_indpro = client.fetch_final_series("INDPRO", observation_start="1980-01-01")
print(df_indpro.head())

# 2) 拉带 vintage 的 GDP 增长率（注意可能很大）
df_gdp_vint = client.fetch_vintage_series_full("A191RL1Q225SBEA", observation_start="1980-01-01")
print(df_gdp_vint.head())
