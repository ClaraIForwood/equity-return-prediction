from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# QuantConnect Research imports
from QuantConnect.Research import QuantBook
from QuantConnect import Resolution

# Create a QuantBook instance (research environment) or use the algorithm context in an algorithm.
qb = QuantBook()

# Subscribe to SPY (uses AlgoSeek daily US-equity data in QuantConnect Research)
spy = qb.AddEquity("SPY").Symbol

# Define the time window - last 365 calendar days
end_date = datetime(2026, 2, 20)               # current date in Europe/London
start_date = end_date - timedelta(days=365)

# Request historical data (daily resolution) for SPY
history = qb.History([spy], start_date, end_date, Resolution.Daily)

# The returned DataFrame has a multi-index: date & symbol
spy_history = history.loc[spy]

# Plot closing prices
plt.figure(figsize=(10,6))
plt.plot(spy_history.index, spy_history['close'], label='SPY')
plt.title("SPY Closing Prices - last 1 year")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.grid(True)
plt.show()
