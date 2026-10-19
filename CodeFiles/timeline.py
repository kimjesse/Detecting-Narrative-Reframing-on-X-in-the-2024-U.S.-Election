import textwrap
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


df = pd.read_csv(SCRIPT_DIR.parent / 'DataFiles' / 'events_list.csv')

df['Date'] = pd.to_datetime(df['Date'])

fig, ax = plt.subplots(figsize=(35, 10), constrained_layout=True)

ax.set_ylim(-2, 2)

# Create baseline timeline axis
ax.axhline(0, color="black", lw=2)
ax.plot(df['Date'], [0] * len(df['Date']), "o", color="royalblue", markersize=10, zorder=3)

# Alternating heights to prevent overlap
levels = np.tile([0.8,-0.8,0.2,-0.2,1.2,-1.3], int(np.ceil(len(df['Date'])/3)))[:len(df['Date'])]

for i, (date, event, level) in enumerate(zip(df['Date'], df['Event'], levels)):
    # Format the date to display inside the text block
    date_str = date.strftime("%b %d, %Y")

    # Wrap the long description text at 35 characters
    wrapped_event = "\n".join(textwrap.wrap(event, width=35))

    # Combine date and text (Date is bolded via matplotlib's math text styling)
    #Create newline variable so it can be interpreted by the raw string
    nl = "\n"
    full_text = rf"$\mathbf{{{date_str}}}${nl}{wrapped_event}"

    # Draw vertical stem line from the timeline dot to the text box
    ax.vlines(date, 0, level, color="gray", linestyle="--", linewidth=1)

    # Position text box above or below the line based on level sign
    va = 'bottom' if level > 0 else 'top'

    ax.text(
        date, level, full_text,
        ha='center', va=va,
        bbox={'boxstyle':"round,pad=0.5", 'fc':"whitesmoke", 'ec':"royalblue", 'lw':1.5},
        fontsize=9.5
    )

# 4. Clean up axes and format timelines
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

# Hide y-axis and unnecessary borders
ax.get_yaxis().set_visible(False)
for spine in ["left", "right", "top"]:
    ax.spines[spine].set_visible(False)

plt.savefig("timeline.png")
plt.show()
