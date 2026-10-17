import pandas as pd
import textwrap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

#TODO Clean it up

df = pd.read_csv('F:/Milestone 2/events_list.csv')

df['Date'] = pd.to_datetime(df['Date'])

# 2. Setup the figure
fig, ax = plt.subplots(figsize=(30, 10), constrained_layout=True)

ax.set_ylim(-2, 2)

# Create baseline timeline axis
ax.axhline(0, color="black", lw=2)
ax.plot(df['Date'], [0] * len(df['Date']), "o", color="royalblue", markersize=10, zorder=3)

# Alternating heights to prevent overlap
levels = np.tile([1,-1,0.2,-0.2,1.8,-1.8], int(np.ceil(len(df['Date'])/3)))[:len(df['Date'])]

for i, (date, event, level) in enumerate(zip(df['Date'], df['Event'], levels)):
    # Format the date to display inside the text block
    date_str = date.strftime("%b %d, %Y")

    # Wrap the long description text at 35 characters
    wrapped_event = "\n".join(textwrap.wrap(event, width=35))

    # Combine date and text (Date is bolded via matplotlib's math text styling)
    full_text = f"$\mathbf{{{date_str}}}$\n{wrapped_event}"

    # Draw vertical stem line from the timeline dot to the text box
    ax.vlines(date, 0, level, color="gray", linestyle="--", linewidth=1)

    # Position text box above or below the line based on level sign
    va = 'bottom' if level > 0 else 'top'

    ax.text(
        date, level, full_text,
        ha='center', va=va,
        bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", ec="royalblue", lw=1.5),
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

plt.show()