from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

log_file = "../logs/sac/SAC_33/events.out.tfevents.1775240671.AsusTuf.9076.0"

ea = event_accumulator.EventAccumulator(log_file)
ea.Reload()

data = []

for tag in ea.Tags()["scalars"]:
    events = ea.Scalars(tag)
    if len(events) > 0:
        last = events[-1]
        data.append({
            "metric": tag,
            "step": last.step,
            "value": last.value
        })

df = pd.DataFrame(data)

# sort nicely
df = df.sort_values(by="metric")

print(df.to_string(index=False))