import time
import pandas as pd

def sleep(duration):
    duration=duration*1000000000
    now = time.perf_counter_ns()
    end = now + duration
    while now < end:
        now = time.perf_counter_ns()

df=pd.read_csv("../datasets/sleep_4mins.csv", index_col=False, header=None)
df = df.iloc[:, [0,1,7]]
df.columns=['t','f','y']

df_murcia=pd.read_csv("../datasets/sleep_4mins_murcia.csv", index_col=False, header=None)
df_murcia=df_murcia.iloc[:,[2,5,6]]
df_murcia.columns=['t','f','y']

print(df)
print(df_murcia)

df=df.append(df_murcia)

print(df)
df.to_csv("../datasets/sleep_4mins_Murciaconcat.csv",header=False,index=False)