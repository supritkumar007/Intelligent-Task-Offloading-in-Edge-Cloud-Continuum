from fastapi import FastAPI
import threading
import time

app = FastAPI()

latest_bw = 20.0

def monitor():

    global latest_bw

    prev_rx = None
    prev_time = time.time()

    while True:

        try:

            with open("/proc/net/dev") as f:
                lines = f.readlines()

            line = next(
                x for x in lines
                if "ogstun" in x
            )

            rx_bytes = int(
                line.split(":")[1].split()[0]
            )

            now = time.time()

            if prev_rx is not None:

                delta_bytes = rx_bytes - prev_rx
                delta_time = now - prev_time

                latest_bw = max(
                    (delta_bytes * 8) /
                    delta_time /
                    1e6,
                    0.1
                )

            prev_rx = rx_bytes
            prev_time = now

        except Exception as e:
            print(e)

        time.sleep(1)

threading.Thread(
    target=monitor,
    daemon=True
).start()



@app.get("/latest")
def latest():

    return {
        "bandwidth_mbps":
            round(latest_bw, 2)
    }
