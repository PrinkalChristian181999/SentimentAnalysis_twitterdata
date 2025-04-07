import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from queue import Queue
import threading


windowSize = 50
sentimentScores = deque(maxlen=windowSize)
data_queue = Queue()

def watchSentimentUpdate(db):
    collection = db["sentimentAnalysis"]
    pipeline = []

    def run_watch():
        with collection.watch(pipeline) as stream:
            print("Watching for real-time sentiment updates...")
            for change in stream:
                if change["operationType"] == "insert":
                    graphChange = change["fullDocument"]
                    sentiment = graphChange.get("sentimentAnalysis", {})
                    score = sentiment.get("score", None)
                    if score is not None:
                        sentimentScores.append(score)
                        data_queue.put(list(sentimentScores))  # Send a copy

    # Run watcher in background thread
    threading.Thread(target=run_watch, daemon=True).start()

def updateGraph():
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'bo-', label="Sentiment Score")
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, windowSize)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("Live Sentiment Score")
    ax.set_xlabel("Data Point")
    ax.set_ylabel("Sentiment Score")
    ax.grid(True)
    
    ax.legend()

    def update(frame):
        if not data_queue.empty():
            data = data_queue.get()
            x_vals = list(range(len(data)))
            y_vals = data

            colors = ['red' if score > 0.25 else 'blue' for score in y_vals]
            

            line.set_data(x_vals, y_vals)

            for i, score in enumerate(y_vals):
                line.set_markerfacecolor(colors[i])

            ax.set_xlim(0, max(len(data), windowSize))
        return line,

    ani = animation.FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
