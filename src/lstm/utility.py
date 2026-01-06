# 23.12.2025; Happy new years everyone
import time

class TimeTracker:
    def __init__(self): # Time is started after class initialization
        self.start_time = time.time()

    def get_elapsed_time(self):
        return round(time.time() - self.start_time, 2)
    
    def reset_timer(self):
        self.start_time = time.time()
    

if __name__ == "__main__":
    timer = TimeTracker()

    time.sleep(2)

    print(f"1. Time: {timer.get_elapsed_time()} seconds")
    timer.reset_timer()
    time.sleep(3)

    print(f"2. Time after reset: {timer.get_elapsed_time()} seconds")