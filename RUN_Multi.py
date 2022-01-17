import subprocess
import subprocess
import time
import threading
import queue


python = "./venv/Scripts/python.exe"

class FlushPipe(object):
    def __init__(self, model):
        self.model = model
        self.command = [python, './RUN.py', model]
        self.process = None
        self.process_output = queue.LifoQueue(0)
        self.capture_output = threading.Thread(target=self.output_reader)

    def output_reader(self):
        for line in iter(self.process.stdout.readline, b''):
            self.process_output.put_nowait(line)

    def start_process(self):
        self.process = subprocess.Popen(self.command,
                                        stdout=subprocess.PIPE)
        self.capture_output.start()

    def get_output_for_processing(self):
        line = str(self.process_output.get())
        if self.model in line: print (line.lstrip())


if __name__ == "__main__":
    # models = ["DDPG", "SAC", "TD3"]
    models = ["DDPG", "TD3"]

    procs = []

    for m in models:
        flush_pipe = FlushPipe(m)
        flush_pipe.start_process()
        procs.append(flush_pipe)
        time.sleep(10)

    now = time.time()
    while True:
        for pp in procs:
            pp.get_output_for_processing()
            time.sleep(0.1)
            pp.capture_output.join(timeout=0.001)


#
#
#
#
# if __name__ == '__main__':
#     python = "C:/Users/sukhoon.jung/PycharmProjects/stable-baselines3/venv/Scripts/python.exe"
#     models = ["DDPG", "SAC", "TD#"]
#     procs = []
#     for m in models:
#         procs.append(subprocess.Popen([python,  "RUN.py", m], stdout=subprocess.PIPE))
#
#
#
#
