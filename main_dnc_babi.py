from controller_implementations.dnc.dnc import *
from controller_implementations.feedforward import *
from controller_implementations.lstm import *

from task_implementations.copy_task import *
from task_implementations.bAbI.bAbI import *
from utils import project_path, init_wrapper

weight_initializer = init_wrapper(tf.random_normal)

n_blocks = 6
vector_size = n_blocks + 1
min_seq = 5
train_max_seq = 6
n_copies = 1
out_vector_size = vector_size


# Task | DNC | DeepMind's DNC | LSTM 256 | LSTM 512 | DeepMind LSTM 512 |  for BABI

# xsm choose Copy and repeat copy tasks and babi task
#task = CopyTask(vector_size, min_seq, train_max_seq, n_copies)    #Copy and repeat copy tasks
task = bAbITask(os.path.join("tasks_1-20_v1-2", "en-10k"))     # babi task


print("Loaded task")


# XSM TASK ITERATIONS TIMES 

class Hp:
    """
    Hyperparameters
    """
    batch_size = 4
#    steps = 1000000
#    steps = 200000
    steps = 6000

    lstm_memory_size = 128
    n_layers = 1

    stddev = 0.1

    class Mem:
        word_size = 8
        mem_size = 10
        num_read_heads = 1

print("DNC BABI TASK")
print("batch_size,steps,lstm_memory_size,n_layers,stddev", Hp.batch_size,Hp.steps,Hp.lstm_memory_size,Hp.n_layers,Hp.stddev)
print("word_size,mem_size,num_read_heads", Hp.Mem.word_size,Hp.Mem.mem_size,Hp.Mem.num_read_heads)
# choose dnc or lstm--2



controller = Feedforward(task.vector_size, Hp.batch_size, [256, 512])   #   1 dnc  *1
print("controller = Feedforward")





#controller = LSTM(task.vector_size, Hp.lstm_memory_size, Hp.n_layers, initializer=weight_initializer,initial_stddev=Hp.stddev)    # 2lstm  *2
#print("controller = LSTM")





out_vector_size=task.vector_size # 2 lstmis a needed argument to LSTM if you want to test just the LSTM outside of DNC  *3


dnc = DNC(controller, Hp.batch_size, task.vector_size, Hp.Mem, initializer=weight_initializer, initial_stddev=Hp.stddev)  #1  *4

print("Loaded controller")

restore_path = os.path.join(project_path.log_dir, "June_09__19:32", "train", "model.chpt")   #  2 lstm  *5
dnc.run_session(task, Hp, project_path)  # , restore_path=restore_path)   # 1   *6
