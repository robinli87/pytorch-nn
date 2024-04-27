import torch.multiprocessing as mp
import torch


myarray = torch.zeros(5).cuda()
myarray.share_memory_()

def train(value1, value2):
    myarray[rank] = (value1+ value2)

num_processes = 4

#if __name__ == "__main__":
# Create a list of processes and start each process with the train function
processes = []
for rank in range(num_processes):
    p = mp.Process(target=train, args=(rank, rank))
    p.start()
    processes.append(p)
    print(f"Started {p.name}")

# Wait for all processes to finish
for p in processes:
    p.join()
    print(f"Finished {p.name}")

print(myarray)