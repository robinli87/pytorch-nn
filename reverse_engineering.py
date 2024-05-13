import torch.multiprocessing as mp
import torch
import copy

try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# myarray.share_memory_()

total = torch.zeros(1, 2).cuda()
total.share_memory_()


def train(procnum, return_dict):
    # myarray[0] = (value1 + value2)
    # print("hello I am trying")
    myarray = torch.normal(0, 1, (1, 2)).cuda()
    clone = copy.deepcopy(myarray)
    return_dict[procnum] = [procnum, clone]
    # rint("I am done")
    return (myarray)


def submit(result):
    total = torch.add(total, myarray)


num_processes = 4
total = []


def iterable():
    manager = mp.Manager()
    return_dict = manager.dict()

    # Create a list of processes and start each process with the train function
    processes = []

    for i in range(0, 5):

        p = mp.Process(target=train, args=(i, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        p.close()

    print(return_dict.values())

    # print(myarray)
if __name__ == "__main__":
    for i in range(0, 10):
        iterable()
else:
    # print("entry not main")
    pass
