import random
random.seed(42)
import bisect
import copy
from queue import PriorityQueue
from collections import deque


class MyData:
    def __init__(self, item):
        self.weight = item[0]
        self.data = item[1]

    def __lt__(self, other):
        return self.weight < other.weight

    def __str__(self):
        return '{} {}'.format(self.weight, self.data)

    def __repr__(self):
        return '({}, {})'.format(self.weight, self.data)


class Memory:
    def __init__(self, max_size_per_class, num_classes):
        self.max_size_per_class = max_size_per_class
        self.memory = [None] * num_classes

    def update(self, data):
        data = copy.deepcopy(data)
        for data_dict in data:
            for obj in data_dict['annotations']:
                category_id = obj['category_id']
                if self.memory[category_id] is None:
                    self.memory[category_id] = [data_dict]
                else:
                    if data_dict in self.memory[category_id]:
                        continue
                    if len(self.memory[category_id]) < self.max_size_per_class:
                        self.memory[category_id].append(data_dict)
                    else:
                        del_idx = random.randint(0, self.max_size_per_class)
                        if del_idx != self.max_size_per_class:
                            self.memory[category_id][del_idx] = data_dict

    def retrieve(self, max_num=16, samples_per_class=10):
        memory_list = []
        for cls_memory in self.memory:
            if cls_memory is not None:
                memory_list += random.sample(cls_memory, min(len(cls_memory), samples_per_class))
                # memory_list.append(random.choice(cls_memory))  # randomly choice one sample per class
        if len(memory_list) > max_num:
            memory_list = random.sample(memory_list, max_num)
        else:
            if len(memory_list) > 0:
                memory_list = (memory_list * (max_num // len(memory_list) + 1))[:max_num]
        return memory_list


class RandomMemory:
    def __init__(self, max_size_per_class, num_classes):
        self.memory = []
        self.max_size = max_size_per_class * num_classes

    def update(self, data):
        data = copy.deepcopy(data)
        for data_dict in data:
            if len(self.memory) < self.max_size:
                self.memory.append(data_dict)
            else:
                del_idx = random.randint(0, self.max_size)
                if del_idx != self.max_size:
                    self.memory[del_idx] = data_dict

    def retrieve(self, max_num=16, samples_per_class=10):
        memory_list = self.memory
        if len(memory_list) > max_num:
            memory_list = random.sample(memory_list, max_num)
        else:
            if len(memory_list) > 0:
                memory_list = (memory_list * (max_num // len(memory_list) + 1))[:max_num]
        return memory_list


class PriorityMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)    # [(loss, item), ...]

    def full(self):
        return len(self.queue) == self.max_size

    def update(self, items):
        for item in items:
            if self.full():
                if item[0] > self.queue[0].weight:
                    self.queue.popleft()
                    bisect.insort(self.queue, MyData(item))
            else:
                bisect.insort(self.queue, MyData(item))

    def retrieve(self, n, mode='max'):
        n = min(len(self.queue), n)
        items = []
        if mode == 'random':
            for _ in range(n):
                ind = random.choice(range(len(self.queue)))
                item = self.queue[ind]
                items.append((item.weight, item.data))
                del self.queue[ind]
        elif mode == 'max':
            for _ in range(n):
                item = self.queue.pop()
                items.append((item.weight, item.data))
        elif mode == 'min':
            for _ in range(n):
                item = self.queue.popleft()
                items.append((item.weight, item.data))
        elif mode == 'class_random':
            cls2ind = {}
            for ind, x in enumerate(self.queue):
                for obj in x.data['annotations']:
                    cls2ind[obj['category_id']] = cls2ind.get(obj['category_id'], []) + [ind]
            cls2ind = {k: list(set(v)) for k, v, in cls2ind.items()}
            inds = []
            for cls in cls2ind.keys():
                unselected = [x for x in cls2ind[cls] if x not in inds]
                inds += random.sample(unselected, min(len(unselected), 2))
            inds = random.sample(inds, min(len(inds), n))
            for ind in sorted(inds, reverse=True):
                item = self.queue[ind]
                items.append((item.weight, item.data))
                del self.queue[ind]
        elif mode == 'class_sorted':
            cls2ind = {}
            for ind, x in enumerate(self.queue):
                for obj in x.data['annotations']:
                    cls2ind[obj['category_id']] = cls2ind.get(obj['category_id'], []) + [ind]
            cls2ind = {k: list(set(v)) for k, v, in cls2ind.items()}
            inds = []
            for cls in cls2ind.keys():
                unselected = [x for x in cls2ind[cls] if x not in inds]
                inds += unselected[-2:]   
            inds = sorted(inds, key=lambda x: self.queue[x].weight, reverse=True)[:n] 
            for ind in sorted(inds, reverse=True):
                item = self.queue[ind]
                items.append((item.weight, item.data))
                del self.queue[ind]
        else:
            raise NotImplementedError('mode {} is not implemented'.format(mode))

        return items


if __name__ == '__main__':
    memory = PriorityMemory(max_size=5)

    values = [(1, {'annotations': [{'category_id': 14}, {'category_id': 18}]}),
              (5, {'annotations': [{'category_id': 14}, {'category_id': 18}]}),
              (3, {'annotations': [{'category_id': 18}]}),
              (2, {'annotations': [{'category_id': 20}]}),
              (4, {'annotations': [{'category_id': 21}]})]
    memory.update(values)
    replay_data = memory.retrieve(3, mode='class_sorted')
    print(replay_data)
    for i in range(len(replay_data)):
        tmp = replay_data[i]
        tmp = (tmp[0]-10, tmp[1])
        replay_data[i] = tmp
    memory.update(replay_data)
    print(memory.queue)
