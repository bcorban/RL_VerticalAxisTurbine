import ray 
ray.init()
ressources = ray.available_resources()
print(ressources)

import time 

def usefull(n):
  count=0
  for _ in range(n):
    count+=1
  return count

n=int(4e8)
t0=time.time()
[usefull(n) for k in range(2)]
print("It takes {}s to count 2 times up to {} without ray".format(time.time()-t0,n))


@ray.remote
def usefull(n):
  count=0
  for _ in range(n):
    count+=1
  return count

n=int(4e8)
t0=time.time()
ray.get([usefull.remote(n) for k in range(2)])
print("It takes {}s to count 2 times up to {} with ray".format(time.time()-t0,n))