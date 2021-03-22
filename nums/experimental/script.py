from collections import Counter
import socket
import time

import ray
import os

# if os.environ.get('https_proxy'):
#     del os.environ['https_proxy']
# if os.environ.get('http_proxy'):
#     del os.environ['http_proxy']

ray.init(address='auto', _redis_password='5241590000000000')
# ray.init(address='auto')
print('''This cluster consists of
    {} nodes in total
    {} GPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()))

@ray.remote
def f():
    time.sleep(0.002)
    # Return IP address.
    return socket.gethostbyname(socket.gethostname())

# # Get a list of the IP addresses of the nodes that have joined the cluster.
# set(ray.get([f.remote() for _ in range(1000)]))
object_ids = [f.remote() for _ in range(10000)]
ip_addresses = ray.get(object_ids)

print('Tasks executed')
for ip_address, num_tasks in Counter(ip_addresses).items():
    print('    {} tasks on {}'.format(num_tasks, ip_address))