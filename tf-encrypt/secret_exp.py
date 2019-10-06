import random
import numpy as np

def share(secret):
  # share_0 will become some random number
  # print(0, secret.shape[0])
  share_0 = random.uniform(0, secret.shape[0])
  # create share_1 by subtracting the secret from the random number
  # this means we can recreate the secret by adding the two shares together
  share_1 = secret - share_0
  
  return share_0, share_1

# example usage
secret = np.array([1,2,3])
share_0, share_1 = share(secret)

print(share_0)
print(share_1)

# Sample data, the shares become large random numbers
# share_0 = [48392048932, 92983012832, 38920494829]
# share_1 = [77473819238, 38291029389, 18283782912]