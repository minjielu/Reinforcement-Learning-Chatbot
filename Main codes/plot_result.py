
import matplotlib.pyplot as plt
import numpy as np

'''
loss_acc = np.loadtxt('./logs/RI_history_reward_baseline_1.csv')

loss = loss_acc[0::2]
acc = loss_acc[1::2]
batch_num = np.arange(1,loss.shape[0]+1)

fig, ax = plt.subplots(1,2)
ax[0].plot(batch_num, loss, color = 'blue')
ax[0].set_xlabel('Batch # (*10)')
ax[0].set_ylabel('Loss')
ax[0].set_title('Training loss')


ax[1].plot(batch_num[::10], acc[::10], color = 'blue')
ax[1].set_xlabel('Batch # (*10)')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Training accuracy')

plt.show()
'''

score = np.loadtxt('./logs/Critic_history_1.csv')

batch_num = np.arange(1,score.shape[0]+1)*10

fig, ax = plt.subplots()
ax.plot(batch_num, score, color = 'blue')
ax.set_xlabel('Episode #')
ax.set_ylabel('Sum of the squared advantages')
ax.set_title('Training performance of the critic')

plt.show()
