import pandas as pd
import matplotlib.pyplot as plt

MAP_dataframe = pd.read_csv('MAPdataframe.txt', sep = ' ')
MPE_dataframe = pd.read_csv('MPEdataframe.txt', sep = ' ')
print(MAP_dataframe)
#MAP_dataframe['number of variables'] = [10,20,30,40,50,60,70,80,90,100]
# MAP_dataframe.plot(y = ['random (no heuristic)', 'min fill heurisic', 'min degree heuristic'], x = 'number of variables', style='.-')
# plt.title('Performance of MAP')
# plt.ylabel('Time (in seconds)')
# plt.xlabel('Number of variables')
# plt.show()
MPE_dataframe['number of variables'] = [10,20,30,40,50,60,70,80,90,100]
MPE_dataframe.plot(y = ['random (no heuristic)', 'min fill heurisic', 'min degree heuristic'], x = 'number of variables', style='.-')
plt.title('Performance of MPE')
plt.ylabel('Time (in seconds)')
plt.xlabel('Number of variables')
plt.show()