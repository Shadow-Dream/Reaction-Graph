import pandas as pd
import json
from tqdm import tqdm
dataset = pd.read_csv("condition/stage6/dataset.csv")
dataset.fillna("",inplace=True)
dataset = dataset.drop_duplicates()
dataset.to_csv("condition/stage7/dataset.csv",index=False)
# #dataset有catalyst1，solvent1，solvent2，agent1， agent2列
# #统计catalyst1为""，其余只有一个不为""的行数
# print(len(dataset[~((dataset["catalyst1"]=="") & (
#                    ((dataset["solvent1"]!="") & (dataset["solvent2"]=="") & (dataset["agent1"]=="") & (dataset["agent2"]=="")) |
#                    ((dataset["solvent1"]=="") & (dataset["solvent2"]!="") & (dataset["agent1"]=="") & (dataset["agent2"]=="")) |
#                    ((dataset["solvent1"]=="") & (dataset["solvent2"]=="") & (dataset["agent1"]!="") & (dataset["agent2"]=="")) |
#                    ((dataset["solvent1"]=="") & (dataset["solvent2"]=="") & (dataset["agent1"]=="") & (dataset["agent2"]!="")) |

#                    ((dataset["solvent1"]!="") & (dataset["solvent2"]!="") & (dataset["agent1"]=="") & (dataset["agent2"]=="")) |
#                    ((dataset["solvent1"]!="") & (dataset["solvent2"]=="") & (dataset["agent1"]!="") & (dataset["agent2"]=="")) |
#                    ((dataset["solvent1"]!="") & (dataset["solvent2"]=="") & (dataset["agent1"]=="") & (dataset["agent2"]!="")) |

#                    ((dataset["solvent1"]!="") & (dataset["solvent2"]!="") & (dataset["agent1"]=="") & (dataset["agent2"]=="")) |
#                    ((dataset["solvent1"]=="") & (dataset["solvent2"]!="") & (dataset["agent1"]!="") & (dataset["agent2"]=="")) |
#                    ((dataset["solvent1"]=="") & (dataset["solvent2"]!="") & (dataset["agent1"]=="") & (dataset["agent2"]!="")) |

#                    ((dataset["solvent1"]!="") & (dataset["solvent2"]=="") & (dataset["agent1"]!="") & (dataset["agent2"]=="")) |
#                    ((dataset["solvent1"]=="") & (dataset["solvent2"]!="") & (dataset["agent1"]!="") & (dataset["agent2"]=="")) |
#                    ((dataset["solvent1"]=="") & (dataset["solvent2"]=="") & (dataset["agent1"]!="") & (dataset["agent2"]!="")) |

#                    ((dataset["solvent1"]!="") & (dataset["solvent2"]=="") & (dataset["agent1"]=="") & (dataset["agent2"]!="")) |
#                    ((dataset["solvent1"]=="") & (dataset["solvent2"]!="") & (dataset["agent1"]=="") & (dataset["agent2"]!="")) |
#                    ((dataset["solvent1"]=="") & (dataset["solvent2"]=="") & (dataset["agent1"]!="") & (dataset["agent2"]!=""))
#                    )) ]))