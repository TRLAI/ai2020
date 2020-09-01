profit = int(input('净利润：万元'))
bonus = 0
thresholds = [0,10,20,40,60,100]
rates = [0.1,0.075,0.05,0.03,0.015,0.01]
for i in range(1,len(thresholds)):
    if profit < thresholds[i]:
        bonus += (profit - thresholds[i-1]) * rates[i-1]
        break
    else:
        bonus += (thresholds[i]-thresholds[i-1]) * rates[i-1]
else:
    bonus += (profit - thresholds[-1])* rates [-1]
print(bonus)