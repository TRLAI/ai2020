import math
while True:
    try:
        input_profit = input('请输入利润（单位：万），按q退出：')
        if input_profit == 'q':
            break
        input_profit = float(input_profit)
        if input_profit <= 10:
            bonus = input_profit * 0.1
        elif input_profit > 10 and input_profit <= 20:
            bonus = 1 + (input_profit - 10) * 0.075
        elif input_profit > 20 and input_profit <= 40:
            bonus = 1.75 + (input_profit - 20) * 0.05
        elif input_profit > 40 and input_profit <= 60:
            bonus = 2.75 + (input_profit - 40) * 0.03
        elif input_profit > 60 and input_profit <= 100:
            bonus = 3.35 + (input_profit - 60) * 0.015
        elif input_profit > 100:
            bonus = 3.95 + (input_profit - 100) * 0.01
        print('应发放奖金：{}'.format(bonus))
    except ValueError:
        print('请输入正确的数字')
