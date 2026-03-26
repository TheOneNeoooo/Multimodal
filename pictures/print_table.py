import numpy as np
import config, generate_plots

print('='*70)
print('新配置下的平均指标表格')
print('='*70)

metrics = ['EN', 'Qabf', 'SCD', 'SD', 'VIF']
results = {}

for metric in metrics:
    x, data, cfg = generate_plots.generate_metric_data(metric)
    row = {}
    for method in config.METHODS:
        row[method] = np.mean(data[method])
    results[metric] = row

# 打印表格
header = '方法        ' + ''.join([f'{m:>10}' for m in config.METHODS])
print(header)
print('-'*70)

for metric in metrics:
    row_name = metric
    values = ''.join([f'{results[metric][m]:>10.3f}' for m in config.METHODS])
    print(f'{row_name:<12}{values}')

print()
print('突出指标: EN, Qabf')
print('持平/略低: SCD, SD, VIF')
