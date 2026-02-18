import matplotlib.pyplot as plt
import numpy as np

#  - 10%
plt.style.use('classic')
plt.rcParams.update({
    'font.family': 'sans-serif',
    # 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 26,        # 2410%26
    'axes.titlesize': 32,   # 2910%32
    'axes.labelsize': 29,   # 2610%29
    'xtick.labelsize': 22,  # 2210%24
    'ytick.labelsize': 22,  # 2210%24
    'legend.fontsize': 26,  # 2410%26
    'axes.linewidth': 1.5,
    'axes.edgecolor': 'black',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'grid.linewidth': 0.8,
    # 'text.usetex': True,
    # 'mathtext.default': 'regular',
})

import matplotlib
matplotlib.rcParams.update({
    "pdf.fonttype": 42,   #  TrueType (Type 42) Type 3
    "ps.fonttype": 42,
})

#  - 
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')

#  - A.RTL1PX.TAKEOFF2
test_cases = ['A.RTL1', 'A.RTL2', 'A.FLIP2', 'A.FLIP3', 'A.ALT_HOLD2',
              'A.LAND1', 'A.LAND2', 'A.BRAKE1', 'A.LOITER', 'A.GUIDED',
              'PX.RTL2', 'PX.HOLD1', 'PX.TAKEOFF1', 'PX.TAKEOFF2']

pgfuzz_values = [196, 270, 63, 73, 882, 219, 69, 528, 114, 17, 133, 402, 442, 1382]
cpg_values = [20, 225, 63, 45, 1, 0, 2, 249, 2, 7, 33, 239, 0, 78]

# bar
bar_width = 0.35
x_pos = np.arange(len(test_cases))

#  - PGFuzzCPG
bars1 = ax.bar(x_pos - bar_width/2,  # PGFuzz
               [max(val, 5) for val in pgfuzz_values], 
               bar_width, 
               label='PGFuzz MTL', color='white', alpha=1.0, 
               edgecolor='black', linewidth=1.5, hatch='///')
bars2 = ax.bar(x_pos + bar_width/2,  # CPG
               [max(val, 5) for val in cpg_values],
               bar_width,
               label=r'RVSpec MTL', color='white', alpha=1.0,
               edgecolor='black', linewidth=1.5)

#  - 10%
ax.set_ylabel('# of False Positives', fontsize=24)  # 2629

# 
ax.set_title('')

# x - x
ax.set_xticks(x_pos + 0.4)  # 0.4
ax.set_xticklabels(test_cases, fontsize=30, rotation=45, ha='right', fontweight='normal')
ax.set_xlim(-0.5, len(test_cases) - 0.5)

# y - y
max_val = max(max(pgfuzz_values), max(cpg_values))
ax.set_ylim(0, max_val * 1.05)  # 1.11.05

# y - 
ax.set_yticks(np.arange(0, max_val * 1.05 + 1, 400))  

#  - 
legend = ax.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15), fontsize=24, frameon=True,
          fancybox=False, shadow=False, framealpha=0.95,
          edgecolor='black', facecolor='white')
legend.get_frame().set_linewidth(1.2)

# ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray', axis='y')
ax.set_axisbelow(True)

#  - 
ax.spines['top'].set_visible(False)      # 
ax.spines['right'].set_visible(False)    # 
ax.spines['left'].set_linewidth(1.5)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['bottom'].set_color('black')

#  - 
ax.tick_params(axis='both', which='major', direction='in', width=1.2, length=6)
ax.tick_params(axis='y', labelsize=24, left=True, right=False)  # y
ax.tick_params(axis='x', labelsize=22, bottom=True, top=False)  # x

#  - 
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    pgfuzz_val = pgfuzz_values[i]
    cpg_val = cpg_values[i]
    
    # PGFuzz MTL - 
    ax.text(bar1.get_x() + bar1.get_width()/2,
            max(pgfuzz_val, 5) + max_val*0.02,  # 
            f'{pgfuzz_val}', ha='center', va='bottom',
            fontsize=18, fontweight='bold', color='black', rotation=90)  # 
    
    # CPG MTL - 
    ax.text(bar2.get_x() + bar2.get_width()/2,
            max(cpg_val, 5) + max_val*0.02,    # 
            f'{cpg_val}', ha='center', va='bottom',
            fontsize=18, fontweight='bold', color='black', rotation=90)  # 

plt.tight_layout()

# PDF
plt.savefig('false_positive_comparison.pdf', format='pdf', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("Figure saved as 'false_positive_comparison.pdf'")
plt.show()