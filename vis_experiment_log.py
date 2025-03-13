import glob
import os
import re
import pandas as pd
from IPython.display import display


def parse_memo_txt(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    sections = re.split(r'\n\s*\n', content.strip())

    memo = sections[0].strip()

    params_pattern = r'"([\w_]+)":\s*(-?\d+\.?\d*)'
    params_matches = re.findall(params_pattern, content)
    params = {k: float(v) for k, v in params_matches}

    result = ''
    plan = ''

    for sec in sections[1:]:
        if sec.lower().startswith("result:"):
            result = sec.replace('Result:', '').strip()
        elif sec.lower().startswith('plan:'):
            plan = sec.replace('Plan:', '').strip()

    params['memo'] = memo
    params['result'] = result
    params['plan'] = plan
    params['experiment'] = os.path.basename(os.path.dirname(filepath))

    return params


base_dir = os.path.expanduser('~/Genesis/logs/go2_slosh_free_v2/')
memo_files = glob.glob(os.path.join(base_dir, '*/memo.txt'))

print("Found memo files:")
for f in memo_files:
    print(f)

records = [parse_memo_txt(path) for path in memo_files]

# Dynamically create DataFrame columns
df = pd.DataFrame(records)

# Always put these columns first if they exist
fixed_cols_order = ['experiment', 'memo', 'result', 'plan']
dynamic_cols = [col for col in df.columns if col not in fixed_cols_order]

# Reorder DataFrame columns
cols_order = fixed_cols_order + sorted(dynamic_cols)
df = df[[col for col in cols_order if col in df.columns]]

# Display DataFrame
print(df.to_markdown(index=False))

# Save outputs
output_dir = os.path.expanduser('~/Genesis/logs/go2_slosh_free_v2/')
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, 'experiment_summary.csv'), index=False)
df.to_excel(os.path.join(output_dir, 'experiment_summary.xlsx'), index=False)