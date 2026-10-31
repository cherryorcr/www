"""Scan batch size for arxiv (only large graph available in DATA_CFG)."""
import sys, torch, numpy as np, json
sys.path.insert(0, '/home/pc/research/TAG_condensation/code/v2_codex')

device = 'cuda:0'
dataset = 'arxiv'

print(f"Loading {dataset}...")
# Use the actual load_data function from run_multiscale_asm.py
from run_multiscale_asm import load_data
x, texts, y, edges, splits = load_data(dataset)

dev_mask = splits['dev']
n = len(x)
print(f"n={n} dev={dev_mask.sum()} edges={edges.shape[1]}")

# Build initial memory
from run_multiscale_asm import build_initial, CLIPStudent
k = max(int(n * 0.05), 10)
syn = build_initial(x, texts, y, edges, k=k, text_emb_cache={}, device=device)
print(f"k={len(syn['x'])}")

results = []
for batch in [16, 24, 32, 48, 64, 96]:
    if batch > len(syn['x']):
        continue

    f1_list = []
    for seed in [200, 201, 202]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = CLIPStudent(syn['x'].shape[1]).to(device)

        # Train
        from run_multiscale_asm import train_student
        model = train_student(syn, seed, device,
                            steps=min(2048, 16*len(syn['x'])),
                            batch=batch, lr=1e-3)

        # Eval
        with torch.no_grad():
            x_dev = torch.tensor(x[dev_mask], dtype=torch.float32).to(device)
            out = model(x_dev, syn['t'].to(device), syn['x'].to(device))[0]
            pred = out.argmax(1).cpu().numpy()
            y_dev = y[dev_mask]
            from sklearn.metrics import f1_score
            f1 = f1_score(y_dev, pred, average='macro', zero_division=0) * 100
        f1_list.append(f1)

    mean_f1 = np.mean(f1_list)
    print(f"batch={batch:3d}: F1={mean_f1:.2f}")
    results.append({"batch": batch, "f1_mean": mean_f1, "f1_list": f1_list})

best = max(results, key=lambda r: r['f1_mean'])
print(f"\nBest: batch={best['batch']} F1={best['f1_mean']:.2f}")

with open('/home/pc/research/problem_validation_results/arxiv_batch_scan.json', 'w') as f:
    json.dump(results, f, indent=2)
print("DONE")
