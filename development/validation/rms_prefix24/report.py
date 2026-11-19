"""Verify completed frozen sequences from the vendor's flat OPPROF archive."""
from physical import measure
from campaign import CHIPS, OUT, PLAN, RAW, ROOT, host, json, verify, write

def metrics(rows, key):
    errors = [abs(r[key]-r['ca_us']) for r in rows]
    return dict(n=len(rows), wape_pct=100*sum(errors)/sum(r['ca_us'] for r in rows),
                max_abs_pct=100*max(e/r['ca_us'] for e,r in zip(errors,rows)))

def report():
    plan = json.loads(PLAN.read_text())
    eligible = []
    batches = []
    future = {(r['group'],r['i']):r['partial_us'] for r in plan['rows'] if r['chip']==CHIPS[2]}
    for group in plan['groups']:
        for chip in plan['chip_order']:
            name = group+'-'+chip
            dest = OUT/(name+'.json')
            if not dest.is_file():
                continue
            original = json.loads(dest.read_text())
            folder = RAW/name
            jobs = [r for r in plan['rows'] if (r['chip'],r['group'])==(chip,group)]
            receipt = OUT/(name+'-windows.json')
            if not receipt.exists():
                cost = original['cost']
                measured = []
                rev = dict(registered=verify.digest(PLAN), original_receipt=dest.name, rows=measured)
                try:
                    verify.checked(not cost['stop_reason'] and cost['exit_code']==0 and cost['started_utc']>plan['utc'],'process completion')
                    expected_inputs=''.join(f"rmsnorm {r['i']} {r['shape'][0]} {r['shape'][1]} 0\n" for r in jobs)
                    verify.checked((folder/'inputs.txt').read_text()==expected_inputs,'input binding')
                    tasks,tables = host(folder,chip,jobs)
                    profile, = folder.glob('profile/OPPROF_*/'+verify.KERNEL)
                    elf = verify.text_bytes(verify.VENDOR)
                    verify.checked(all(len(t)==j['active_aiv'] for t,j in zip(tables,jobs)), 'prefix active roster')
                    previous_end = 0
                    rev['failed'] = []
                    for j,task,table in zip(jobs,tasks,tables):
                        launch = profile/str(j['i'])
                        try:
                            physical = measure(launch,chip,j,table,elf)
                            verify.checked(physical['first_tick']>previous_end,'physical launch order')
                            previous_end = physical['last_tick']
                        except (ValueError,OSError,IndexError,KeyError,TypeError) as err:
                            rev['failed'].append(dict(i=j['i'],reason=repr(err)))
                            continue
                        try:
                            row = verify.launch(launch,chip,j,task,table,elf)
                            strict = row['stage']['ca_us']
                            strict_error = None
                        except (ValueError,OSError,IndexError,KeyError,TypeError) as err:
                            strict, strict_error = None, str(err)
                        measured.append(dict(**j,**physical,strict_label_us=strict,strict_error=strict_error,
                                             goldens=task['golden'],future_partial_us=future[(group,j['i'])]))
                except (ValueError,OSError,IndexError,KeyError,TypeError) as err:
                    rev['error']=repr(err)
                write(receipt,rev)
            rev = json.loads(receipt.read_text())
            batches.append(dict(chip=chip,group=group,wall_s=original['cost']['seconds'],windows=len(rev['rows']),reason=rev.get('error')))
            eligible.extend(r for r in rev['rows'] if r['role']=='target')
    seen = {(r['group'],r['chip'],r['i']) for r in eligible}
    recaptures = []
    for path in sorted(OUT.glob('recovery-*-Ascend950*.json')):
        receipt = json.loads(path.read_text())
        r = receipt.get('target')
        recaptures.append(dict(receipt=path.name,qualified=bool(r),wall_s=receipt['cost']['seconds']))
        if r is not None:
            assert (r['group'],r['chip'],r['i']) not in seen and r['i']==receipt['original_failure']['i']
            eligible.append(r)
            seen.add((r['group'],r['chip'],r['i']))
    summary = {c:{k:metrics(rs,k) for k in ('previous_partial_us','partial_us','full_us')} for c in CHIPS[:2]
               if (rs:=[r for r in eligible if r['chip']==c])}
    strict = {c:{k:metrics(rs,k) for k in ('previous_partial_us','partial_us','full_us')} for c in CHIPS[:2]
              if (rs:=[dict(r,ca_us=r['strict_label_us']) for r in eligible if r['chip']==c and r['strict_label_us'] is not None])}
    by_key = {(r['chip'],tuple(r['shape'])):r for r in eligible}
    pairs = []
    for (chip,shape),a in by_key.items():
        b = by_key.get((CHIPS[1],shape))
        if chip==CHIPS[0] and b:
            pairs.append(dict(shape=shape,**{k:a[k]/b[k] for k in ('previous_partial_us','partial_us','full_us','ca_us')}))
    ratios = {k:metrics(pairs,k) for k in ('previous_partial_us','partial_us','full_us')} if pairs else {}
    snapshot = dict(batches=batches,recaptures=recaptures,summary=summary,strict= strict,dt_over_pr=ratios,pairs=pairs,rows=eligible)
    write(OUT/'evaluation.json',snapshot)
    lines=['# RMSNorm 可见前序的冻结独立检验','',
           '| 芯片 | 主设备窗口 | shape 数 | v0.23 部分前序 WAPE / 最大 | v0.24 同集 | 完整前序同集 |',
           '|---|---|---:|---:|---:|---:|']
    def pair(s): return f"{s['wape_pct']:.2f}% / {s['max_abs_pct']:.2f}%"
    for label,grouped in [('物理退休',summary),('含导出终止标签',strict)]:
        for c,values in grouped.items():
            old,new,full = [values[k] for k in ('previous_partial_us','partial_us','full_us')]
            lines.append(f"| {c} | {label} | {old['n']} | {pair(old)} | {pair(new)} | {pair(full)} |")
    lines+=['','每个进程的第一次调用对调度器隐藏，第二次可见但不计分，之后所有目标均在冻结后实测。窄起步和多行组在扩核过程中校验；宽起步组先隐藏一次 55 核调用。部分前序列只传可见调用，完整前序列也传隐藏调用用于参照。970DT 仅按配置模拟相同可见顺序。','',
        '各数值为 µs。两个 950 的主物理口径从全核首指令到最后物理指令退休。单独复算所有核都有 END_LABEL 的子集，与冻结的旧严格口径对照。两种口径共用实测窗口，样本数各自统计。', '',
        '| 芯片 | shape [T,D] | 已知下界 L / 实际前序 H | v0.23 / v0.24 | 物理 CA | 含标签 CA | 相对误差 | 970DT 外推 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in eligible:
        tagged = '—' if r['strict_label_us'] is None else f"{r['strict_label_us']:.3f}"
        lines.append(f"| {r['chip'][6:11]} | {r['shape']} | {r['observed_lower']} / {r['true_H']} | {r['previous_partial_us']:.3f} / {r['partial_us']:.3f} | {r['ca_us']:.3f} | {tagged} | {100*(r['partial_us']/r['ca_us']-1):+.2f}% | {r['future_partial_us']:.3f} |")
    if ratios:
        lines+=['',f"双端均合格的 {len(pairs)} 个形状按 DT 耗时除以 PR 耗时比较加速比，旧/新 WAPE 和最大偏差分别为 {pair(ratios['previous_partial_us'])}、{pair(ratios['partial_us'])}。差值接近零的样本不能可靠判断哪张卡更快。"]
    cost=(sum(b['wall_s'] for b in batches)+sum(b['wall_s'] for b in recaptures))/60
    lines+=['',f'已收回 {len(batches)}/6 个原进程、{len(recaptures)} 个同前序补采回执，累计 CA 墙钟 {cost:.2f} 分钟。冻结清单包括测量前的时间点、10 个目标、2 个 950 的推理和 970 情景。每次调用校验全部 y 与 rstd、实际 `aclnnRmsNorm`、BF16 库 ELF 和硬件配置。核分片与聚合 trace 对上全部非等待物理指令，实际写字节与输入形状一致。初版采集器查找了含 device0 的导出目录，本机实际为扁平目录。图与评分直接复核留存的相同原始文件，未因此重跑 CA。[冻结、逐窗证据与开销](evaluation.json) · [采集前登记](freeze.json)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(summary)

if __name__=='__main__':
    report()
